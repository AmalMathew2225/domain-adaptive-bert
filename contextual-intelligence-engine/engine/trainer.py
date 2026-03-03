"""
Trainer module for the Contextual Intelligence Engine.
Implements the full fine-tuning loop with evaluation and checkpointing.
"""

import logging
import os
import time
from typing import Optional, Dict, Callable

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import EngineConfig
from .model import ContextualModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Domain-adaptive fine-tuning trainer.

    Supports:
    - Linear warmup + linear decay LR scheduling
    - Gradient accumulation
    - Mixed-precision (fp16) training via torch.cuda.amp
    - Per-step logging and periodic checkpointing
    - Pluggable metric function

    Example:
        trainer = Trainer(ctx_model, config)
        trainer.train(train_loader, eval_loader=eval_loader)
    """

    def __init__(
        self,
        ctx_model: ContextualModel,
        config: EngineConfig,
        metric_fn: Optional[Callable] = None,
    ):
        """
        Args:
            ctx_model: Initialised ContextualModel instance.
            config: EngineConfig with all training hyperparameters.
            metric_fn: Optional callable(labels, preds) -> dict of metric names
                       to float values. Defaults to accuracy.
        """
        self.ctx_model = ctx_model
        self.model = ctx_model.model
        self.device = ctx_model.device
        self.config = config
        self.metric_fn = metric_fn or _default_accuracy

        self.optimizer = AdamW(
            self._get_param_groups(),
            lr=config.learning_rate,
            eps=1e-8,
        )
        self.scaler = GradScaler(enabled=config.fp16)
        self.global_step = 0
        self.best_eval_loss = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            eval_loader: Optional DataLoader for evaluation.

        Returns:
            Dict with final 'train_loss' and (if eval_loader) 'eval_loss'.
        """
        total_steps = (
            len(train_loader) // self.config.gradient_accumulation_steps
            * self.config.num_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(
            f"Starting training — epochs={self.config.num_epochs}, "
            f"total_steps={total_steps}, device={self.device}"
        )

        history: Dict[str, float] = {}

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, scheduler, epoch)
            history["train_loss"] = train_loss
            logger.info(f"[Epoch {epoch}] train_loss={train_loss:.4f}")

            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                history.update(eval_metrics)
                logger.info(f"[Epoch {epoch}] eval metrics: {eval_metrics}")

                if eval_metrics.get("eval_loss", float("inf")) < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.ctx_model.save(
                        os.path.join(self.config.output_dir, "best_checkpoint")
                    )
                    logger.info("New best checkpoint saved.")

        # Save final model
        self.ctx_model.save(os.path.join(self.config.output_dir, "final_model"))
        logger.info("Training complete. Final model saved.")
        return history

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        Args:
            eval_loader: DataLoader for evaluation data.

        Returns:
            Dict with at least 'eval_loss' and any metric_fn outputs.
        """
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                if hasattr(outputs, "logits"):
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    labels = batch["labels"].cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels)

        metrics = {"eval_loss": total_loss / max(total_samples, 1)}
        if all_labels:
            metrics.update(self.metric_fn(all_labels, all_preds))
        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss, total_samples = 0.0, 0
        self.optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.config.fp16):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.logging_steps == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"loss={loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                        f"elapsed={elapsed:.1f}s"
                    )

                if self.global_step % self.config.save_steps == 0:
                    ckpt_dir = os.path.join(
                        self.config.output_dir,
                        f"checkpoint-{self.global_step}",
                    )
                    self.ctx_model.save(ckpt_dir)

            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)

        return total_loss / max(total_samples, 1)

    def _get_param_groups(self):
        """Apply weight decay only to non-bias/norm parameters."""
        no_decay = {"bias", "LayerNorm.weight"}
        return [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]


# ------------------------------------------------------------------
# Default metric
# ------------------------------------------------------------------

def _default_accuracy(labels, preds) -> Dict[str, float]:
    correct = sum(l == p for l, p in zip(labels, preds))
    return {"accuracy": correct / len(labels) if labels else 0.0}
