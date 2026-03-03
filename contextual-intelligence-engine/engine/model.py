"""
Model module for the Contextual Intelligence Engine.
Wraps HuggingFace transformers with a clean, task-aware interface.
"""

import logging
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)

from .config import EngineConfig

logger = logging.getLogger(__name__)

# Map task types to HuggingFace model classes
_TASK_MODEL_MAP = {
    "classification": AutoModelForSequenceClassification,
    "ner": AutoModelForTokenClassification,
    "mlm": AutoModelForMaskedLM,
}


class ContextualModel:
    """
    A unified wrapper around HuggingFace transformer models.

    Handles model and tokenizer loading, device placement,
    and exposes a clean save/load interface.

    Example:
        config = EngineConfig(model_name="bert-base-uncased", num_labels=3)
        ctx_model = ContextualModel(config)
        model, tokenizer = ctx_model.model, ctx_model.tokenizer
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model, self.tokenizer = self._build(config)
        self.model.to(self.device)
        logger.info(
            f"Model '{config.model_name}' loaded for task='{config.task_type}' "
            f"on device='{self.device}'."
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Save model weights and tokenizer to disk.

        Args:
            path: Target directory. Falls back to config.output_dir.

        Returns:
            Absolute path where the model was saved.
        """
        save_dir = path or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model and tokenizer saved to '{save_dir}'.")
        return save_dir

    @classmethod
    def load(cls, path: str, config: EngineConfig) -> "ContextualModel":
        """
        Load a previously saved model from disk.

        Args:
            path: Directory containing saved model artefacts.
            config: EngineConfig describing task type and device.

        Returns:
            A fully initialised ContextualModel instance.
        """
        model_cls = _TASK_MODEL_MAP.get(config.task_type)
        if model_cls is None:
            raise ValueError(f"Unsupported task_type: '{config.task_type}'")

        instance = cls.__new__(cls)
        instance.config = config
        instance.device = cls._resolve_device(config.device)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = model_cls.from_pretrained(path)
        instance.model.to(instance.device)
        logger.info(f"Model loaded from '{path}' on device='{instance.device}'.")
        return instance

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable-only) parameter count."""
        params = (
            p for p in self.model.parameters()
            if (not trainable_only or p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def freeze_base(self, unfreeze_layers: int = 0) -> None:
        """
        Freeze all base encoder layers.

        Args:
            unfreeze_layers: If > 0, unfreeze the last N transformer blocks
                             while keeping the rest frozen. Useful for
                             gradual unfreezing strategies.
        """
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0 and hasattr(self.model, "bert"):
            encoder_layers = self.model.bert.encoder.layer
            for layer in encoder_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(
                f"Base frozen; last {unfreeze_layers} encoder layers unfrozen."
            )

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build(
        config: EngineConfig,
    ):
        """Instantiate model and tokenizer from HuggingFace Hub or local path."""
        model_cls = _TASK_MODEL_MAP.get(config.task_type)
        if model_cls is None:
            raise ValueError(
                f"Unsupported task_type '{config.task_type}'. "
                f"Choose from {list(_TASK_MODEL_MAP)}."
            )

        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        )
        model: PreTrainedModel = model_cls.from_pretrained(
            config.model_name,
            config=hf_config,
            ignore_mismatched_sizes=True,
        )
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            config.model_name
        )
        return model, tokenizer

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto" or device == "":
            return torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        return torch.device(device)
