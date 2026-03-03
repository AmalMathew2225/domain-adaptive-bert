"""
Inference module for the Contextual Intelligence Engine.
Provides a high-level pipeline for making predictions on new text.
"""

import logging
from typing import List, Dict, Union, Optional

import torch
import torch.nn.functional as F

from .config import EngineConfig
from .model import ContextualModel

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-level inference pipeline for the Contextual Intelligence Engine.

    Wraps a trained ContextualModel to provide simple predict/predict_batch APIs
    with optional probability outputs and label name resolution.

    Example:
        engine = InferenceEngine.from_pretrained("./outputs/best_checkpoint", config)
        results = engine.predict(["This is a great product!", "Terrible experience."])
        # [{'label': 'POSITIVE', 'score': 0.98}, {'label': 'NEGATIVE', 'score': 0.94}]
    """

    def __init__(
        self,
        ctx_model: ContextualModel,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            ctx_model: A trained ContextualModel instance.
            label_names: Optional list of human-readable label names,
                         indexed by class ID. If None, uses "LABEL_0", "LABEL_1", ...
        """
        self.ctx_model = ctx_model
        self.model = ctx_model.model
        self.tokenizer = ctx_model.tokenizer
        self.device = ctx_model.device
        self.config = ctx_model.config
        self.label_names = label_names or ctx_model.config.label_names
        self.model.eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: EngineConfig,
        label_names: Optional[List[str]] = None,
    ) -> "InferenceEngine":
        """
        Load a saved model and return a ready-to-use InferenceEngine.

        Args:
            model_path: Path to a directory with saved model artefacts.
            config: EngineConfig for the task type, device, etc.
            label_names: Optional human-readable label names.

        Returns:
            Initialised InferenceEngine.
        """
        ctx_model = ContextualModel.load(model_path, config)
        return cls(ctx_model, label_names=label_names)

    # ------------------------------------------------------------------
    # Public predict APIs
    # ------------------------------------------------------------------

    def predict(
        self,
        texts: Union[str, List[str]],
        return_all_scores: bool = False,
        batch_size: int = 32,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Run inference on one or more raw text inputs.

        Args:
            texts: A single string or a list of strings.
            return_all_scores: If True, include probability for every class.
            batch_size: Internal micro-batch size for large inputs.

        Returns:
            List of dicts, each with 'label' and 'score' keys.
            If return_all_scores=True, also includes 'all_scores' dict.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            all_results.extend(
                self._predict_batch(chunk, return_all_scores=return_all_scores)
            )
        return all_results

    def predict_proba(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Return raw softmax probability tensor for each input.

        Args:
            texts: A single string or a list of strings.

        Returns:
            Tensor of shape (N, num_labels).
        """
        if isinstance(texts, str):
            texts = [texts]

        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
        return F.softmax(logits, dim=-1).cpu()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_batch(
        self,
        texts: List[str],
        return_all_scores: bool = False,
    ) -> List[Dict]:
        probs = self.predict_proba(texts)  # (N, C)
        results = []

        for prob_vec in probs:
            pred_idx = int(prob_vec.argmax())
            label = self._resolve_label(pred_idx)
            entry: Dict = {
                "label": label,
                "score": round(prob_vec[pred_idx].item(), 4),
            }
            if return_all_scores:
                entry["all_scores"] = {
                    self._resolve_label(i): round(p.item(), 4)
                    for i, p in enumerate(prob_vec)
                }
            results.append(entry)

        return results

    def _resolve_label(self, idx: int) -> str:
        if self.label_names and idx < len(self.label_names):
            return self.label_names[idx]
        return f"LABEL_{idx}"
