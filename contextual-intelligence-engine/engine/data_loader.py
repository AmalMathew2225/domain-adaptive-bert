"""
Data loading and preprocessing module.
Handles tokenization and PyTorch Dataset/DataLoader creation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from .config import EngineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Classes
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """Dataset for sentence/sequence classification tasks."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class NERDataset(Dataset):
    """Dataset for token-level Named Entity Recognition tasks."""

    def __init__(
        self,
        tokenized_inputs: Dict[str, torch.Tensor],
        labels: List[List[int]],
    ):
        self.encodings = tokenized_inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class MLMDataset(Dataset):
    """Dataset for Masked Language Modelling (domain-adaptive pretraining)."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.encodings["input_ids"][idx].clone()
        attention_mask = self.encodings["attention_mask"][idx]
        input_ids, labels = self._mask_tokens(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_tokens(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            ),
            dtype=torch.bool,
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only compute loss on masked tokens
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        return input_ids, labels


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

class ContextualDataLoader:
    """
    High-level factory that builds train/eval DataLoader objects.

    Usage:
        loader = ContextualDataLoader(config, tokenizer)
        train_dl, eval_dl = loader.build(train_texts, train_labels,
                                          eval_texts, eval_labels)
    """

    def __init__(self, config: EngineConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def build(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Build PyTorch DataLoaders for training (and optionally evaluation).

        Args:
            train_texts: List of raw training strings.
            train_labels: Corresponding integer labels.
            eval_texts: Optional list of validation strings.
            eval_labels: Optional corresponding validation labels.

        Returns:
            Tuple of (train_dataloader, eval_dataloader).
            eval_dataloader is None if eval data is not provided.
        """
        train_dataset = TextClassificationDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.config.max_length,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"Train DataLoader ready — {len(train_dataset)} samples.")

        eval_loader = None
        if eval_texts is not None and eval_labels is not None:
            eval_dataset = TextClassificationDataset(
                eval_texts,
                eval_labels,
                self.tokenizer,
                self.config.max_length,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            logger.info(f"Eval DataLoader ready — {len(eval_dataset)} samples.")

        return train_loader, eval_loader

    def from_hf_dataset(
        self,
        hf_dataset,
        text_column: str = "text",
        label_column: str = "label",
        split: str = "train",
    ) -> DataLoader:
        """
        Build a DataLoader directly from a HuggingFace datasets.Dataset object.

        Args:
            hf_dataset: A datasets.DatasetDict or datasets.Dataset instance.
            text_column: Name of the text column.
            label_column: Name of the label column.
            split: Which split to use ('train', 'validation', 'test').

        Returns:
            A PyTorch DataLoader.
        """
        ds = hf_dataset[split] if hasattr(hf_dataset, "__getitem__") else hf_dataset
        texts = ds[text_column]
        labels = ds[label_column]
        return self.build(texts, labels)[0]
