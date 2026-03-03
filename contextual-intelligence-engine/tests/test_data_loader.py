"""
Tests for ContextualDataLoader.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from engine.config import EngineConfig
from engine.data_loader import (
    TextClassificationDataset,
    MLMDataset,
    ContextualDataLoader,
)


@pytest.fixture
def tokenizer():
    """Mock tokenizer that returns a simple encoding."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(4, 32, dtype=torch.long),
        "attention_mask": torch.ones(4, 32, dtype=torch.long),
    }
    tok.get_special_tokens_mask.return_value = [0] * 32
    tok.mask_token_id = 103
    return tok


@pytest.fixture
def sample_data():
    texts = ["Hello world", "Test sentence", "BERT is great", "Fine-tuning rocks"]
    labels = [0, 1, 0, 1]
    return texts, labels


def test_classification_dataset_length(tokenizer, sample_data):
    texts, labels = sample_data
    ds = TextClassificationDataset(texts, labels, tokenizer, max_length=32)
    assert len(ds) == 4


def test_classification_dataset_item_keys(tokenizer, sample_data):
    texts, labels = sample_data
    ds = TextClassificationDataset(texts, labels, tokenizer, max_length=32)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


def test_dataloader_build_returns_loaders(tokenizer, sample_data):
    texts, labels = sample_data
    cfg = EngineConfig(batch_size=2, eval_batch_size=2)
    loader = ContextualDataLoader(cfg, tokenizer)
    train_dl, eval_dl = loader.build(texts, labels, texts, labels)
    assert train_dl is not None
    assert eval_dl is not None


def test_dataloader_build_no_eval(tokenizer, sample_data):
    texts, labels = sample_data
    cfg = EngineConfig(batch_size=2)
    loader = ContextualDataLoader(cfg, tokenizer)
    train_dl, eval_dl = loader.build(texts, labels)
    assert train_dl is not None
    assert eval_dl is None
