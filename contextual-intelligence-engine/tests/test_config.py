"""
Tests for EngineConfig.
"""

import pytest
from engine.config import EngineConfig


def test_default_config():
    cfg = EngineConfig()
    assert cfg.model_name == "bert-base-uncased"
    assert cfg.task_type == "classification"
    assert cfg.num_labels == 2


def test_custom_config():
    cfg = EngineConfig(model_name="roberta-base", num_labels=5, task_type="ner")
    assert cfg.model_name == "roberta-base"
    assert cfg.num_labels == 5
    assert cfg.task_type == "ner"


def test_invalid_task_type():
    with pytest.raises(ValueError, match="Invalid task_type"):
        EngineConfig(task_type="invalid_task")


def test_round_trip_serialization():
    cfg = EngineConfig(num_labels=3, learning_rate=3e-5)
    restored = EngineConfig.from_dict(cfg.to_dict())
    assert restored.num_labels == cfg.num_labels
    assert restored.learning_rate == cfg.learning_rate
