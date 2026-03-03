"""
Tests for ContextualModel.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from engine.config import EngineConfig


def test_resolve_device_cpu():
    from engine.model import ContextualModel
    device = ContextualModel._resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto():
    from engine.model import ContextualModel
    device = ContextualModel._resolve_device("auto")
    assert isinstance(device, torch.device)


@patch("engine.model.AutoModelForSequenceClassification.from_pretrained")
@patch("engine.model.AutoTokenizer.from_pretrained")
@patch("engine.model.AutoConfig.from_pretrained")
def test_model_init_classification(mock_cfg, mock_tok, mock_model):
    mock_model.return_value = MagicMock()
    mock_model.return_value.to = MagicMock(return_value=mock_model.return_value)
    mock_tok.return_value = MagicMock()
    mock_cfg.return_value = MagicMock()

    from engine.model import ContextualModel
    config = EngineConfig(model_name="bert-base-uncased", task_type="classification", device="cpu")
    ctx = ContextualModel(config)
    assert ctx.device == torch.device("cpu")
    mock_model.assert_called_once()


@patch("engine.model.AutoModelForSequenceClassification.from_pretrained")
@patch("engine.model.AutoTokenizer.from_pretrained")
@patch("engine.model.AutoConfig.from_pretrained")
def test_num_parameters(mock_cfg, mock_tok, mock_model):
    mock_inner = MagicMock()
    mock_inner.parameters.return_value = [torch.randn(10, 10), torch.randn(5)]
    mock_inner.to = MagicMock(return_value=mock_inner)
    mock_model.return_value = mock_inner
    mock_tok.return_value = MagicMock()
    mock_cfg.return_value = MagicMock()

    from engine.model import ContextualModel
    config = EngineConfig(device="cpu")
    ctx = ContextualModel(config)
    n = ctx.num_parameters()
    assert n == 105  # 10*10 + 5
