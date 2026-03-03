"""
Contextual Intelligence Engine
A domain-adaptive BERT fine-tuning framework for contextual understanding.
"""

from .config import EngineConfig
from .model import ContextualModel
from .trainer import Trainer
from .inference import InferenceEngine

__all__ = ["EngineConfig", "ContextualModel", "Trainer", "InferenceEngine"]
__version__ = "0.1.0"
