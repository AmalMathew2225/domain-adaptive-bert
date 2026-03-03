"""
Configuration module for the Contextual Intelligence Engine.
Centralizes all hyperparameters and training settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EngineConfig:
    """
    Central configuration class for training and inference.

    Attributes:
        model_name: Pretrained HuggingFace model identifier.
        num_labels: Number of output classes (for classification tasks).
        max_length: Maximum token sequence length.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        learning_rate: AdamW optimizer learning rate.
        num_epochs: Number of training epochs.
        warmup_steps: Scheduler warmup steps.
        weight_decay: L2 regularization coefficient.
        output_dir: Directory to save checkpoints and results.
        seed: Random seed for reproducibility.
        device: Compute device ('cpu', 'cuda', or 'mps').
        fp16: Enable mixed-precision training (requires CUDA).
        task_type: Task type — 'classification', 'ner', or 'mlm'.
        label_names: Optional list of class label names.
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        gradient_accumulation_steps: Accumulate gradients before updating.
    """

    # Model
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_length: int = 128

    # Training
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    fp16: bool = False

    # Task
    task_type: str = "classification"  # "classification" | "ner" | "mlm"
    label_names: Optional[List[str]] = None

    # I/O
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "cpu"  # auto-detected in Trainer if not set

    # Logging & checkpointing
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500

    def __post_init__(self):
        valid_tasks = {"classification", "ner", "mlm"}
        if self.task_type not in valid_tasks:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. Choose from {valid_tasks}."
            )

    def to_dict(self) -> dict:
        """Serialize config to a plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        """Instantiate config from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
