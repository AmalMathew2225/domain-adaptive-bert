# Contextual Intelligence Engine 🧠

> A modular, hackathon-ready framework for domain-adaptive BERT fine-tuning — covering classification, NER, and masked language modelling.

---

## ✨ Features

- **Plug-and-play config** — single `EngineConfig` dataclass controls everything
- **Multi-task support** — text classification, token classification (NER), and domain-adaptive MLM pretraining
- **Clean data pipeline** — `TextClassificationDataset`, `MLMDataset`, and HuggingFace `datasets` integration
- **Production-grade training loop** — gradient accumulation, mixed-precision (fp16), LR warmup/decay, auto-checkpointing
- **Lazy inference pipeline** — load any saved checkpoint and call `predict()` with one line
- **pip-installable** — standard `setup.py` for packaging

---

## 📁 Project Structure

```
contextual-intelligence-engine/
│
├── engine/
│   ├── __init__.py       ← Public API exports
│   ├── config.py         ← EngineConfig dataclass
│   ├── data_loader.py    ← Dataset classes + DataLoader factory
│   ├── model.py          ← ContextualModel (HF wrapper)
│   ├── trainer.py        ← Fine-tuning loop
│   └── inference.py      ← Inference pipeline
│
├── examples/
│   └── sample_finetune.ipynb  ← End-to-end walkthrough notebook
│
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   └── test_model.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone the repo
git clone https://github.com/your-org/contextual-intelligence-engine.git
cd contextual-intelligence-engine

# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e .
```

### 2. Fine-tune on your own data

```python
from engine import EngineConfig, ContextualModel, Trainer
from engine.data_loader import ContextualDataLoader

# 1 — Configure
config = EngineConfig(
    model_name="bert-base-uncased",
    num_labels=2,
    task_type="classification",
    num_epochs=3,
    learning_rate=2e-5,
    output_dir="./outputs",
)

# 2 — Load model
ctx_model = ContextualModel(config)

# 3 — Build DataLoaders
loader = ContextualDataLoader(config, ctx_model.tokenizer)
train_dl, eval_dl = loader.build(
    train_texts=["I love this!", "This is terrible."],
    train_labels=[1, 0],
    eval_texts=["Pretty good.", "Not great."],
    eval_labels=[1, 0],
)

# 4 — Train
trainer = Trainer(ctx_model, config)
history = trainer.train(train_dl, eval_loader=eval_dl)
print(history)
```

### 3. Run inference

```python
from engine import EngineConfig
from engine.inference import InferenceEngine

config = EngineConfig(task_type="classification", device="cpu")
engine = InferenceEngine.from_pretrained(
    "./outputs/best_checkpoint",
    config,
    label_names=["NEGATIVE", "POSITIVE"],
)

results = engine.predict(["This framework is amazing!", "I can't get it to work."])
# [{'label': 'POSITIVE', 'score': 0.97}, {'label': 'NEGATIVE', 'score': 0.91}]
```

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `bert-base-uncased` | HuggingFace model ID or local path |
| `task_type` | `classification` | `classification` \| `ner` \| `mlm` |
| `num_labels` | `2` | Number of output classes |
| `max_length` | `128` | Max tokenisation length |
| `batch_size` | `16` | Training batch size |
| `learning_rate` | `2e-5` | AdamW learning rate |
| `num_epochs` | `3` | Training epochs |
| `fp16` | `False` | Mixed-precision (CUDA only) |
| `output_dir` | `./outputs` | Checkpoint / model save directory |
| `device` | `cpu` | `cpu` \| `cuda` \| `mps` \| `auto` |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=engine
```

---

## 📓 Example Notebook

Open `examples/sample_finetune.ipynb` for an end-to-end walkthrough using the IMDb dataset:

```bash
jupyter notebook examples/sample_finetune.ipynb
```

---

## 🗺️ Roadmap

- [ ] WandB / MLflow experiment tracking integration
- [ ] LoRA / QLoRA parameter-efficient fine-tuning
- [ ] Multi-GPU training via HuggingFace Accelerate
- [ ] ONNX export for production serving
- [ ] Gradio demo interface

---

## 📜 License

MIT © 2026
