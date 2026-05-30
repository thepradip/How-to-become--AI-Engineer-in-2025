# DL 01 · CNN from Scratch (Fashion-MNIST) 🟢

**Problem.** Image classification is the "hello world" of deep learning. Here you build a
convolutional neural network **from scratch** in PyTorch — conv/ReLU/pool blocks + a classifier head
— and train it to recognise clothing categories.

**What you build.** A `SimpleCNN`, a training loop (shared `_shared/torch_utils.py`), evaluation, and a
chat UI that trains briefly and classifies example images.

## Dataset (real)
**Fashion-MNIST** (60k train / 10k test, 28×28 grayscale, 10 classes) via torchvision — downloads to
`./data` on first run.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # CPU smoke test (synthetic tensors)
streamlit run app.py      # "train" (set epochs/batches in sidebar) → "demo"
```
Full training to ~90%+ accuracy:
```python
from src.cnn import SimpleCNN, real_loaders
from _shared.torch_utils import train_classifier, accuracy, get_device
tr, te = real_loaders(); m = SimpleCNN()
train_classifier(m, tr, epochs=5, device=get_device()); print(accuracy(m, te))
```

## What you learned
Conv/pool architecture · a PyTorch training loop · CPU vs. CUDA/MPS device handling · why accuracy on
a held-out set matters.

## Tested on
CPU (Python 3.11) — `pytest` runs a synthetic-tensor smoke test (model forward, 1 training step,
handler). **Full Fashion-MNIST training is light enough for CPU but faster on a T4/M2 GPU.** Tests
`importorskip('torch')`, so they skip cleanly if torch isn't installed.

> **Why it matters.** Custom image classifiers (defect detection, content moderation, sorting)
> are common CV projects; this is the foundation.
