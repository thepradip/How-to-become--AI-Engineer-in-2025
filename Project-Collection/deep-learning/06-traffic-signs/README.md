# DL 06 · Traffic Sign Recognition (GTSRB) 🟡

**Problem.** Classify road signs from images — a building block of driver-assistance and autonomous
driving. 43 sign types, varying lighting and angles.

**What you build.** A color-image CNN over 32×32 crops, trained and evaluated in the shared chat UI.

## Dataset (real)
**GTSRB** (German Traffic Sign Recognition Benchmark, ~50k images, 43 classes) via
`torchvision.datasets.GTSRB`. Smoke test uses synthetic tensors (offline).

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # CPU smoke test
streamlit run app.py      # "train"
```
Real training: `from src.model import real_loaders, TrafficCNN` then `train_classifier(...)`.

## What you learned
Multi-class image classification · handling many classes · data augmentation opportunities (rotation,
brightness) for robustness.

## Tested on
CPU (Python 3.11) — smoke test (synthetic, offline). Real GTSRB training is light but faster on GPU.

> **Why it matters.** Sign/symbol/logo classification appears across automotive, mapping and
> brand-monitoring projects.
