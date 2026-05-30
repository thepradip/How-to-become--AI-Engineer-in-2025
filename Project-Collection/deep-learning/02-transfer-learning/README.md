# DL 02 · Transfer Learning (ResNet-18) 🟡

**Problem.** You rarely have millions of labelled images. Transfer learning lets you train a strong
classifier on **hundreds** of images by reusing an ImageNet-pretrained backbone and training a new head.

**What you build.** A `build_model` that swaps ResNet-18's classifier head, freezes the backbone, and
fine-tunes on your classes — in the shared chat UI. Works with any `ImageFolder` directory.

## Dataset (real)
Bring your own labelled images as `class_name/*.jpg` folders (e.g., Oxford Flowers, Food-101 subset).
The smoke demo uses synthetic tensors so it runs offline.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # CPU smoke test (offline, weights=None)
streamlit run app.py      # "train" (synthetic demo)
```
Real training:
```python
from src.transfer import build_model, real_loader
from _shared.torch_utils import train_classifier, get_device
loader, classes = real_loader("my_images/")        # folders = classes
m = build_model(len(classes), pretrained=True)      # downloads ImageNet weights
train_classifier(m, loader, epochs=5, device=get_device())
```

## What you learned
Freezing/unfreezing layers · replacing a classifier head · why pretrained features transfer · the
data-efficiency of fine-tuning.

## Tested on
CPU (Python 3.11) — smoke test builds the model (`weights=None`, offline) and trains one step on
synthetic data. **Real fine-tuning with pretrained weights is much faster on a T4/M2 GPU.**

> **Freelance relevance.** "Classify our product images / detect defects" is a staple CV contract;
> transfer learning is how you deliver it on a small dataset.
