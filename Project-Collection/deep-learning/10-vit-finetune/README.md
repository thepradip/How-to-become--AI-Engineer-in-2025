# DL 10 · Vision Transformer Fine-Tune + W&B Sweeps 🔴 · *capstone*

**Problem.** Transformers now lead image tasks. This capstone fine-tunes a **ViT-B/16** on custom
classes and runs **Weights & Biases** experiment tracking + hyperparameter **sweeps** — the modern,
reproducible training workflow.

**What you build.** A ViT with a swapped classification head, a fine-tuning loop, and optional W&B
logging — in the shared chat UI.

## Model & data (real)
torchvision **ViT-B/16** (ImageNet-pretrained for real runs). Bring images as an `ImageFolder`
(reuse DL 02's `real_loader` pattern at 224×224). Smoke tests use synthetic 224×224 tensors offline.

## Run it
```bash
pip install -r requirements.txt        # add: pip install wandb
pytest -q                              # CPU smoke test (offline, weights=None)
streamlit run app.py                   # "train"
```
W&B hyperparameter sweep (real):
```bash
wandb login
wandb sweep sweep.yaml      # define lr/batch/epochs ranges
wandb agent <entity/project/sweep_id>
```

## What you learned
Vision Transformers vs. CNNs · fine-tuning a large pretrained model · **experiment tracking &
sweeps** with W&B (the reproducibility skill teams expect). 

## Tested on
CPU (Python 3.11) — smoke test builds ViT (`weights=None`, offline), checks output shape and one
training step. **Real ViT fine-tuning needs a T4/M2 (16 GB) GPU.** W&B is optional (offline mode used
if installed).

> **Freelance relevance.** "Fine-tune a SOTA model on our images and track experiments" is exactly the
> deliverable here.
