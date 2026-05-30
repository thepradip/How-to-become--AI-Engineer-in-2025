# DL 03 · Pneumonia Detection from Chest X-rays (+ Grad-CAM) 🔴

**Problem.** Flag likely pneumonia in chest X-rays to help triage. In medical AI, a prediction alone
isn't enough — clinicians need to see **why**. So we add **Grad-CAM** heatmaps showing where the model
focused.

**What you build.** A fine-tuned ResNet-18 (NORMAL vs PNEUMONIA) plus a from-scratch **Grad-CAM**
implementation and an overlay visualisation — in the shared chat UI.

## Dataset (real)
**Chest X-Ray Pneumonia** (Kaggle `paultimothymooney/chest-xray-pneumonia`, ~5,800 images in
NORMAL/PNEUMONIA folders). Download with the Kaggle API into `./data`. Smoke tests use synthetic
tensors so they run offline.

## Approach
ImageNet-pretrained ResNet-18 → 2-class head → fine-tune. Grad-CAM hooks the last conv block, weights
its activations by the gradient of the predicted class, and renders a heatmap over the X-ray.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # CPU smoke test (offline)
streamlit run app.py      # "train" → "explain" (Grad-CAM)
```

## What you learned
Medical-image fine-tuning · **explainability with Grad-CAM** (forward/backward hooks) · why
explanations are often the real deliverable in healthcare AI.

## Tested on
CPU (Python 3.11) — smoke test covers model output shape, Grad-CAM heatmap shape/range, the overlay,
and the handler, all on synthetic data (`weights=None`, offline). **Real fine-tuning wants a T4/M2 GPU.**

> **Freelance relevance.** Medical/industrial image triage **with explainability** is a high-value,
> trust-sensitive engagement.
