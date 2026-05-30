# GenAI 06 · Image Generation 🟡

**Problem.** Generate images from text — for product, marketing and creative tooling. Production uses
diffusion models (**Stable Diffusion 3.5 / FLUX.1**) locally on a 16 GB GPU or via hosted APIs.

**What you build.** A text-to-image chat app. The **offline demo** is a deterministic procedural
generator (so it runs anywhere); the **real path** uses `diffusers`.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # type a prompt
```
Real diffusion (GPU):
```python
# pip install diffusers transformers accelerate torch
from src.imagegen import generate_diffusers
generate_diffusers("a photorealistic red fox in snow")   # SD 3.5 / FLUX
```

## What you learned
The text-to-image workflow · prompt → image · local diffusion vs. hosted APIs · seeds & determinism.
**Extensions:** ComfyUI workflows, LoRA style fine-tuning, img2img/inpainting.

## Tested on
CPU (Python 3.11), offline (procedural demo). Real diffusion needs `diffusers` + a 16 GB GPU.

> **Freelance relevance.** Image-gen pipelines (and editing/inpainting) are booming — AI image work is
> among the fastest-growing freelance categories in 2026.
