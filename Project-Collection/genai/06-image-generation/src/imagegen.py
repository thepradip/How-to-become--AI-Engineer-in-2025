"""Image generation.

Real text-to-image uses diffusion models (**Stable Diffusion 3.5 / FLUX.1**) via the
``diffusers`` library on a 16 GB GPU — the documented production path. So the app runs
*offline anywhere*, we ship a tiny deterministic procedural generator that turns a text
prompt into a colourful abstract image (seeded by the prompt). It's a placeholder for
the real model, not a diffusion model.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**32)


def generate(prompt: str, size: int = 256) -> np.ndarray:
    """Deterministic abstract RGB image (H,W,3 uint8) seeded by the prompt (offline demo)."""
    rng = np.random.default_rng(_seed(prompt))
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.zeros((size, size, 3), dtype="float32")
    for _ in range(3 + rng.integers(0, 4)):  # blend a few sine gradients & blobs
        fx, fy = rng.uniform(1, 6, 2)
        phase = rng.uniform(0, 6.28)
        ch = rng.integers(0, 3)
        img[..., ch] += 0.5 + 0.5 * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
    cx, cy, r = rng.uniform(0.2, 0.8, 2).tolist() + [rng.uniform(0.1, 0.3)]
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    img += blob[..., None] * rng.uniform(0, 1, 3)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    return (img * 255).astype("uint8")


def generate_diffusers(prompt: str, model: str = "stabilityai/stable-diffusion-3.5-medium"):
    """Real text-to-image with diffusers (GPU). Documented path."""
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
    return pipe(prompt).images[0]
