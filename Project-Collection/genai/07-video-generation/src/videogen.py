"""Video generation.

Real text-to-video uses open models (**Wan 2.2 / LTX-Video**, 16 GB GPU) or hosted APIs
(Veo, Kling, Runway, Seedance). To run offline anywhere, we generate a short animated
clip procedurally (a moving pattern seeded by the prompt) and assemble it into a GIF.
It's a placeholder for the real diffusion-video model.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np


def _seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % (2**32)


def generate_frames(prompt: str, n_frames: int = 16, size: int = 128) -> np.ndarray:
    """Return (n_frames, H, W, 3) uint8 — an animated abstract clip (offline demo)."""
    rng = np.random.default_rng(_seed(prompt))
    yy, xx = np.mgrid[0:size, 0:size] / size
    fx, fy = rng.uniform(2, 6, 2)
    frames = []
    for t in range(n_frames):
        phase = 2 * np.pi * t / n_frames
        r = np.sin(2 * np.pi * (fx * xx) + phase) * 0.5 + 0.5
        g = np.sin(2 * np.pi * (fy * yy) + phase * 1.3) * 0.5 + 0.5
        b = np.sin(2 * np.pi * (fx * xx + fy * yy) + phase * 0.7) * 0.5 + 0.5
        frames.append((np.stack([r, g, b], -1) * 255).astype("uint8"))
    return np.array(frames)


def to_gif(frames: np.ndarray, duration: int = 80) -> bytes:
    """Encode frames into an animated GIF (bytes)."""
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0)
    return buf.getvalue()


def generate_ltx(prompt: str):
    """Real local text-to-video with LTX-Video via diffusers (GPU). Documented path."""
    import torch
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")
    return pipe(prompt=prompt, num_frames=49).frames[0]
