# GenAI 07 · Video Generation 🔴

**Problem.** Generate short video from text — the fastest-growing AI media category (Upwork: AI video
+329% YoY in 2026). Run open models locally (**Wan 2.2 / LTX-Video**, 16 GB GPU) or call hosted APIs
(**Veo 3.1, Kling 3.0, Runway, Seedance**).

**What you build.** A text-to-video chat app. The **offline demo** synthesises a short animated GIF
procedurally; the **real path** uses `diffusers` (LTX-Video) or a hosted API.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # type a prompt → animated clip
```
Real text-to-video (GPU):
```python
# pip install diffusers torch
from src.videogen import generate_ltx
generate_ltx("a drone shot over a forest at sunrise")     # LTX-Video
```

## What you learned
The text-to-video landscape (open vs. hosted) · frames → clip assembly · native-audio trend in 2026.
**Extensions:** image-to-video, ComfyUI video workflows, Replicate hosted calls.

## Tested on
CPU (Python 3.11), offline (procedural GIF). Real video gen needs a GPU (LTX ~12 GB) or a hosted API key.

> **Freelance relevance.** AI video creation/editing is the single fastest-growing freelance AI skill.
