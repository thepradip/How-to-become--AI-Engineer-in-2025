# GenAI 05 · 1-bit LLMs — BitNet / Bonsai 🔴

**Problem.** Frontier models are huge. **1-bit LLMs** store weights as {-1, 0, 1} (1.58-bit) — or true
1-bit (Bonsai) — shrinking models 10–16× so they run on a CPU/phone with far less energy.

**What you build.** A from-scratch **ternary (absmean) quantization** demo that converts a weight
matrix to {-1,0,1}, reports the compression ratio and reconstruction error — the exact operation at
BitNet's core — in the shared chat UI.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "demo"
```
Run a real 1-bit model:
```python
# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
# or bitnet.cpp for fast CPU inference; PrismML Bonsai for true 1-bit
```

## What you learned
Quantization math (absmean ternary) · the size/quality trade-off · why 1-bit enables on-device AI.

## Tested on
CPU (Python 3.11), fully offline (numpy quantization demo). Running a real BitNet/Bonsai model needs
`transformers`/`bitnet.cpp` + a weight download.

> **Freelance relevance.** On-device / edge LLMs are an emerging niche; understanding quantization is
> key to cheap deployment.
