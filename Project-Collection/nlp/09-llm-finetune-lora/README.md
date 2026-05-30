# NLP 09 · LLM Fine-Tune with LoRA/QLoRA (Unsloth) 🔴 · *16 GB GPU*

**Problem.** Adapt an open LLM to your domain/voice without renting a cluster. **LoRA** trains tiny
adapters; **QLoRA** adds 4-bit quantization so a 7–8B model fits a single **T4 / Apple M-series (16 GB)**.
**Unsloth** makes it ~2× faster with lower memory.

**What you build.** Instruction-data prep (Alpaca format) + a complete Unsloth QLoRA training script.
The chat UI previews the formatted data; training runs on a GPU.

## Run it
```bash
pip install -r requirements.txt        # app/tests
pytest -q                              # data-prep utilities (offline)
streamlit run app.py                   # "data" → preview a training example
```
Fine-tune (GPU):
```bash
pip install unsloth trl peft transformers datasets bitsandbytes
python -c "from src.finetune import train_lora; train_lora()"   # QLoRA on Qwen3-4B (4-bit)
```

## What you learned
LoRA vs. full fine-tuning · QLoRA 4-bit memory savings · the SFT instruction format · using Unsloth/TRL.

## Tested on
CPU (Python 3.11) — `pytest` covers the **data-prep** utilities offline (no model). **Actual training
requires a CUDA GPU (16 GB)**; `train_lora` is marked `# pragma: no cover` and documented, not run here.
We do not claim a training run that didn't happen.

> **Freelance relevance.** "Fine-tune an open model on our data" is one of the highest-paid LLM gigs
> ($150–250/hr per Upwork's 2026 data).
