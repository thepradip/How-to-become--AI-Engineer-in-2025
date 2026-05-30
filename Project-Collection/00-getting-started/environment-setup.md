# Environment Setup

## 1. Python & a virtual environment
Use Python **3.11+**. One venv per track is plenty (projects in a track share most deps).

```bash
# with the standard library
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# …or with uv (much faster — recommended)
uv venv && source .venv/bin/activate
```
Then, inside any project: `pip install -r requirements.txt` (or `uv pip install -r requirements.txt`).

## 2. Hardware tiers
| Tier | Examples | Good for |
|------|----------|----------|
| **CPU only** | any laptop | All `ml/`, classic `nlp/`, RAG, agents (API-backed), 1-bit models |
| **16 GB GPU** | Colab/Kaggle **T4**, Apple **M2/M3** | `deep-learning/`, LoRA/QLoRA fine-tune, local 4-bit LLMs, image gen |
| **Hosted API** | OpenAI/Anthropic/Google, Replicate | GenAI/agent showcases, video gen |

> No GPU? Free **Google Colab** and **Kaggle Notebooks** both give a T4 (16 GB). Every GPU project's
> README says exactly what needs a GPU and what runs on CPU.

## 3. Local model serving
- **[Ollama](https://ollama.com)** — one-command local models. `ollama run qwen3:4b`. Best for laptops & quick chat.
- **[vLLM](https://docs.vllm.ai)** — high-throughput serving for a 16 GB GPU; OpenAI-compatible API
  (`vllm serve Qwen/Qwen3-4B`). Used in the local-inference and agent projects.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp) / GGUF** — run quantized models anywhere, incl. CPU.

## 4. Recommended 2026 local models (fit 16 GB / 4-bit)
| Use | Model | Notes |
|-----|-------|-------|
| Text (all-round) | **Qwen3-4B**, **Gemma 3 4B** | ~2.5 GB at 4-bit; great default |
| Reasoning | **DeepSeek-R1-Distill-Qwen-7B** | thinking traces |
| Tiny / on-device | **SmolLM2-1.7B** | ~1 GB |
| **1-bit / ternary** | **BitNet b1.58 2B4T**, **Bonsai** | runs on CPU; covered in `genai/05` |
| Image gen | **SD 3.5 Medium**, **FLUX.1** | via `diffusers` / ComfyUI |
| Video gen | **LTX-Video** (12 GB), **Wan 2.2** | `genai/07`; hosted fallback (Veo/Kling) |
| OCR / docs | **olmOCR**, **Docling**, **Qwen2.5-VL** | `genai/08` |
| Speech | **faster-whisper** (STT), **Piper/Coqui** (TTS) | voice agent in `agents/11` |

## 5. API keys (only for hosted-model projects)
Copy `.env.example` → `.env` inside a project and fill what it needs. Common ones:
```
OPENAI_API_KEY=…
ANTHROPIC_API_KEY=…
GOOGLE_API_KEY=…
REPLICATE_API_TOKEN=…        # hosted image/video
HF_TOKEN=…                   # gated Hugging Face models
```
`.env` is git-ignored. **Never commit keys.** Projects that can run fully local say so and need none.

## 6. Datasets
Real data downloads on first run via each project's `src/download_data.py` (OpenML, UCI, Hugging
Face, yfinance, or the Kaggle API). For Kaggle-hosted sets: install `kaggle`, drop your
`kaggle.json` token in `~/.kaggle/`, and the script does the rest. Data is never committed.
