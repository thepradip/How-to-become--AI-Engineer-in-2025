# GenAI 04 · Local LLM Chat (Ollama / vLLM) 🟡

**Problem.** Run a capable LLM **locally** — for privacy, zero per-token cost, and offline use. On a
16 GB GPU (or Apple M-series) a 4-bit Qwen3-4B / Gemma 3 runs comfortably.

**What you build.** A chat app that talks to a local model via **Ollama** or a **vLLM** OpenAI-compatible
server (shared LLM client), keeping short conversation history. Falls back to an offline mock.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (mock)
# real local model:
ollama run qwen3:4b       # or: vllm serve Qwen/Qwen3-4B
streamlit run app.py      # chat; `reset` clears; `providers` lists backends
```

## What you learned
Local inference with Ollama/vLLM · OpenAI-compatible endpoints · conversation history · cost/privacy
trade-offs of local vs. hosted.

## Tested on
CPU (Python 3.11), offline via mock. A real local model needs Ollama or vLLM (16 GB GPU recommended).

> **Why it matters.** "Run it on-prem / no data leaves our network" is a common enterprise
> requirement local models satisfy.
