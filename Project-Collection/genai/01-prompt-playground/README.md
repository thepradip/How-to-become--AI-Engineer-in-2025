# GenAI 01 · Prompt Engineering Playground 🟢

**Problem.** The same model gives very different results depending on the prompt. This playground lets
you experiment with prompting techniques against whichever LLM you have available.

**What you build.** A chat app on top of the **shared LLM client** (`_shared/llm.py`) that routes to
OpenAI / Anthropic / Ollama / an offline mock, shows which provider answered, and demonstrates a
chain-of-thought transformation.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (mock provider)
streamlit run app.py      # type a task; `cot:` for step-by-step; `providers` to list backends
```
Use a real model:
```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY=...
# or run a local model:  ollama run qwen3:4b
```

## What you learned
Zero-shot vs. few-shot vs. chain-of-thought · system prompts · provider abstraction & graceful
fallback (a pattern you'll reuse in every LLM app).

## Tested on
CPU (Python 3.11), fully offline via the mock provider. Real providers need a key or a local Ollama.

> **Freelance relevance.** Prompt design + a provider-agnostic client is the foundation of every LLM
> feature you'll ship.
