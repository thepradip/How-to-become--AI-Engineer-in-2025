# Agents 07 · Type-safe Tool Agent (PydanticAI) 🟡

**Problem.** Agents that call tools with wrong/garbage arguments fail in production. **PydanticAI**
gives every tool a **Pydantic-typed project**, so bad calls are rejected before they run.

**What you build.** Tools (`get_weather`, `convert_currency`) with Pydantic arg models + validators,
and a router that validates before executing — in the shared chat UI. Production uses **PydanticAI**'s
`Agent` where an LLM fills the same typed arguments.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "weather in Paris", "convert 100 USD to EUR"
```
Production:
```python
# pip install pydantic-ai
# Agent("openai:gpt-4o-mini") with @agent.tool functions whose args are Pydantic models.
```

## What you learned
Typed tool projects · validation as a safety boundary · structured tool I/O. **Extensions:** real
PydanticAI agent, dependency injection, retries on validation failure.

## Tested on
CPU (Python 3.11), fully offline (regex routing + Pydantic validation). LLM version needs a key.

> **Why it matters.** Type-safety is what makes tool-using agents reliable enough to ship.
