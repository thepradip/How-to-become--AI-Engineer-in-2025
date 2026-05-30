# GenAI 09 · Structured Extraction + Guardrails 🟡

**Problem.** LLMs emit free text, but apps need **typed, validated** data. Schema validation is a
guardrail: malformed output (bad email, negative budget) is rejected before it corrupts your system.

**What you build.** A **Pydantic** schema (`Lead`) + extraction that validates on the way in. The
offline path uses regex; the production path uses **PydanticAI**, which makes an LLM fill the same
schema with type guarantees.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (regex extraction + validation)
streamlit run app.py      # paste a message → validated JSON (or a guardrail error)
```
LLM-backed (robust) extraction:
```python
# pip install pydantic-ai
from src.extract import extract_with_pydantic_ai
extract_with_pydantic_ai("Reach me at x@y.com, I'm Sam from Hooli, budget 9000")
```

## What you learned
Schema-first extraction · Pydantic validators as **guardrails** · type-safe LLM output with PydanticAI.
**Extensions:** retries on validation failure, nested schemas, JSON-mode / function-calling.

## Tested on
CPU (Python 3.11), fully offline (regex + Pydantic validation). LLM extraction needs `pydantic-ai` + a key.

> **Freelance relevance.** Reliable structured output is what separates a demo from a shippable LLM
> feature — clients pay for the guardrails.
