# GenAI 10 · Evals + Guardrails Harness 🔴 · *LLMOps capstone*

**Problem.** Shipping LLM apps without measurement is flying blind, and without guardrails it's unsafe.
This harness scores output quality and enforces safety — the LLMOps layer clients increasingly require.

**What you build.** Quality metrics (**exact match**, **keyword/faithfulness recall**) + **guardrails**
(PII detection, banned-content checks), aggregated into a report — in the shared chat UI.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "demo" runs the harness; paste text to guardrail-check it
```

## What you learned
Evaluating generations (not just classifiers) · faithfulness vs. correctness · PII/safety guardrails ·
turning evals into a regression gate. **Production tools:** **Ragas**/**DeepEval** for RAG/LLM evals,
**Guardrails AI**/**NeMo Guardrails** for input/output safety, wired into CI.

## Tested on
CPU (Python 3.11), fully offline (regex/string metrics). No GPU. Production evaluators add LLM-as-judge
metrics (need a model).

> **Why it matters.** "Make sure it's accurate and safe" — evals + guardrails are how you prove it,
> and a frequent standalone engagement.
