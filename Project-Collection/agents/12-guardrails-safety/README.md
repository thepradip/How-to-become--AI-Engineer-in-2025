# Agents 12 · Guardrails / Safety Rails 🟡

**Problem.** Agents need protection on both sides: block jailbreaks/abuse on the way **in**, and scrub
PII/unsafe content on the way **out**. Rails are how you make an agent safe to ship.

**What you build.** An **input rail** (jailbreak + disallowed-request blocking) and an **output rail**
(PII masking) wrapping a model, in the shared chat UI. Production uses **NeMo Guardrails** /
**Guardrails AI** to declare these rails as config.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # test a jailbreak input, or `out: <text with PII>`
```

## What you learned
Input vs. output rails · jailbreak detection · PII redaction · why rails belong *outside* the model.
**Production:** NeMo Guardrails Colang flows, Guardrails AI validators, topical rails, fact-checking rails.

## Tested on
CPU (Python 3.11), fully offline (rule-based rails). No GPU.

> **Why it matters.** "Make our chatbot safe / compliant" is a frequent, standalone hardening project.
