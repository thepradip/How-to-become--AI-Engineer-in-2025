# Agents 09 · Healthcare Agent (guardrailed) 🔴

**Problem.** Healthcare AI is high-stakes. A safe assistant shares **general, sourced** information but
**refuses to diagnose, prescribe, or dose**, and **escalates emergencies** — guardrails are the whole
point.

**What you build.** A guideline-RAG agent wrapped in safety guardrails: emergency escalation, refusal
of diagnosis/prescription, and a disclaimer on every answer — in the shared chat UI.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # checks escalation, refusal, and disclaimers
streamlit run app.py      # ask general wellness questions
```

## What you learned
Domain guardrails (refuse/escalate) · sourced answers + disclaimers · why safety policy matters more
than model quality in healthcare. **Production:** a clinically-validated KB, HIPAA-compliant hosting,
audit logging, and human oversight.

## Tested on
CPU (Python 3.11), fully offline (TF-IDF retrieval + rule guardrails). No GPU.

> ⚠️ Educational only — **not a medical device**. > **Freelance relevance:** healthcare GenAI
> (with compliance/guardrails) is a high-value, regulated niche.
