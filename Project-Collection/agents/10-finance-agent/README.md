# Agents 10 · Finance Agent 🟡

**Problem.** Analyse company financials and answer questions (growth, profit, margin) while **refusing
investment advice** — finance is regulated, so guardrails are mandatory.

**What you build.** A finance analyst over a financials table (SQLite) with a no-advice guardrail, in
the shared chat UI. Production adds **RAG over filings/10-Ks** + an LLM + compliance controls.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # ask about revenue growth / profit / margin
```

## What you learned
Combining structured analysis (SQL) with guardrails · refusing out-of-scope (advice) requests ·
finance-domain caution. **Extensions:** RAG over earnings reports, anomaly detection, document automation.

## Tested on
CPU (Python 3.11), fully offline (SQLite). No GPU.

> ⚠️ Educational — **not investment advice.** > **Freelance relevance:** finance document automation &
> analysis are high-value, compliance-sensitive engagements.
