# Agents 02 · SQL Agent (LangGraph) 🟡

**Problem.** Let non-technical users query a database in plain English. A SQL agent turns the question
into SQL, runs it, and returns results — a hugely popular "chat with your data" build.

**What you build.** NL → SQL over a real **SQLite** DB with results in the shared chat UI. The
translation is rule-based here (offline); production uses **LangGraph** + an LLM that sees the schema.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # ask about the employees table
```
Production (LLM-driven, with guardrails):
```python
# pip install langgraph langchain-community
# Build a graph: question -> (LLM writes SQL) -> validate -> execute -> (LLM summarises)
```

## What you learned
NL→SQL · executing generated SQL safely · why you validate/limit queries (read-only, row caps) ·
schema-in-prompt. **Extensions:** LangGraph state machine, query validation, multi-table joins.

## Tested on
CPU (Python 3.11), fully offline (SQLite + rule-based NL→SQL). LLM-driven version needs a key.

> **Freelance relevance.** "Chat with our database" dashboards are a frequent, high-value request.
