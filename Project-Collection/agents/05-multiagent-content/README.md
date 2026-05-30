# Agents 05 · Multi-Agent Content Team (CrewAI) 🟡

**Problem.** Some work decomposes into roles — research, writing, editing. Multi-agent "crews" assign
each role to a specialised agent that hands off to the next, often producing better results than one
agent doing everything.

**What you build.** A researcher → writer → editor pipeline that turns a topic into an article, showing
each role's output, in the shared chat UI. Offline it's deterministic; production uses **CrewAI**.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # give a topic (e.g. "RAG")
```
Production (CrewAI sequential process):
```python
# pip install crewai
# researcher/writer/editor Agents + Tasks, Process.sequential — each is an LLM with a goal.
```

## What you learned
Role decomposition · sequential hand-off (pipeline) · when multi-agent beats single-agent. **Extensions:**
CrewAI with real LLMs/tools, hierarchical process, parallel roles.

## Tested on
CPU (Python 3.11), fully offline (deterministic roles). Real crews need CrewAI + an LLM key.

> **Freelance relevance.** Content/marketing automation crews are a popular productised-agent offering.
