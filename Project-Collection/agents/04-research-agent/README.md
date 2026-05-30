# Agents 04 · Research Agent (CrewAI) 🟡

**Problem.** Answer a question by gathering and synthesizing information from multiple sources — what a
human researcher does, automated.

**What you build.** An agent that searches a corpus, ranks sources, and writes a **cited** synthesis,
in the shared chat UI. Offline it searches a local corpus; production uses **CrewAI** with a web-search
tool + an LLM.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # ask a research question
```
Production (CrewAI):
```python
# pip install crewai crewai-tools
# A "researcher" agent with a SerperDevTool searches the web; a "writer" agent synthesizes.
```

## What you learned
Search + synthesize + cite · grounding answers in retrieved sources · the single-agent research
pattern (precursor to multi-agent crews — Agents 05).

## Tested on
CPU (Python 3.11), fully offline (local corpus). Real web research needs CrewAI + a search API + an LLM.

> **Why it matters.** Automated research/briefing agents are a popular productivity build.
