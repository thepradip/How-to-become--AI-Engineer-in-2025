# Agents 08 · A2A — Agents Across Frameworks 🔴

**Problem.** Real systems mix agents built with different frameworks. The **A2A (Agent-to-Agent)
protocol** standardises how they advertise skills (agent cards) and exchange tasks, so a LangGraph
agent can call a CrewAI agent and vice-versa.

**What you build.** A tiny A2A message schema + agent cards + two agents (a LangGraph-style math-bot
and a CrewAI-style writer-bot) collaborating on a task, with the message exchange shown in the chat UI.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # give a math expression; watch the two agents talk
```

## What you learned
The A2A concepts — **agent cards / discovery**, a shared **message format**, task delegation across
frameworks. **Production:** the official A2A SDK + framework adapters (LangGraph/CrewAI/ADK).

## Tested on
CPU (Python 3.11), fully offline (in-process A2A simulation). No GPU/keys.

> **Freelance relevance.** Interop matters as orgs accumulate agents from different teams/vendors.
