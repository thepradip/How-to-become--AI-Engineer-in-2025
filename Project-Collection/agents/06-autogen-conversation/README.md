# Agents 06 · Conversational Multi-Agent (AutoGen) 🟡

**Problem.** Some answers improve when agents **debate** — a solver proposes, a critic pushes back, the
solver revises. AutoGen/AG2 pioneered this multi-turn conversational pattern.

**What you build.** A Solver↔Critic dialogue that refines an answer over rounds, with the full
transcript, in the shared chat UI. Offline it's deterministic; production uses **AutoGen/AG2**.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # give a task; watch the agents refine it
```
Production (AutoGen):
```python
# pip install autogen-agentchat
# AssistantAgent (solver) + AssistantAgent (critic) in a RoundRobinGroupChat.
```

## What you learned
Conversational refinement · solver/critic roles · termination conditions. **Extensions:** AutoGen
GroupChat, tool-using agents, human-in-the-loop.

## Tested on
CPU (Python 3.11), fully offline (deterministic dialogue). Real conversations need AutoGen + an LLM.

> **Freelance relevance.** Debate/critique loops improve quality on open-ended generation tasks.
