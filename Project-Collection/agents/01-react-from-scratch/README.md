# Agents 01 · ReAct From Scratch 🟢

**Problem.** Before reaching for a framework, understand how an agent actually works: the **ReAct**
loop — **Reason** (Thought) then **Act** (call a tool), observe the result, repeat, then answer.

**What you build.** A ReAct executor with two tools (a safe calculator + a knowledge lookup) and a
transparent Thought/Action/Observation trace, in the shared chat UI. The policy is rule-based so it
runs offline; swapping an LLM in to choose actions is the only change for the real thing.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # try "12 * (3 + 4)" or "what is RAG?"
```

## What you learned
The ReAct loop · tool calling · safe tool execution (no `eval`) · why a visible trace matters for
debugging agents. **Next:** let an LLM pick actions (LangChain/LangGraph) — Agents 02+.

## Tested on
CPU (Python 3.11), fully offline (rule-based policy + safe calculator). No GPU/keys.

> **Freelance relevance.** Agent literacy starts here; clients want builders who understand the loop,
> not just glue frameworks together.
