# Agents 13 · Agent Harness / Orchestration 🔴 · *capstone*

**Problem.** A loop isn't a product. Shipping agents needs a **harness**: a skill/tool **registry**, a
**router/orchestrator**, **observability** (traces + metrics), and **retries**. This capstone builds
those pieces.

**What you build.** A harness that routes a task to the right skill, executes with retries, and returns
a full **trace + metrics** (steps, attempts, approx tokens) — in the shared chat UI. Production maps to
the **Claude Agent SDK** / LangGraph + a tracer (Langfuse/LangSmith).

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "12 * (3 + 4)", "upper: hi", "echo this"
```

## What you learned
Skill registry + routing · observability (traces/metrics) · retries & error handling · the gap between
a demo loop and a production harness. **Production:** Claude Agent SDK, LangGraph checkpoints/rollback,
tracing, evals-in-CI (ties back to GenAI 10).

## Tested on
CPU (Python 3.11), fully offline. No GPU/keys.

> **Why it matters.** Turning a prototype agent into something observable, retry-safe and
> deployable is exactly what clients pay senior engineers for.
