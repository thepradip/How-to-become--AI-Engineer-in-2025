"""Summarization brain. Uses the shared chat UI — paste text, get a summary."""

from __future__ import annotations

from . import summarize as S

_SAMPLE = (
    "Large language models have transformed natural language processing. They are trained on vast "
    "text corpora and can generate fluent text, answer questions, and write code. However, they can "
    "hallucinate facts and are expensive to run at scale. Retrieval-augmented generation grounds "
    "their answers in trusted documents, reducing errors. Quantization and distillation make them "
    "cheaper to deploy. As a result, smaller models now rival last year's giants on many tasks."
)


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    cfg = state.get("config", {})
    n = int(cfg.get("n_sentences", 2))
    text = _SAMPLE if message.strip().lower() in {"demo", "example", ""} else message
    if len(S.split_sentences(text)) < 2:
        return Reply("Give me a longer passage (a few sentences) to summarize, or type `demo`.")
    summary = S.extractive_summary(text, n)
    return [Reply(f"**Extractive summary ({n} sentences):**"), Reply(summary, "text"),
            Reply(f"_Compressed {len(text)} → {len(summary)} chars._", "text")]
