"""Text summarization — extractive (offline) + optional abstractive (BART).

Extractive summarization picks the most important *existing* sentences (fast, faithful,
no model download). Abstractive summarization (BART/Pegasus) *writes* new sentences and
reads more fluently but needs a model. We ship extractive for offline use and document
the transformer path.
"""

from __future__ import annotations

import re
from collections import Counter

_STOP = set("a an the and or but if while of to in on for with as at by from is are was were be "
            "this that these those it its their his her they we you i he she him them our your".split())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOP and len(w) > 2]


def extractive_summary(text: str, n_sentences: int = 3) -> str:
    """Return the top-N most informative sentences, in their original order."""
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return text.strip()
    freq = Counter(w for s in sentences for w in _words(s))
    if not freq:
        return " ".join(sentences[:n_sentences])
    peak = max(freq.values())
    scores = []
    for i, s in enumerate(sentences):
        ws = _words(s)
        score = sum(freq[w] / peak for w in ws) / (len(ws) + 1e-6)  # normalised, length-adjusted
        scores.append((score, i, s))
    chosen = sorted(scores, reverse=True)[:n_sentences]
    return " ".join(s for _, _, s in sorted(chosen, key=lambda t: t[1]))


def abstractive_summary(text: str, max_len: int = 120) -> str:
    """Optional: HF BART summarizer (downloads a model). Documented path."""
    from transformers import pipeline

    summ = pipeline("summarization", model="facebook/bart-large-cnn")
    return summ(text, max_length=max_len, min_length=30, do_sample=False)[0]["summary_text"]
