"""Extractive Question Answering — find the answer span inside a context passage.

Offline baseline: split the context into sentences and return the one most similar to
the question (TF-IDF cosine). The production path uses a fine-tuned **BERT/DistilBERT
SQuAD** model that pinpoints the exact answer span (``answer_transformer``).
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_CONTEXT = (
    "Retrieval-augmented generation, or RAG, grounds a language model's answers in "
    "retrieved documents to reduce hallucination. A RAG pipeline first embeds documents "
    "into a vector database. At query time it retrieves the most relevant chunks and "
    "passes them to the model as context. Evaluation uses metrics like faithfulness and "
    "answer relevance. RAG is the dominant pattern for production LLM applications in 2026."
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def answer(question: str, context: str = SAMPLE_CONTEXT) -> dict:
    """Return the most relevant sentence as the answer (extractive baseline)."""
    sents = _sentences(context)
    if not sents:
        return {"answer": "", "score": 0.0}
    vec = TfidfVectorizer(stop_words="english").fit(sents + [question])
    sims = cosine_similarity(vec.transform([question]), vec.transform(sents))[0]
    best = int(sims.argmax())
    return {"answer": sents[best], "score": round(float(sims[best]), 3)}


def answer_transformer(question: str, context: str = SAMPLE_CONTEXT) -> dict:
    """Real span extraction with a SQuAD-fine-tuned model (downloads). Documented path."""
    from transformers import pipeline

    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    r = qa(question=question, context=context)
    return {"answer": r["answer"], "score": round(float(r["score"]), 3)}
