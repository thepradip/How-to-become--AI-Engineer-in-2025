"""Research agent — search multiple sources, then synthesize a cited answer.

A research agent gathers information from several sources and combines it. Offline we
search a small local corpus (TF-IDF) and synthesize an extractive summary with
citations; the README shows the **CrewAI** multi-tool version (web search + scrape + LLM).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = [
    {"title": "RAG basics", "text": "Retrieval-augmented generation grounds LLM answers in retrieved documents to reduce hallucination."},
    {"title": "Vector DBs", "text": "Vector databases like Chroma and Qdrant store embeddings for fast similarity search in RAG."},
    {"title": "Evaluation", "text": "RAG systems are evaluated on faithfulness and answer relevance, often with Ragas."},
    {"title": "Agents", "text": "Agents use tools and a reasoning loop; frameworks include LangGraph and CrewAI."},
    {"title": "Fine-tuning", "text": "LoRA and QLoRA fine-tune LLMs cheaply; Unsloth speeds this up on a single GPU."},
]


def search(query: str, k: int = 3) -> list[dict]:
    texts = [c["text"] for c in CORPUS]
    vec = TfidfVectorizer(stop_words="english").fit(texts + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(texts))[0]
    order = sims.argsort()[::-1][:k]
    return [{**CORPUS[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0]


def run(query: str) -> dict:
    """Search → synthesize a short, cited research note."""
    hits = search(query)
    if not hits:
        return {"summary": "No relevant sources found.", "sources": []}
    summary = " ".join(f"{h['text']} [{h['title']}]" for h in hits)
    return {"summary": summary, "sources": hits}
