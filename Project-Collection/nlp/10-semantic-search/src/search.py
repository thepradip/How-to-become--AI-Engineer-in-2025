"""Semantic résumé ↔ job matching.

Rank job postings by how well they match a résumé (or vice-versa). Offline baseline:
TF-IDF + cosine similarity (a real, useful matcher). Production upgrade: dense
**sentence-transformer** embeddings, which capture meaning beyond shared keywords.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_JOBS = [
    "Machine learning engineer: build RAG pipelines, fine-tune LLMs, deploy on AWS, Python, PyTorch.",
    "Frontend developer: React, Next.js, TypeScript, CSS, build responsive web apps.",
    "Data analyst: SQL, dashboards, Excel, Tableau, business reporting and KPIs.",
    "Computer vision engineer: PyTorch, object detection, image segmentation, deploy on edge devices.",
    "DevOps engineer: Kubernetes, Docker, CI/CD, Terraform, cloud infrastructure on GCP.",
]


def rank(query: str, documents: list[str], top_k: int = 5) -> pd.DataFrame:
    """Return documents ranked by cosine similarity to the query (TF-IDF)."""
    vec = TfidfVectorizer(stop_words="english").fit(documents + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(documents))[0]
    order = sims.argsort()[::-1][:top_k]
    return pd.DataFrame({"match": [round(float(sims[i]), 3) for i in order],
                         "document": [documents[i] for i in order]})


def rank_semantic(query: str, documents: list[str], top_k: int = 5) -> pd.DataFrame:
    """Dense-embedding ranking with sentence-transformers (downloads a model)."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_docs = model.encode(documents, convert_to_tensor=True)
    emb_q = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(emb_q, emb_docs)[0].tolist()
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return pd.DataFrame({"match": [round(scores[i], 3) for i in order],
                         "document": [documents[i] for i in order]})
