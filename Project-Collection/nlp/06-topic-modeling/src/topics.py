"""Topic modeling: discover themes in a document collection.

Compares two classic unsupervised methods — **NMF** (on TF-IDF) and **LDA** (on counts)
— both shipped in scikit-learn (offline). The modern embedding-based **BERTopic** is
documented as the production upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_TOPICS = {
    "sports": "team match goal player season coach league score win tournament",
    "technology": "software model data cloud chip algorithm network code device app",
    "food": "recipe flavor cook bake fresh ingredient dish taste meal kitchen",
    "finance": "market stock price invest bank rate fund profit trade economy",
}


@dataclass
class TopicResult:
    method: str
    topics: pd.DataFrame   # topic_id -> top words
    doc_topics: np.ndarray


def synthetic_docs(n: int = 200, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    vocabs = {k: v.split() for k, v in _TOPICS.items()}
    docs = []
    for _ in range(n):
        topic = rng.choice(list(vocabs))
        words = rng.choice(vocabs[topic], size=rng.integers(12, 25)).tolist()
        docs.append(" ".join(words))
    return docs


def fit_topics(docs: list[str], n_topics: int = 4, method: str = "NMF", top_n: int = 8) -> TopicResult:
    if method == "NMF":
        vec = TfidfVectorizer(stop_words="english", min_df=2)
        X = vec.fit_transform(docs)
        model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    else:  # LDA on raw counts
        vec = CountVectorizer(stop_words="english", min_df=2)
        X = vec.fit_transform(docs)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    W = model.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for k, comp in enumerate(model.components_):
        top = vocab[comp.argsort()[::-1][:top_n]]
        rows.append({"topic": k, "top_words": ", ".join(top)})
    return TopicResult(method=method, topics=pd.DataFrame(rows), doc_topics=W.argmax(axis=1))
