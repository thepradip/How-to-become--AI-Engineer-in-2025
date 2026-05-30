"""Shared text-classification engine (TF-IDF + multiple algorithms).

Spam, sentiment and fake-news are all "classify this text" tasks. This module does
TF-IDF vectorisation + compares several classifiers (Naive Bayes, Logistic Regression,
Linear SVM), so each NLP project is a thin wrapper that just supplies its dataset. It
also builds a chat handler where the user simply **types text and gets a label** — a
perfect fit for the shared chat UI.

Tested with scikit-learn (offline). Each project's README documents the optional
transformer (DistilBERT/BERT) path for higher accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class TextResult:
    pipelines: dict
    leaderboard: pd.DataFrame
    best_name: str
    classes: list

    @property
    def best(self) -> Pipeline:
        return self.pipelines[self.best_name]


def _estimators() -> dict:
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }


def train_text_classifiers(texts, labels, test_size: float = 0.25, seed: int = 42) -> TextResult:
    """TF-IDF + compare classifiers; return a leaderboard and the best pipeline."""
    texts = list(map(str, texts))
    X_tr, X_te, y_tr, y_te = train_test_split(texts, list(labels), test_size=test_size,
                                              random_state=seed, stratify=list(labels))
    rows, pipelines = [], {}
    for name, est in _estimators().items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
                         ("clf", est)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"Algorithm": name,
                     "Accuracy": round(float(accuracy_score(y_te, pred)), 3),
                     "F1 (macro)": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 3)})
        pipelines[name] = pipe
    lb = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    classes = sorted(set(map(str, labels)))
    return TextResult(pipelines=pipelines, leaderboard=lb,
                      best_name=lb.iloc[0]["Algorithm"], classes=classes)


def make_text_handler(*, load_df: Callable[[], pd.DataFrame], text_col: str, label_col: str,
                      noun: str = "text", help_text: str = ""):
    """Chat handler: `train` / `compare`, otherwise classify whatever the user types."""

    def _ensure(state):
        if "result" not in state:
            df = load_df()
            state["result"] = train_text_classifiers(df[text_col], df[label_col])
        return state["result"]

    def respond(message: str, state: dict):
        from _shared.chat_ui import Reply

        cmd = message.strip().lower()
        if cmd in {"train", "go", "start"}:
            df = load_df()
            res = state["result"] = train_text_classifiers(df[text_col], df[label_col])
            return [Reply(f"Trained & compared {len(res.pipelines)} classifiers on {len(df)} {noun}s. "
                          f"Best: **{res.best_name}**."),
                    Reply(res.leaderboard, "table"),
                    Reply(f"Now just type any {noun} and I'll classify it.", "text")]
        if cmd in {"compare", "leaderboard"}:
            return [Reply("**Classifier comparison:**"), Reply(_ensure(state).leaderboard, "table")]
        if cmd in {"help", ""}:
            return Reply(help_text or f"Type a {noun} to classify, or `train` / `compare`.")

        # Otherwise: classify the user's text with the best model.
        res = _ensure(state)
        label = str(res.best.predict([message])[0])
        out = {f"Predicted": label}
        clf = res.best.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = res.best.predict_proba([message])[0].max()
            out["Confidence"] = f"{proba:.0%}"
        return [Reply(f"> {message}", "text"), Reply(out, "metric")]

    return respond
