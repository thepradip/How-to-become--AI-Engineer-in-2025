"""Topic-modeling brain. Uses the shared chat UI."""

from __future__ import annotations

from . import topics as T


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    cmd = message.strip().lower()
    docs = state.setdefault("docs", T.synthetic_docs(200))

    if cmd in {"nmf", "lda"}:
        res = T.fit_topics(docs, method=cmd.upper())
        return [Reply(f"**{res.method} topics** discovered:"), Reply(res.topics, "table")]

    if cmd in {"discover", "train", "go", "start", ""}:
        nmf = T.fit_topics(docs, method="NMF")
        lda = T.fit_topics(docs, method="LDA")
        return [Reply("Discovered topics with **NMF** and **LDA**."),
                Reply("**NMF:**"), Reply(nmf.topics, "table"),
                Reply("**LDA:**"), Reply(lda.topics, "table"),
                Reply("Paste a document to see which topic it best matches.", "text")]

    # Otherwise: assign the user's text to the nearest NMF topic.
    res = T.fit_topics(docs + [message], method="NMF")
    topic_id = int(res.doc_topics[-1])
    words = res.topics.loc[res.topics["topic"] == topic_id, "top_words"].iloc[0]
    return [Reply(f"Best-matching topic **#{topic_id}**: _{words}_")]
