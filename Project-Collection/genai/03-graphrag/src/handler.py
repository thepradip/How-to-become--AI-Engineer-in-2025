"""GraphRAG brain. Uses the shared chat UI."""

from __future__ import annotations

from . import graphrag as G


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    docs = state.get("docs", G.SAMPLE_DOCS)
    msg = message.strip()

    if msg.lower() in {"", "help", "demo", "graph"}:
        g = G.build_graph(docs)
        edges = [{"from": u, "to": v, "weight": d["weight"]} for u, v, d in g.edges(data=True)]
        import pandas as pd
        return [Reply(f"Built a knowledge graph: **{g.number_of_nodes()} entities, "
                      f"{g.number_of_edges()} relations**."),
                Reply(pd.DataFrame(edges), "table"),
                Reply("Ask e.g. *what is Carol connected to?* or *who founded Acme?*", "text")]

    res = G.answer(docs, msg)
    parts = [Reply(f"`provider: {res['provider']}` · matched entity: **{res['entity'] or '—'}**", "text"),
             Reply(res["answer"], "text")]
    if res["facts"]:
        parts.append(Reply("**Graph context used:**\n" + "\n".join(f"- {f}" for f in res["facts"]), "text"))
    return parts
