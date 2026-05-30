"""GraphRAG — retrieval over a knowledge graph built from text.

Standard RAG retrieves isolated chunks. **GraphRAG** builds a graph of entities and
their relationships, so it can answer "how is X connected to Y?" and gather
multi-hop context that flat retrieval misses. We extract entities, link those that
co-occur in a sentence, then retrieve a query entity's neighbourhood as context.

Offline core uses ``networkx`` + simple entity extraction + the shared LLM client.
Production GraphRAG uses an LLM to extract typed triples into Neo4j (documented).
"""

from __future__ import annotations

import re

import networkx as nx

SAMPLE_DOCS = [
    "Ada founded Acme in London. Ada hired Bob as the lead engineer.",
    "Bob built the Search product at Acme. Carol manages the Search product.",
    "Acme partnered with Globex on the Cloud project. Carol leads the Cloud project.",
]

_STOP = {"The", "A", "An", "In", "On", "At", "As", "And"}


def extract_entities(sentence: str) -> list[str]:
    """Capitalised tokens as entities (simple, offline heuristic)."""
    cands = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    return [c for c in cands if c not in _STOP]


def build_graph(docs: list[str]) -> nx.Graph:
    """Nodes = entities; edges connect entities co-occurring in a sentence."""
    g = nx.Graph()
    for doc in docs:
        for sent in re.split(r"(?<=[.!?])\s+", doc):
            ents = list(dict.fromkeys(extract_entities(sent)))
            for e in ents:
                g.add_node(e)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    w = g.get_edge_data(ents[i], ents[j], {}).get("weight", 0) + 1
                    g.add_edge(ents[i], ents[j], weight=w, context=sent.strip())
    return g


def neighborhood(g: nx.Graph, entity: str, hops: int = 1) -> list[str]:
    """Return context sentences from edges within ``hops`` of ``entity``."""
    if entity not in g:
        return []
    nodes = nx.ego_graph(g, entity, radius=hops).nodes
    facts = {d["context"] for u, v, d in g.edges(nodes, data=True) if "context" in d}
    return sorted(facts)


def answer(docs: list[str], question: str) -> dict:
    """Find a known entity in the question, gather its graph context, then answer."""
    from _shared.llm import complete

    g = build_graph(docs)
    hit = next((n for n in g.nodes if re.search(rf"\b{re.escape(n)}\b", question)), None)
    facts = neighborhood(g, hit, hops=2) if hit else []
    context = "\n".join(facts) if facts else " ".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer from the context."
    res = complete(prompt)
    return {"answer": res["text"], "provider": res["provider"],
            "entity": hit, "facts": facts,
            "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges()}
