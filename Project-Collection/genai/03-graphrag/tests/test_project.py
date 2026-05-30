import pathlib
import sys

import pytest

pytest.importorskip("networkx")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import graphrag as G  # noqa: E402


def test_builds_graph():
    g = G.build_graph(G.SAMPLE_DOCS)
    assert g.number_of_nodes() > 3 and g.number_of_edges() > 0
    assert "Acme" in g.nodes


def test_neighborhood_returns_facts():
    g = G.build_graph(G.SAMPLE_DOCS)
    facts = G.neighborhood(g, "Acme", hops=1)
    assert any("Acme" in f for f in facts)


def test_answer_finds_entity():
    res = G.answer(G.SAMPLE_DOCS, "who founded Acme?")
    assert res["entity"] == "Acme" and res["facts"]


def test_handler_graph():
    from src import handler as H

    assert isinstance(H.respond("graph", {"config": {}}), list)
