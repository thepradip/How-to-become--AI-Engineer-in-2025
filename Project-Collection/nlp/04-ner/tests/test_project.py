import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import ner as N  # noqa: E402

TXT = "Jane Doe joined Acme Corp on March 3, 2026 earning $250,000. Email jane@acme.com about 15% equity."


def test_extracts_core_entity_types():
    ents = N.extract_entities(TXT)
    labels = {e["label"] for e in ents}
    assert "EMAIL" in labels and "MONEY" in labels and "PERCENT" in labels
    assert "DATE" in labels
    texts = " ".join(e["text"] for e in ents)
    assert "jane@acme.com" in texts and "250,000" in texts.replace("$", "")


def test_handler_demo():
    from src import handler as H

    out = H.respond("demo", {"config": {}})
    assert isinstance(out, list)
