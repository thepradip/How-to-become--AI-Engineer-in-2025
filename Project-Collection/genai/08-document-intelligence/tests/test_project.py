import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import docai as D  # noqa: E402


def test_extracts_invoice_fields():
    f = D.extract_fields(D.SAMPLE_INVOICE)
    assert f["invoice_number"] == "INV-2026-00471"
    assert f["date"] == "2026-03-14"
    assert "1,815" in f["total"]


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {}}), list)
