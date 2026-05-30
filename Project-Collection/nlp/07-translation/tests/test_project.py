import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import translate as Tr  # noqa: E402


def test_word_lookup_translation():
    assert Tr.translate("hello world", "fr") == "bonjour monde"
    assert Tr.translate("hello world", "es") == "hola mundo"
    assert Tr.translate("the cat", "de") == "das katze"


def test_unknown_words_pass_through():
    assert "zzz" in Tr.translate("zzz hello", "fr")


def test_handler_demo():
    from src import handler as H

    out = H.respond("demo", {"config": {"target": "es"}})
    assert isinstance(out, list)
