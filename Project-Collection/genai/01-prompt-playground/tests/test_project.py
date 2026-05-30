import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared import llm  # noqa: E402


def test_mock_is_extractive_with_context():
    prompt = "Context: RAG reduces hallucination by grounding answers. \nQuestion: what does RAG reduce?"
    out = llm.mock_complete(prompt)
    assert "hallucination" in out.lower()


def test_complete_returns_text_and_provider():
    res = llm.complete("hello")
    assert "text" in res and "provider" in res
    assert res["provider"] in {"openai", "anthropic", "ollama", "mock"}


def test_handler_responds():
    from src import handler as H

    out = H.respond("explain prompting", {"config": {}})
    assert isinstance(out, list)
