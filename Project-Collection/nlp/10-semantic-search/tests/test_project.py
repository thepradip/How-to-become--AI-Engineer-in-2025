import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import search as S  # noqa: E402


def test_ranks_relevant_job_first():
    resume = "Python PyTorch RAG LLM fine-tuning AWS machine learning"
    ranked = S.rank(resume, S.SAMPLE_JOBS)
    assert {"match", "document"}.issubset(ranked.columns)
    # The ML engineering job should top the list for this profile.
    assert "machine learning engineer" in ranked.iloc[0]["document"].lower()
    # Scores are sorted descending.
    assert list(ranked["match"]) == sorted(ranked["match"], reverse=True)


def test_handler_demo():
    from src import handler as H

    out = H.respond("demo", {"config": {}})
    assert isinstance(out, list)
