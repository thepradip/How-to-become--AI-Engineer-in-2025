import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import crew as C  # noqa: E402


def test_pipeline_runs_all_roles():
    out = C.run("RAG")
    assert out.research and out.draft and out.final
    assert out.final.startswith("#")            # editor adds a title
    assert "hallucination" in out.draft.lower()  # researcher facts flowed through


def test_handler():
    from src import handler as H

    parts = H.respond("AI agents", {"config": {}})
    assert isinstance(parts, list) and len(parts) == 4   # researcher/writer/editor outputs
