import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import segmentation as S  # noqa: E402


def test_rfm_built():
    rfm = S.build_rfm(S.synthetic_transactions(200))
    assert set(S.RFM_COLS).issubset(rfm.columns)
    assert len(rfm) > 0


def test_compares_clustering_algorithms():
    res = S.segment(S.build_rfm(S.synthetic_transactions(300)))
    assert {"algorithm", "k", "silhouette"}.issubset(res.leaderboard.columns)
    assert res.best_algo in {"KMeans", "Agglomerative", "DBSCAN"}
    assert res.best_k >= 2
    assert "label" in res.profiles.columns


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("segment", state), list)
    assert isinstance(H.respond("profiles", state), list)
