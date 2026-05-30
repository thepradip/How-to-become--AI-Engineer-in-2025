import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import recommender as R  # noqa: E402


def test_ratings_generated():
    df = R.synthetic_ratings(100, 50)
    assert {"user_id", "item_id", "rating"}.issubset(df.columns)
    assert df["rating"].between(1, 5).all()


def test_compares_methods():
    res = R.evaluate(R.synthetic_ratings(300, 120))
    assert len(res.leaderboard) == 3
    assert {"method", "RMSE"}.issubset(res.leaderboard.columns)
    # SVD should not be worse than the global-mean baseline on learnable data.
    gm = res.leaderboard.set_index("method").loc["Global mean", "RMSE"]
    svd = res.leaderboard.set_index("method").loc["SVD (matrix factorization)", "RMSE"]
    assert svd <= gm + 0.05


def test_recommend_returns_k():
    res = R.evaluate(R.synthetic_ratings(200, 80))
    recs = R.recommend(res, 0, k=5)
    assert len(recs) <= 5 and "predicted_rating" in recs.columns


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("recommend", state), list)
