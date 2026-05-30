import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import automl as A  # noqa: E402


def test_optuna_tunes_and_compares():
    res = A.run_optuna(n_trials=5)              # small for a fast test
    assert isinstance(res.best_params, dict) and res.best_params
    assert 0.5 <= res.best_cv_score <= 1.0
    assert 0.5 <= res.tuned_test_auc <= 1.0
    # On this clean dataset both models score high; tuned shouldn't be much worse.
    assert res.tuned_test_auc >= res.default_test_auc - 0.05
    assert len(res.history) == 5


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {"n_trials": 5}}
    out = H.respond("tune", state)
    assert isinstance(out, list) and "result" in state
