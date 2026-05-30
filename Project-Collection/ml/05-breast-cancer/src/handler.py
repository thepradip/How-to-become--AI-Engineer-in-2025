"""Breast-cancer diagnosis brain — a thin wrapper over the shared tabular engine.

All multi-algorithm training / comparison / explanation lives in _shared/tabular.py.
Here we only say *what* the data and target are.
"""

from _shared.tabular import make_handler

from .data_loader import POSITIVE, TARGET, load

respond = make_handler(
    load_df=load,
    target=TARGET,
    task="classification",
    positive=POSITIVE,
    help_text=(
        "I diagnose tumours as **malignant** or **benign** from biopsy measurements, "
        "comparing several algorithms + a stacking ensemble. Try:\n"
        "- `train` — train & compare models\n- `compare` — leaderboard\n"
        "- `drivers` — which measurements matter most\n- `demo` — diagnose an example biopsy"
    ),
)
