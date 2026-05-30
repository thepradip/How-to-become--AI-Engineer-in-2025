"""House-price prediction brain — thin wrapper over the shared tabular engine (regression)."""

from _shared.tabular import make_handler

from .data_loader import TARGET, load

respond = make_handler(
    load_df=load,
    target=TARGET,
    task="regression",
    help_text=(
        "I estimate **house sale prices** from property features, comparing several regressors "
        "+ a stacking ensemble. Try:\n- `train` — train & compare models\n- `compare` — leaderboard\n"
        "- `drivers` — which features drive price\n- `demo` — predict a price for an example home"
    ),
)
