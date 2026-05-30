"""Credit-risk brain — thin wrapper over the shared tabular engine (classification)."""

from _shared.tabular import make_handler

from .data_loader import POSITIVE, TARGET, load

respond = make_handler(
    load_df=load,
    target=TARGET,
    task="classification",
    positive=POSITIVE,
    help_text=(
        "I score loan applicants as **good** vs **bad** credit risk, comparing several algorithms "
        "+ a stacking ensemble. Try:\n- `train` · `compare` · `drivers` · `demo`"
    ),
)
