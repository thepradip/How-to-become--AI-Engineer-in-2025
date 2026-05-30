"""Fraud-detection brain — shared tabular engine with SMOTE for the rare-fraud class."""

from _shared.tabular import make_handler

from .data_loader import POSITIVE, TARGET, load

respond = make_handler(
    load_df=load,
    target=TARGET,
    task="classification",
    positive=POSITIVE,
    use_smote=True,  # oversample the rare fraud class during training
    help_text=(
        "I flag **fraudulent transactions** in a highly imbalanced dataset, using SMOTE + several "
        "algorithms + a stacking ensemble. Recall on fraud is what matters. Try:\n"
        "- `train` · `compare` · `drivers` · `demo`"
    ),
)
