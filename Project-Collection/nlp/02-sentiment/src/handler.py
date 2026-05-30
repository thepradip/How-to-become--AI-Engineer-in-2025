"""Sentiment brain — shared text engine."""

from _shared.text import make_text_handler

from .data_loader import LABEL, TEXT, load

respond = make_text_handler(
    load_df=load, text_col=TEXT, label_col=LABEL, noun="review",
    help_text=("I classify review **sentiment** (TF-IDF + NB / LogReg / SVM). Type `train`, then paste "
               "a review. For higher accuracy, fine-tune DistilBERT (see README)."),
)
