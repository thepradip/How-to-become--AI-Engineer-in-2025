"""Fake-news brain — shared text engine."""

from _shared.text import make_text_handler

from .data_loader import LABEL, TEXT, load

respond = make_text_handler(
    load_df=load, text_col=TEXT, label_col=LABEL, noun="headline",
    help_text=("I classify news as **real** vs **fake** (TF-IDF + NB / LogReg / SVM). Type `train`, "
               "then paste a headline. (Educational — real fake-news detection is much harder.)"),
)
