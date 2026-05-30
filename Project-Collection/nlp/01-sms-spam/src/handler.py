"""SMS spam brain — thin wrapper over the shared text engine."""

from _shared.text import make_text_handler

from .data_loader import LABEL, TEXT, load

respond = make_text_handler(
    load_df=load, text_col=TEXT, label_col=LABEL, noun="message",
    help_text=("I flag **spam** SMS messages (TF-IDF + NB / LogReg / SVM compared). "
               "Type `train`, then just paste a message to classify it."),
)
