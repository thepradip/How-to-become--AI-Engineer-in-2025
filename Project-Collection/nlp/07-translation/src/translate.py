"""Machine translation.

Real translation uses Meta's **NLLB-200** (200 languages) via HF transformers — that's
the documented production path (``translate_nllb``). For an offline, dependency-free
*demo* we ship a tiny word-lookup translator so the app and tests run anywhere. It is
intentionally simple — real MT is a neural seq2seq model, not a dictionary.
"""

from __future__ import annotations

# Small demo dictionaries (en → X). Real coverage comes from NLLB.
_DICT = {
    "fr": {"hello": "bonjour", "world": "monde", "the": "le", "cat": "chat", "is": "est",
           "good": "bon", "morning": "matin", "thank": "merci", "you": "vous", "love": "amour",
           "water": "eau", "friend": "ami", "book": "livre", "house": "maison", "today": "aujourd'hui"},
    "es": {"hello": "hola", "world": "mundo", "the": "el", "cat": "gato", "is": "es",
           "good": "bueno", "morning": "mañana", "thank": "gracias", "you": "tú", "love": "amor",
           "water": "agua", "friend": "amigo", "book": "libro", "house": "casa", "today": "hoy"},
    "de": {"hello": "hallo", "world": "welt", "the": "das", "cat": "katze", "is": "ist",
           "good": "gut", "morning": "morgen", "thank": "danke", "you": "du", "love": "liebe",
           "water": "wasser", "friend": "freund", "book": "buch", "house": "haus", "today": "heute"},
}
LANGUAGES = {"fr": "French", "es": "Spanish", "de": "German"}

# NLLB language codes for the real path.
_NLLB = {"fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn"}


def translate(text: str, target: str = "fr") -> str:
    """Offline word-lookup demo translation (unknown words pass through)."""
    table = _DICT.get(target, {})
    return " ".join(table.get(w.lower(), w) for w in text.split())


def translate_nllb(text: str, target: str = "fr") -> str:
    """Real neural translation with NLLB-200 (downloads a model). Documented path."""
    from transformers import pipeline

    pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M",
                    src_lang="eng_Latn", tgt_lang=_NLLB.get(target, "fra_Latn"))
    return pipe(text)[0]["translation_text"]
