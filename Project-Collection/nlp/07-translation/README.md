# NLP 07 · Machine Translation 🟡

**Problem.** Translate text between languages — localisation, support, content. Modern MT is neural
seq2seq; Meta's **NLLB-200** covers 200 languages in one model.

**What you build.** A chat translator. The **offline demo** uses a tiny word-lookup table (so it runs
anywhere); the **real path** calls NLLB-200 via HF transformers.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # pick a language, type English text
```
Real neural translation:
```python
# pip install transformers sentencepiece
from src.translate import translate_nllb
translate_nllb("Good morning, my friend", "fr")   # NLLB-200
```

## What you learned
The difference between dictionary lookup and neural MT · NLLB language codes · why context &
word-order need a sequence model. **Extension:** fine-tune on a domain glossary; add BLEU evaluation.

## Tested on
CPU (Python 3.11), fully offline (word-lookup demo). The NLLB path needs `transformers` +
`sentencepiece` + a model download (GPU recommended for speed).

> **Why it matters.** Localisation pipelines and multilingual chat are common product needs.
