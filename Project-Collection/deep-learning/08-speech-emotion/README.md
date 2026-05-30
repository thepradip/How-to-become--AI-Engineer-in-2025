# DL 08 · Speech Emotion Recognition 🟡

**Problem.** Detect the emotion in a spoken clip (neutral, happy, sad, angry, …) — used in call-centre
analytics, voice assistants and UX research.

**What you build.** An audio → feature → MLP-classifier pipeline in the shared chat UI. Features are
**MFCCs** (via librosa) when available, with a numpy spectral-band fallback so it runs offline.

## Dataset (real)
**RAVDESS** emotional-speech audio (Zenodo). Extract MFCCs with `librosa`, then train the MLP. The
offline demo/tests use synthetic, class-dependent waveforms.

## Run it
```bash
pip install -r requirements.txt           # add: pip install librosa soundfile
pytest -q
streamlit run app.py      # "train" → "demo"
```

## What you learned
Turning raw audio into fixed-length features (MFCCs/spectral bands) · classifying non-image, non-text
data · why feature engineering still matters. **Extensions:** real RAVDESS, a 1-D CNN over
spectrograms, data augmentation (pitch/time shift).

## Tested on
CPU (Python 3.11) — smoke test covers feature extraction, the model, training on separable synthetic
classes (>40% acc, chance = 17%), and the handler. No GPU; no librosa required for tests.

> **Why it matters.** Voice/audio analytics (sentiment, emotion, keyword spotting) is a growing niche.
