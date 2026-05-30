# DL 07 · LSTM Time-Series Forecasting 🟡

**Problem.** Forecast a sequence (energy load, sensor readings, sales) with a recurrent neural net
that learns temporal patterns directly from raw values — complementing the feature-based approach in
ML 07.

**What you build.** An LSTM that reads a window of recent values and predicts the next, then forecasts
multiple steps recursively — visualised in the shared chat UI.

## Dataset
Synthetic seasonal series (trend + sine + noise) so it runs offline. Swap in a real univariate series
(e.g., hourly energy demand) by replacing `synthetic_series`.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "train"
```

## What you learned
RNN/LSTM mechanics · windowing a series for supervised learning · recursive multi-step forecasting
and its error accumulation. **Extensions:** multivariate inputs, teacher forcing, seq2seq, attention.

## Tested on
CPU (Python 3.11) — smoke test (windowing, forward, training reduces loss, multi-step forecast). No GPU.

> **Why it matters.** Sequence forecasting for IoT/energy/finance clients.
