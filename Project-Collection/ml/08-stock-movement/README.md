# ML 08 · Stock Movement Prediction (with honest backtest) 🔴

**Problem.** Can technical features predict whether a stock rises tomorrow? Mostly *no* — markets are
near-efficient. The real lesson here is **how to evaluate a trading idea without fooling yourself**:
strict chronological validation and a backtest against buy-and-hold.

**What you build.** Technical features (returns, SMA ratio, volatility, RSI) → compare Logistic
Regression / Random Forest / Gradient Boosting on a **chronological** split (no shuffling = no
look-ahead leakage) → **backtest** the best model's long/flat strategy vs. buy-and-hold. Shared chat UI.

## Dataset (real)
Daily prices via **yfinance** (`python src/download_data.py SPY`). Offline → a synthetic
geometric-Brownian-motion series.

## Why the leakage point matters
Shuffling time-series data lets the model "see the future" and produces fantastic, fake results. We
split by time and only ever predict forward.

## Run it
```bash
pip install -r requirements.txt
python src/download_data.py AAPL   # real prices (optional)
pytest -q
streamlit run app.py               # "analyze"
```

## What you learned
Time-series cross-validation pitfalls · technical features · backtesting vs. a benchmark · healthy
skepticism about "predicting the market."

## Tested on
CPU (Python 3.11), offline on a synthetic price series. No GPU. ✅

> ⚠️ Educational only — **not investment advice**.
