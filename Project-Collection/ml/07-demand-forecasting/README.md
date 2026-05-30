# ML 07 · Demand Forecasting 🟡

**Problem.** Predict future daily sales so a business can plan inventory and staffing. Under-forecast
→ stock-outs; over-forecast → waste. A recurring, high-value analytics task.

**What you build.** Forecasting as supervised regression on **lag + calendar features**, comparing a
**seasonal-naive baseline** against Linear Regression, Random Forest and Gradient Boosting — judged on
MAE / RMSE / MAPE over a real holdout. Shared chat UI with an actual-vs-forecast plot.

## Dataset (real)
A daily store-sales series (`download_data.py`). Offline → a synthetic series with trend + weekly +
yearly seasonality.

## Approach
Build lag (1/7/14/28) and rolling-mean features + day-of-week/month → train on all but the last
`horizon` days → compare models on that holdout. **Beating the seasonal-naive baseline is the bar.**

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py     # "forecast" → "compare" → "plot"
```

## What you learned
Framing time series as supervised ML · honest backtesting against a baseline · forecasting metrics
(MAPE pitfalls). **Extensions:** add **Prophet** and **LightGBM**, multi-step recursive forecasting,
prediction intervals.

## Tested on
CPU (Python 3.11), fully offline on the synthetic series. No GPU. ✅

> **Freelance relevance.** Sales/demand forecasting is one of the most-requested analytics gigs.
