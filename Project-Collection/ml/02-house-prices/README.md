# ML 02 · House Price Regression 🟢

**Problem.** Estimate a home's sale price from its features — the canonical regression task and a
real service (automated valuation models power Zillow-style estimates and lending decisions).

**What you build.** A regression pipeline that compares **Ridge, Random Forest, Gradient Boosting +
a stacking ensemble**, reports R²/MAE/RMSE, and ranks the features that drive price — in the shared
chat UI.

## Dataset (real)
**Ames House Prices** via OpenML (`house_prices`, ~1,460 homes, 80 features). `download_data.py`
fetches it into `./data`; offline, a schema-realistic synthetic sample is used.

## Approach
Leak-free preprocessing (impute + scale numeric, one-hot categorical) → train & compare four
regressors (shared `_shared/tabular.py`) → pick the best by R² → explain with gradient-boosting
importances.

## Run it
```bash
pip install -r requirements.txt
python src/download_data.py     # real Ames data
pytest -q
streamlit run app.py            # "train" → "compare" → "drivers" → "demo"
```

## What you learned
Regression metrics (R²/MAE/RMSE and when each matters) · comparing linear vs. tree vs. ensemble
models · communicating price drivers.

## Tested on
CPU (Python 3.11). `pytest` runs offline on the synthetic sample (R² > 0.7); the app uses the real
Ames data when online. No GPU. ✅

> **Freelance relevance.** Price/valuation models (real estate, used goods, insurance) are a steady
> source of paid regression work.
