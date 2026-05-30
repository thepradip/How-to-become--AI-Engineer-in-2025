# ML 04 · Credit Risk / Loan Default 🟡

**Problem.** Score a loan applicant as a **good** or **bad** credit risk. Lenders use exactly this to
approve/deny and price loans, so it's a high-value, heavily-regulated ML task (explainability matters).

**What you build.** A multi-algorithm classifier (Logistic Regression, Random Forest, Gradient
Boosting + stacking ensemble) with feature-importance explanations, in the shared chat UI.

## Dataset (real)
**Statlog German Credit** via OpenML (`credit-g`, 1,000 applicants, 20 features). `download_data.py`
fetches it; offline → synthetic fallback.

## Approach
Shared `_shared/tabular.py` engine: leak-free preprocessing → compare four algorithms → best by
ROC-AUC → explain drivers. We watch **recall on the "bad" class** (missing a default is costly).

## Run it
```bash
pip install -r requirements.txt
python src/download_data.py
pytest -q
streamlit run app.py
```

## What you learned
Cost-sensitive classification · the importance of explainability in regulated lending · comparing
models fairly.

## Extensions
Add a probability-of-default → expected-loss calculation · adverse-action reason codes (SHAP) ·
fairness checks across groups.

## Tested on
CPU (Python 3.11), offline on the synthetic sample; real data via OpenML when online. No GPU. ✅

> **Why it matters.** Credit scoring, lead scoring, and risk models are common fintech projects.
