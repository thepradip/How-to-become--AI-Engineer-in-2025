# ML 05 · Breast-Cancer Diagnosis (explainable) 🟢

**Problem.** From a tumour biopsy's measurements, predict **malignant vs. benign** — a classic,
high-stakes medical screening aid. The cost of a false negative (missing a malignant tumour) is far
higher than a false positive, so recall matters.

**What you build.** A multi-algorithm classifier (Logistic Regression, Random Forest, Gradient
Boosting + a **stacking ensemble**) on the real UCI data, surfaced through the shared chat UI, with
feature-importance explanations a clinician can sanity-check.

## Dataset (real)
**UCI Wisconsin Breast Cancer Diagnostic** (569 biopsies, 30 features) — shipped inside scikit-learn
(`load_breast_cancer`), so it's the genuine UCI data and needs **no download** (runs fully offline).

## Approach
Same leak-free preprocessing + multi-algorithm comparison + stacking ensemble as the rest of the ML
track (shared engine in `_shared/tabular.py`). We report ROC-AUC / precision / **recall** / F1 and a
confusion matrix, and rank the most predictive measurements.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
streamlit run app.py     # chat: "train" → "compare" → "drivers" → "demo"
```

## Results (typical)
ROC-AUC ≈ **0.99** — the worst-area, worst-perimeter and worst-concave-points features dominate.

## What you learned
Comparing algorithms fairly · why **recall** is the metric to watch in medical screening · explaining
a model to non-technical (clinical) stakeholders.

## Extensions
Tune the decision threshold for high recall · add SHAP per-patient explanations · calibrate
probabilities.

## Tested on
CPU (Python 3.11), fully offline on the real bundled dataset. No GPU. ✅ `pytest` green.

> **Freelance relevance.** Diagnostic/triage classifiers with an explainability layer are common
> healthcare-AI engagements; the explanation step is often the actual deliverable.
