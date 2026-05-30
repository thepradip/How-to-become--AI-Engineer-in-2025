# 🛠️ Project Collection — Hands-On AI Engineering

A complete, **build-it-yourself** companion course to [*How to Become an AI Engineer in 2026*](../index.html).
Each of the 53 projects solves a **real-world problem** end-to-end — documented Python, a working UI,
automated tests, and a professional README. Together they form a portfolio you can show in interviews
and adapt for production work.

> Built for **working professionals, coders, and non-coders**. If you can follow a recipe and run a
> command, you can ship these. Start with [`00-getting-started`](00-getting-started/) to learn how to
> drive AI coding tools (Claude Code, Codex, Lovable) so the AI writes most of the code *with* you.

---

## How every project is structured

```
<track>/NN-project-slug/
  README.md          # the problem, the real dataset/model, the approach, how to run, results
  requirements.txt   # pinned dependencies
  src/               # documented Python (docstrings + type hints)
  app.py             # the UI — a thin shim over the shared chat interface
  tests/             # pytest (unit + a smoke test)
  data/              # download script only — real data is fetched on first run, never committed
```

### One UI for every app — the shared **chat interface**
We do **not** rebuild a UI per project. Every app uses one shared chat component
([`_shared/chat_ui.py`](_shared/chat_ui.py)), so all ~53 projects feel the same and you focus on the
*AI*, not on plumbing. A project's `app.py` is just a few lines:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))   # add Project-Collection root
from _shared.chat_ui import run_chat_app
from src.handler import respond

run_chat_app(title="…", intro="…", respond=respond, examples=[...])
```

The chat UI renders **text, tables, charts, images, metrics and code** in the transcript, so a
churn model, a chest-X-ray classifier, a RAG bot and a multi-agent team all use the same surface.
A handful of flagship GenAI/agent apps also ship a **Next.js** showcase front-end.

### Levels
🟢 Beginner · 🟡 Intermediate · 🔴 Advanced — each track runs basic → advanced.

---

## Catalog (53 projects)

### `ml/` — Machine Learning · classic algorithms, training & hyperparameter tuning
| # | Project | Real data | Level |
|---|---------|-----------|-------|
| 01 | Customer Churn Prediction (XGBoost + SHAP) | Telco Churn (OpenML) | 🟢 |
| 02 | House Price Regression (GridSearchCV) | Ames Housing | 🟢 |
| 03 | Credit-Card Fraud Detection (imbalanced, SMOTE) | Kaggle creditcard | 🟡 |
| 04 | Credit Risk / Loan Default | LendingClub / UCI | 🟡 |
| 05 | Breast-Cancer Diagnosis (explainable) | UCI Wisconsin | 🟢 |
| 06 | Customer Segmentation (RFM + KMeans) | Online Retail | 🟡 |
| 07 | Demand Forecasting (Prophet + LightGBM) | Store sales | 🟡 |
| 08 | Stock Movement Prediction (backtested) | yfinance | 🔴 |
| 09 | Movie Recommender (collaborative filtering) | MovieLens | 🟡 |
| 10 | AutoML + Optuna + MLflow (capstone) | any tabular | 🔴 |

### `deep-learning/` — PyTorch · training, transfer learning, fine-tuning
CNN from scratch · Transfer-learning classifier · **Pneumonia X-ray + Grad-CAM** · Object detection
(YOLO11) · Segmentation (U-Net) · Traffic signs (GTSRB) · LSTM forecasting · **Speech-emotion** ·
Face verification · **Fine-tune a ViT + W&B sweep** (16 GB GPU).

### `nlp/` — classic NLP → transformers → LLM fine-tuning
SMS spam · Sentiment (DistilBERT) · Fake-news · NER (fine-tune BERT) · Summarization (BART) ·
Topic modeling (BERTopic) · Translation (NLLB) · Extractive QA · **LoRA/QLoRA LLM fine-tune
(Unsloth, 16 GB)** · Semantic résumé ↔ job search.

### `genai/` — LLM apps, RAG, image/video, local inference, guardrails
Prompt playground · **RAG with citations** · **GraphRAG** · Local LLM chat (**Ollama + vLLM**) ·
**Bonsai / BitNet 1-bit** text gen · Image generation (SD 3.5 / FLUX) · **Video generation
(Wan 2.2 / LTX)** · Document intelligence / **OCR** · Structured extraction + **guardrails** ·
Evals + guardrails harness.

### `agents/` — frameworks & real agent types
ReAct from scratch · **SQL agent (LangGraph)** · **Web/browser agent** · **Research agent (CrewAI)** ·
Multi-agent content team (CrewAI) · Conversational multi-agent (**AutoGen/AG2**) · Type-safe tool
agent (**PydanticAI**) · **A2A** multi-framework · **Healthcare agent** · **Finance agent** ·
**Voice agent** (Whisper → LLM → TTS) · **Guardrails/safety** (NeMo) · **Agent harness/orchestration**.

---

## Quick start

```bash
cd Project-Collection
python -m venv .venv && source .venv/bin/activate    # or: uv venv && source .venv/bin/activate
cd ml/01-customer-churn
pip install -r requirements.txt
python src/download_data.py          # fetches the real dataset into ./data
pytest -q                            # tests should pass
streamlit run app.py                 # open the chat UI in your browser
```

See [`00-getting-started/environment-setup.md`](00-getting-started/environment-setup.md) for GPU
notes (T4 / Apple M-series 16 GB), Ollama, vLLM, and the recommended 2026 local-model list.

## Conventions
- **Real, current datasets & models** — loaded via open registries/APIs (OpenML, UCI, Hugging Face,
  yfinance, Kaggle) on first run.
- **Never commit data, model weights, or secrets** — `.gitignore` enforces it.
- Each README has an honest **“Tested on”** line stating exactly what was run (CPU vs. GPU-required).

## Progress
This collection is built in reviewable batches. See [`PROGRESS.md`](PROGRESS.md) for what's shipped.
