# DL 09 · Face Verification (embeddings) 🔴

**Problem.** Decide whether two face images are the **same person** — the basis of face login, photo
de-duplication, and access control. Unlike classification, the model must generalise to identities it
never saw in training, so we learn **embeddings** and compare them.

**What you build.** A CNN that maps a face to an L2-normalised embedding (trained as an identity
classifier), then verifies pairs by **cosine similarity** vs. a threshold — in the shared chat UI.

## Dataset
Synthetic "identities" (distinct base images + noise) so it runs offline. For real faces, use
**facenet-pytorch** (pretrained on VGGFace2) + the **LFW** benchmark.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "train" → "verify"
```

## What you learned
Metric/embedding learning vs. classification · L2-normalised embeddings + cosine similarity ·
verification thresholds & the FAR/FRR trade-off. **Extensions:** triplet/contrastive loss, real LFW
evaluation (ROC, EER).

## Tested on
CPU (Python 3.11) — smoke test checks unit-norm embeddings, the verify API, and that **same-identity
pairs score higher than different-identity pairs after training**. No GPU required for the demo.

> **Why it matters.** Face/image matching and de-duplication appear in identity, security and
> media-asset projects.
