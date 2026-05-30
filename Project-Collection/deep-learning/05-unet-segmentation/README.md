# DL 05 · Image Segmentation with U-Net 🔴

**Problem.** Segmentation assigns a class to **every pixel** — outlining tumours in scans, roads in
satellite imagery, or defects on a production line. We build the classic **U-Net** encoder-decoder.

**What you build.** A compact U-Net (with skip connections), a pixel-wise cross-entropy training loop,
and a chat UI that trains and shows input-vs-predicted-mask.

## Dataset
A synthetic "find the bright disk" task (real masks generated in code) so it runs offline. Swap in a
real mask dataset — e.g., medical (ISIC skin lesions) or satellite — by replacing `synthetic_dataset`.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # CPU smoke test
streamlit run app.py      # "train" → "demo"
```

## What you learned
The U-Net architecture & skip connections · pixel-wise loss · how segmentation differs from
classification/detection. **Extensions:** Dice/IoU loss & metrics, real medical/satellite masks,
augmentation.

## Tested on
CPU (Python 3.11) — smoke test covers model output shape, a training step, and mask prediction on the
synthetic task. **Real, larger images train faster on a T4/M2 GPU.**

> **Why it matters.** Pixel-level segmentation (medical, geospatial, manufacturing QA) is
> specialised, in-demand CV work.
