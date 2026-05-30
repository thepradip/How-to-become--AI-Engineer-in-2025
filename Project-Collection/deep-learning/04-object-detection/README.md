# DL 04 · Object Detection with YOLO11 🟡

**Problem.** Detect and locate objects (people, cars, products…) in images — the basis of security
analytics, retail shelf monitoring, autonomous systems, and more.

**What you build.** A YOLO11 detector (Ultralytics) wrapped in the shared chat UI: **upload an image**
and get back an annotated image + a table of detected objects and confidences.

## Model & data (real)
**YOLO11-n**, COCO-pretrained (80 classes). Weights download on first real use. Train on your own data
later with a YOLO dataset YAML (`model.train(data='your.yaml', epochs=...)`).

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline: builds the architecture from YAML + runs the forward path
streamlit run app.py      # upload an image, or type "demo" (downloads weights + sample image)
```

## What you learned
Detection vs. classification (boxes + labels) · using a SOTA detector via a clean API · the
pretrained → fine-tune path for custom objects.

## Tested on
CPU (Python 3.11) — smoke test builds the YOLO11 architecture from the bundled YAML (offline, random
weights) and runs the forward/annotate path on a blank image. **Real detection downloads COCO weights;
custom training wants a T4/M2 GPU.** Tests `importorskip` torch & ultralytics.

> **Freelance relevance.** "Count/track objects in our camera feed or images" is a very common,
> deployable CV contract.
