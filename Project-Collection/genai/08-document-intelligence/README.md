# GenAI 08 · Document Intelligence / OCR 📄 🟡

**Problem.** Turn messy documents (scanned invoices, contracts, forms, PDFs) into structured data.
The backbone of finance/ops/legal automation — and a constant freelance request.

**What you build.** The pipeline scan/PDF → **OCR** → **field extraction**. The extraction step
(regex/LLM over OCR'd text) runs and is tested offline; the OCR step uses a modern engine
(**Docling / olmOCR / Mistral OCR / Qwen2.5-VL**), documented.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (field extraction on sample invoice text)
streamlit run app.py      # paste OCR'd text, or "demo"
```
OCR a real scan/PDF:
```python
# pip install docling
from src.docai import ocr_image, extract_fields
extract_fields(ocr_image("invoice.pdf"))     # Docling → text → fields
```

## What you learned
The OCR → extraction pipeline · VLM-based OCR (2026 SOTA) vs. legacy engines · regex vs. LLM field
extraction. **Upgrade:** an LLM with a structured schema (see GenAI 09) for robust extraction.

## Tested on
CPU (Python 3.11), offline (field extraction). OCR of scans/PDFs needs an OCR engine (`docling`, etc.).

> **Freelance relevance.** "Extract data from our invoices/forms/PDFs" is one of the most common,
> immediately-valuable automation gigs.
