# Agents 03 · Web / Browser Agent 🟡

**Problem.** Automate web tasks — open pages, click, read, extract — that have no API. Browser agents
were one of the fastest-growing agent categories in 2025-26.

**What you build.** An agent that navigates a site and extracts answers, with a visible navigation
trace, in the shared chat UI. Offline it drives a **mock site**; the real path uses **browser-use** /
**Playwright** to control a live browser.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # ask for the product price / founding year
```
Real browsing:
```bash
pip install browser-use playwright && playwright install chromium
# browser-use lets an LLM drive a real Chromium to complete web tasks
```

## What you learned
Navigation + extraction loops · why browser agents matter (no-API sites) · the mock-vs-live trade-off.
**Extensions:** real browser-use runs, form filling, pagination, auth-aware scraping.

## Tested on
CPU (Python 3.11), fully offline (mock site). Live browsing needs Playwright/browser-use.

> **Freelance relevance.** "Automate this web workflow / extract data from this portal" — steady,
> well-paid automation work.
