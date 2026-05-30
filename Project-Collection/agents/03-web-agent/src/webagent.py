"""Web / browser agent — navigate pages and extract information.

Browser agents automate real web tasks (open a page, click, read, extract). This
category exploded in 2025-26. To run offline & deterministically, we drive a small
**mock website** (a dict of pages with links); the README shows the real stack —
**browser-use** / **Playwright** — for live browsers.
"""

from __future__ import annotations

import re

MOCK_SITE = {
    "/": {"text": "Welcome to Acme. Visit our product and about pages.",
          "links": {"product": "/product", "about": "/about"}},
    "/product": {"text": "SuperWidget — price: $49.99. In stock. Free shipping over $50.", "links": {}},
    "/about": {"text": "Acme was founded in 2020 in London by Ada Lovelace.", "links": {}},
}


def fetch(url: str) -> dict:
    return MOCK_SITE.get(url, {"text": "404 not found", "links": {}})


def navigate_to(keyword: str, start: str = "/") -> tuple[str, str]:
    """From the start page, follow a link whose name matches the keyword."""
    page = fetch(start)
    for name, href in page.get("links", {}).items():
        if keyword in name:
            return href, fetch(href)["text"]
    return start, page["text"]


def run(task: str) -> dict:
    """Tiny browser-agent policy over the mock site."""
    t = task.lower()
    trace = ["open /"]
    if "price" in t or "cost" in t or "product" in t:
        url, text = navigate_to("product"); trace.append(f"click product → {url}")
        price = re.search(r"\$\d+(?:\.\d+)?", text)
        return {"trace": trace, "answer": price.group() if price else text, "page": text}
    if "found" in t or "about" in t or "who" in t or "where" in t:
        url, text = navigate_to("about"); trace.append(f"click about → {url}")
        return {"trace": trace, "answer": text, "page": text}
    return {"trace": trace, "answer": fetch("/")["text"], "page": fetch("/")["text"]}
