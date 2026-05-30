"""Finance agent — analyse company financials (SQL) with a no-advice guardrail.

Combines structured analysis (querying a financials table) with a safety guardrail that
refuses to give investment advice. Offline it uses an in-memory SQLite table; production
adds RAG over filings + an LLM (documented).
"""

from __future__ import annotations

import sqlite3

_ROWS = [  # quarter, revenue, expenses (in $k)
    ("2025-Q1", 1200, 900), ("2025-Q2", 1350, 950),
    ("2025-Q3", 1500, 1000), ("2025-Q4", 1700, 1100),
]
_ADVICE = ["should i invest", "should i buy", "buy or sell", "is it a good investment", "financial advice"]


def build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE financials (quarter TEXT, revenue INT, expenses INT)")
    conn.executemany("INSERT INTO financials VALUES (?,?,?)", _ROWS)
    conn.commit()
    return conn


def guardrail(question: str) -> str | None:
    if any(a in question.lower() for a in _ADVICE):
        return ("I can analyse the numbers but can't give investment advice. "
                "Consult a licensed financial advisor.")
    return None


def answer(question: str) -> dict:
    blocked = guardrail(question)
    if blocked:
        return {"answer": blocked, "blocked": True}
    q = question.lower()
    conn = build_db()
    if "growth" in q:
        rows = conn.execute("SELECT revenue FROM financials ORDER BY quarter").fetchall()
        g = (rows[-1][0] - rows[0][0]) / rows[0][0] * 100
        ans = f"Revenue grew {g:.1f}% from {_ROWS[0][0]}k to {_ROWS[-1][0]}k across the year."
    elif "profit" in q:
        r = conn.execute("SELECT quarter, revenue-expenses FROM financials ORDER BY quarter DESC LIMIT 1").fetchone()
        ans = f"Latest profit ({r[0]}): ${r[1]}k."
    elif "total revenue" in q or "annual revenue" in q:
        tot = conn.execute("SELECT SUM(revenue) FROM financials").fetchone()[0]
        ans = f"Total annual revenue: ${tot}k."
    elif "margin" in q:
        r = conn.execute("SELECT SUM(revenue), SUM(expenses) FROM financials").fetchone()
        ans = f"Net margin: {(r[0]-r[1])/r[0]*100:.1f}%."
    else:
        ans = "Ask about revenue growth, profit, total revenue, or margin."
    conn.close()
    return {"answer": ans, "blocked": False}
