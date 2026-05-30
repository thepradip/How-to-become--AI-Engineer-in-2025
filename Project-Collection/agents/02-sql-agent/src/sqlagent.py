"""SQL agent — answer natural-language questions over a real database.

A SQL agent turns "how many employees are in Engineering?" into SQL, runs it, and
returns the result. We use a real **SQLite** DB; the NL→SQL step is rule-based here so
it runs offline and deterministically. Production uses **LangGraph** + an LLM that
receives the schema and writes the SQL (documented in the README).
"""

from __future__ import annotations

import sqlite3

SCHEMA = "employees(id, name, department, salary)"
_ROWS = [
    (1, "Ada", "Engineering", 145000), (2, "Bob", "Engineering", 120000),
    (3, "Carol", "Sales", 95000), (4, "Dan", "Sales", 88000),
    (5, "Eve", "Marketing", 99000), (6, "Frank", "Engineering", 160000),
]


def build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary INTEGER)")
    conn.executemany("INSERT INTO employees VALUES (?,?,?,?)", _ROWS)
    conn.commit()
    return conn


def nl_to_sql(question: str) -> str:
    """Rule-based NL→SQL for the employees table (offline demo)."""
    q = question.lower()
    dept = next((d for d in ("engineering", "sales", "marketing") if d in q), None)
    where = f" WHERE department='{dept.capitalize()}'" if dept else ""
    if "how many" in q or "count" in q:
        return f"SELECT COUNT(*) AS n FROM employees{where}"
    if "average" in q or "avg" in q or "mean" in q:
        return f"SELECT AVG(salary) AS avg_salary FROM employees{where}"
    if ("total" in q or "sum" in q) and "department" in q:
        return "SELECT department, SUM(salary) AS total FROM employees GROUP BY department"
    if "highest" in q or "top" in q or "most" in q:
        return f"SELECT name, salary FROM employees{where} ORDER BY salary DESC LIMIT 1"
    if "list" in q or "show" in q or "who" in q:
        return f"SELECT name, department, salary FROM employees{where}"
    return f"SELECT * FROM employees{where} LIMIT 5"


def answer(question: str) -> dict:
    conn = build_db()
    sql = nl_to_sql(question)
    cur = conn.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return {"sql": sql, "rows": rows}
