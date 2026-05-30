"""Reusable chat-style Streamlit UI shared by EVERY Project-Collection app.

Why this exists
---------------
All ~53 projects use one consistent chat interface instead of a bespoke UI per
project. Benefits:
  * Learners get a single mental model — every project "feels" the same.
  * We write the UI once (no per-project UI rebuilding / token waste).
  * ML, DL, NLP, GenAI and agent projects all map onto the same chat metaphor:
    the user sends a message (a question, some inputs, or a file) and the app
    replies in the transcript with text, a table, a chart, an image or code.

How a project uses it
---------------------
Each project writes a tiny ``app.py``::

    from _shared.chat_ui import run_chat_app, Reply
    from src.handler import respond          # the project's brain

    run_chat_app(
        title="Customer Churn Predictor",
        intro="Ask me to score a customer or type `demo`.",
        respond=respond,                     # Callable[[str, dict], Reply | list[Reply]]
        examples=["score a demo customer", "what drives churn?"],
        sidebar=my_sidebar_fn,               # optional Callable[[], dict]
    )

The ``respond`` callback receives ``(user_message, state)`` where ``state`` is a
mutable per-session dict (cache models there). It returns one ``Reply`` or a list.

This module is intentionally dependency-light: only ``streamlit`` is required.
``pandas`` / ``matplotlib`` / ``PIL`` are imported lazily when a reply uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import streamlit as st

# A reply "part" the handler can emit. ``kind`` selects how it is rendered.
#   text     -> markdown string
#   code     -> fenced code block (use ``meta={"language": "python"}``)
#   table    -> pandas DataFrame
#   figure   -> matplotlib Figure or plotly Figure
#   image    -> file path, URL, PIL.Image or bytes
#   metric   -> dict {label: value} rendered as st.metric row
#   error    -> red error markdown
@dataclass
class Reply:
    """A single chunk of assistant output."""

    content: Any
    kind: str = "text"
    meta: dict = field(default_factory=dict)


ReplyLike = Union[Reply, str, list]
RespondFn = Callable[[str, dict], ReplyLike]
SidebarFn = Callable[[], dict]


def _coerce(reply: ReplyLike) -> list[Reply]:
    """Normalise whatever the handler returned into a list[Reply]."""
    if reply is None:
        return [Reply("…", "text")]
    if isinstance(reply, Reply):
        return [reply]
    if isinstance(reply, str):
        return [Reply(reply, "text")]
    if isinstance(reply, list):
        out: list[Reply] = []
        for r in reply:
            out.extend(_coerce(r))
        return out
    # Unknown type -> show its repr so nothing is silently dropped.
    return [Reply(f"```\n{reply!r}\n```", "text")]


def _render(reply: Reply) -> None:
    """Render one Reply part into the current Streamlit container."""
    kind = reply.kind
    if kind == "text":
        st.markdown(str(reply.content))
    elif kind == "code":
        st.code(str(reply.content), language=reply.meta.get("language", "python"))
    elif kind == "table":
        st.dataframe(reply.content, use_container_width=True)
    elif kind == "figure":
        # matplotlib Figure has ``savefig``; plotly Figure does not.
        if hasattr(reply.content, "savefig"):
            st.pyplot(reply.content)
        else:
            st.plotly_chart(reply.content, use_container_width=True)
    elif kind == "image":
        st.image(reply.content, use_container_width=reply.meta.get("full_width", False))
    elif kind == "metric":
        cols = st.columns(len(reply.content))
        for col, (label, value) in zip(cols, reply.content.items()):
            col.metric(label, value)
    elif kind == "error":
        st.error(str(reply.content))
    else:
        st.markdown(str(reply.content))


def _render_all(parts: list[Reply]) -> None:
    for part in parts:
        _render(part)


def run_chat_app(
    *,
    title: str,
    respond: RespondFn,
    intro: str = "",
    examples: Optional[list[str]] = None,
    sidebar: Optional[SidebarFn] = None,
    page_icon: str = "🤖",
    accepts_files: bool = False,
) -> None:
    """Launch the shared chat UI. See module docstring for the contract.

    Parameters
    ----------
    title:    App title shown in the header and browser tab.
    respond:  ``(user_message, state) -> Reply | str | list`` — the project brain.
    intro:    Markdown shown once at the top of an empty conversation.
    examples: Optional clickable example prompts.
    sidebar:  Optional callable rendering sidebar controls; its return dict is
              merged into ``state["config"]`` so the handler can read settings.
    accepts_files: If True, show a file uploader; the path is passed to ``respond``
              via ``state["uploaded_file"]``.
    """
    st.set_page_config(page_title=title, page_icon=page_icon, layout="centered")
    st.title(f"{page_icon} {title}")

    # Per-session scratch space shared with the handler (cache models here).
    state: dict = st.session_state.setdefault("_state", {"config": {}})
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if sidebar is not None:
        with st.sidebar:
            state["config"].update(sidebar() or {})

    if accepts_files:
        with st.sidebar:
            up = st.file_uploader("Upload a file")
            state["uploaded_file"] = up

    if intro and not st.session_state.messages:
        st.info(intro)

    # Replay history. Assistant turns store a list[Reply]; user turns store text.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_all(msg["parts"])
            else:
                st.markdown(msg["content"])

    # Example prompt buttons (only on a fresh conversation).
    prompt: Optional[str] = None
    if examples and not st.session_state.messages:
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                prompt = ex

    typed = st.chat_input("Type a message…")
    prompt = typed or prompt
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                parts = _coerce(respond(prompt, state))
        except Exception as exc:  # surface errors in-chat instead of crashing
            parts = [Reply(f"**Error:** {exc}", "error")]
        _render_all(parts)

    st.session_state.messages.append({"role": "assistant", "parts": parts})
