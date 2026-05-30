"""Image-gen brain. Uses the shared chat UI — type a prompt, get an image."""

from __future__ import annotations

from . import imagegen as I


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    prompt = message.strip() or "a calm sunset over mountains"
    if prompt.lower() in {"demo", "help"}:
        prompt = "a calm sunset over mountains"
    img = I.generate(prompt, size=int(state.get("config", {}).get("size", 256)))
    return [
        Reply(f"Generated (offline procedural demo) for: _{prompt}_", "text"),
        Reply(img, "image", {"full_width": True}),
        Reply("For real photoreal output, run Stable Diffusion 3.5 / FLUX via `diffusers` (README).", "text"),
    ]
