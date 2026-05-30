"""Video-gen brain. Uses the shared chat UI — type a prompt, get an animated GIF."""

from __future__ import annotations

from . import videogen as V


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    prompt = message.strip() or "flowing colors"
    if prompt.lower() in {"demo", "help"}:
        prompt = "flowing colors"
    frames = V.generate_frames(prompt, n_frames=int(state.get("config", {}).get("frames", 16)))
    gif = V.to_gif(frames)
    return [
        Reply(f"Generated a {len(frames)}-frame clip (offline procedural demo) for: _{prompt}_", "text"),
        Reply(gif, "image", {"full_width": True}),
        Reply("For real text-to-video, run **Wan 2.2 / LTX-Video** (GPU) or a hosted API (README).", "text"),
    ]
