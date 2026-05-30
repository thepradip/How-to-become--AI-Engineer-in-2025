"""Voice-agent brain. Uses the shared chat UI — type a 'transcript', get the reply."""

from __future__ import annotations

from . import voice as V


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    transcript = message.strip() or "what's the weather like?"
    res = V.pipeline(transcript)
    return [
        Reply(f"🎙️ **Heard (STT):** {res['transcript']}"),
        Reply(f"🔊 **Reply (→ TTS):** {res['reply']}"),
        Reply("_Text orchestration shown. Wire faster-whisper (STT) + Piper (TTS) for full voice — README._", "text"),
    ]
