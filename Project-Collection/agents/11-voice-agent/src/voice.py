"""Voice agent — speech-to-text → LLM → text-to-speech.

A voice assistant pipeline: transcribe audio (**faster-whisper**), reason (LLM), speak the
reply (**Piper/Coqui TTS**) — all runnable locally on 16 GB. The STT/TTS edges need audio
models (documented); the **text orchestration** in the middle is tested offline using the
shared LLM client (mock when no model is available).
"""

from __future__ import annotations


def transcribe(audio_path: str) -> str:
    """Speech → text with faster-whisper (local, GPU/CPU). Documented path."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join(s.text for s in segments).strip()


def respond_text(transcript: str) -> str:
    """The 'brain' — generate a spoken-style reply (offline-safe via shared LLM client)."""
    from _shared.llm import complete

    res = complete(f"You are a friendly voice assistant. Reply in one short sentence.\nUser: {transcript}")
    return res["text"]


def speak(text: str, out_path: str = "reply.wav") -> str:
    """Text → speech with Piper/Coqui TTS (local). Documented path."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path


def pipeline(transcript: str) -> dict:
    """Run the middle of the pipeline (text in → text out) for an offline demo/test."""
    reply = respond_text(transcript)
    return {"transcript": transcript, "reply": reply}
