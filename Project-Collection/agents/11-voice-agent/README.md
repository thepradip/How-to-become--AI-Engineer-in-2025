# Agents 11 · Voice Agent (STT → LLM → TTS) 🔴

**Problem.** Build a talking assistant: hear speech, reason, speak back. Runnable locally on 16 GB with
**faster-whisper** (STT) + an LLM + **Piper/Coqui** (TTS), or via the OpenAI Realtime API.

**What you build.** The voice pipeline. The **text orchestration** (transcript → reply) is tested
offline via the shared LLM client; the STT/TTS edges use audio models (documented).

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # text orchestration (offline)
streamlit run app.py      # type a 'transcript', get a reply
```
Full voice loop:
```python
# pip install faster-whisper pyttsx3
from src.voice import transcribe, respond_text, speak
speak(respond_text(transcribe("clip.wav")))     # STT -> LLM -> TTS
```

## What you learned
The STT→LLM→TTS pipeline · local speech models on 16 GB · latency considerations (why Realtime APIs
exist). **Extensions:** streaming, barge-in, wake-word, OpenAI Realtime.

## Tested on
CPU (Python 3.11), offline (text middle of the pipeline via mock LLM). STT/TTS need audio models.

> **Why it matters.** Voice agents (support, IVR, assistants) are a fast-growing, in-demand build.
