"""Healthcare agent — guideline Q&A with strict safety guardrails.

Healthcare AI must be cautious: it should share general, sourced information but **refuse
to diagnose, prescribe, or dose**, and escalate emergencies. This agent retrieves from a
small wellness knowledge base and wraps every answer in guardrails + a disclaimer. The
README covers HIPAA-aware, clinically-validated production builds.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DISCLAIMER = ("⚠️ This is general information, not medical advice. "
              "Consult a qualified healthcare professional for your situation.")

KB = [
    "Staying hydrated and resting helps the body recover from mild colds.",
    "Regular physical activity supports cardiovascular and mental health.",
    "Frequent hand-washing reduces the spread of common infections.",
    "A balanced diet rich in vegetables and whole grains supports overall health.",
    "Good sleep hygiene improves mood, focus, and immune function.",
]

_EMERGENCY = ["chest pain", "can't breathe", "cant breathe", "suicidal", "overdose", "unconscious", "stroke"]
_BLOCK = ["diagnose", "diagnosis", "prescribe", "prescription", "dosage", "how much should i take", "mg of"]


def guardrail(question: str) -> str | None:
    """Return a safety response if the question must be refused/escalated, else None."""
    q = question.lower()
    if any(e in q for e in _EMERGENCY):
        return ("🚨 This may be an emergency. Please call your local emergency number "
                "or go to the nearest emergency department now.")
    if any(b in q for b in _BLOCK):
        return ("I can't provide a diagnosis, prescription, or dosage. Please consult a "
                "licensed clinician or pharmacist. " + DISCLAIMER)
    return None


def answer(question: str) -> dict:
    blocked = guardrail(question)
    if blocked:
        return {"answer": blocked, "blocked": True, "source": None}
    vec = TfidfVectorizer(stop_words="english").fit(KB + [question])
    sims = cosine_similarity(vec.transform([question]), vec.transform(KB))[0]
    i = int(sims.argmax())
    info = KB[i] if sims[i] > 0 else "I don't have information on that. Please ask a professional."
    return {"answer": f"{info}\n\n{DISCLAIMER}", "blocked": False, "source": info}
