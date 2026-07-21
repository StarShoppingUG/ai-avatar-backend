"""
ai.py — LLM connectivity for the avatar backend.

This module owns exactly two things:
  1. The Groq client (OpenAI-compatible) and a single `call_llm()` entry
     point used by both backend.py (character replies) and translation.py
     (EN<->JA translation).
  2. A date/time context builder so the LLM always knows "today".

Everything else the character actually says or does (personas, TTS/visemes,
voice catalog, response validation) lives in backend.py — this file used to
duplicate all of that too (its own think(), smalltalk classifier, TTS
pipeline, voice catalog), but none of it was ever called, so it's gone.
"""
import os
import re
import asyncio
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ==========================================
# LLM CONFIG & CLIENT (Groq, via the OpenAI-compatible API)
# ==========================================
def _normalize_groq_base_url(raw: str) -> str:
    """
    The OpenAI SDK just appends '/chat/completions' to base_url, so base_url
    MUST be Groq's *API* host ('api.groq.com', not the marketing site
    'groq.com') and end in '/openai/v1', or every request 405s. This
    tolerates whatever people put in .env — bare host, the marketing domain,
    trailing slash, already-correct value, etc — and always resolves to the
    one URL shape that actually works.
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return "https://api.groq.com/openai/v1"
    # Common mix-up: the marketing site (groq.com) instead of the API host
    # (api.groq.com) — same domain minus the 'api.' subdomain.
    url = re.sub(r"://(?:www\.)?groq\.com", "://api.groq.com", url)
    if url.endswith("/openai/v1") or url.endswith("/chat/completions"):
        return url
    return f"{url}/openai/v1"

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = _normalize_groq_base_url(os.environ.get("GROQ_BASE_URL", "https://api.groq.com"))

# Groq's Whisper endpoint — same account/key as the chat model above, just a
# different model name against the OpenAI-compatible /audio/transcriptions
# route. "turbo" is faster and cheap; swap to "whisper-large-v3" via env if
# you want the highest-accuracy (slightly slower) variant instead.
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

try:
    llm_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if GROQ_API_KEY else None
except Exception as e:
    print(f"⚠️ LLM client init failed: {e}")
    llm_client = None


def ai_available() -> bool:
    return llm_client is not None


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm", language: str | None = None) -> str:
    """
    Speech-to-text via Groq's hosted Whisper (same client/key as call_llm).
    `filename` just needs a plausible extension — Groq uses it to sniff the
    container format (webm/mp4/wav/etc), it isn't saved anywhere.
    `language` is an optional ISO-639-1 hint ("en"/"ja"); Whisper auto-detects
    when omitted, but passing it in when you already know it (e.g. from the
    UI's response-language toggle) measurably improves accuracy and latency.
    """
    if not ai_available():
        raise RuntimeError("No GROQ_API_KEY set in .env file")

    loop = asyncio.get_event_loop()

    def _call():
        kwargs = {
            "model": GROQ_STT_MODEL,
            "file": (filename, audio_bytes),
        }
        if language:
            kwargs["language"] = language
        return llm_client.audio.transcriptions.create(**kwargs)

    result = await loop.run_in_executor(None, _call)
    text = getattr(result, "text", None)
    if text is None:
        text = str(result)
    return text.strip()


async def call_llm(messages: list, json_mode: bool = False) -> str:
    """Call the LLM (Groq) and return the raw text response, fences stripped."""
    if not ai_available():
        raise RuntimeError("No GROQ_API_KEY set in .env file")

    loop = asyncio.get_event_loop()
    kwargs = {"model": GROQ_MODEL, "messages": messages, "max_tokens": 1024, "temperature": 0.7}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await loop.run_in_executor(
        None,
        lambda: llm_client.chat.completions.create(**kwargs),
    )
    return (
        response.choices[0].message.content
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


# ==========================================
# DATE / TIME CONTEXT
# ==========================================
def _get_current_date_context(user_timezone: str = None) -> str:
    """
    Build a short block that grounds the LLM in the current real-world date,
    optionally localized to the user's timezone.
    """
    if user_timezone:
        try:
            import pytz
            tz = pytz.timezone(user_timezone)
            now = datetime.now(tz)
            tz_name = user_timezone
        except Exception:
            now = datetime.now()
            tz_name = "Server local time"
    else:
        now = datetime.now()
        tz_name = "Server local time"

    yesterday = now - timedelta(days=1)
    tomorrow = now + timedelta(days=1)

    return (
        "═══════════════════════════════════════════════════════\n"
        "TEMPORAL CONTEXT\n"
        "═══════════════════════════════════════════════════════\n"
        f"Today is {now.strftime('%A, %B %d, %Y')}, current time {now.strftime('%I:%M %p')} ({tz_name}).\n"
        f"Yesterday was {yesterday.strftime('%A, %B %d, %Y')}; tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}.\n"
        "Your training data has a cutoff, but you always know the real current date shown above — "
        "never say 'as of 2023/2024' or claim not to know today's date.\n"
        "═══════════════════════════════════════════════════════\n\n"
    )