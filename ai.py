
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

    url = (raw or "").strip().rstrip("/")
    if not url:
        return "https://api.groq.com/openai/v1"

    url = re.sub(r"://(?:www\.)?groq\.com", "://api.groq.com", url)
    if url.endswith("/openai/v1") or url.endswith("/chat/completions"):
        return url
    return f"{url}/openai/v1"

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "").strip()

GROQ_MODEL    = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL = _normalize_groq_base_url(os.environ.get("GROQ_BASE_URL", "https://api.groq.com"))

# Used only for turns that need live/current info (see _NEEDS_LIVE_INFO_PAT
# in app.py's think()) — compound-mini can call a web search tool itself,
# unlike GROQ_MODEL above. Kept as a separate override (not just swapping
# GROQ_MODEL globally) because compound systems have tighter rate limits
# and higher latency than a bare text model, so they're only worth paying
# for on the turns that actually need current data.
GROQ_COMPOUND_MODEL = os.environ.get("GROQ_COMPOUND_MODEL", "groq/compound-mini")


GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")

try:
    llm_client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL) if GROQ_API_KEY else None
except Exception as e:
    print(f"⚠️ LLM client init failed: {e}")
    llm_client = None


def ai_available() -> bool:
    return llm_client is not None


def translation_ai_available() -> bool:
    """Kept as a separate name (rather than having translation.py call
    ai_available() directly) so nothing else needs to change if translation
    ever needs its own availability check again later — right now it's
    identical to ai_available() since Groq is the only provider."""
    return ai_available()


async def call_translation_llm(messages: list, json_mode: bool = False, temperature: float = 0.2, max_tokens: int = 2048) -> str:

    return await call_llm(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm", language: str | None = None) -> str:

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


async def call_llm(messages: list, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 1024, reasoning_effort: str | None = None, model: str | None = None) -> str:

    if not ai_available():
        raise RuntimeError("No GROQ_API_KEY set in .env file")

    active_model = model or GROQ_MODEL
    loop = asyncio.get_event_loop()
    kwargs = {"model": active_model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort and (active_model.startswith(("openai/gpt-oss", "qwen/qwen3")) or "compound" in active_model):
        kwargs["reasoning_effort"] = reasoning_effort

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