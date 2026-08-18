import ast
import os
import random
import re
import json
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form, UploadFile, File, Query, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select
import edge_tts

try:
    from .translation import translate_to_japanese, translate_to_english, is_japanese
    from .ai import ai_available, call_llm, GROQ_MODEL, _get_current_date_context, transcribe_audio
    from .json_utils import normalize_json_like, extract_quoted_value
    from .db import init_db, get_session, ChatMessage, UserSettings, AppSettings
except ImportError:
    from translation import translate_to_japanese, translate_to_english, is_japanese
    from ai import ai_available, call_llm, GROQ_MODEL, _get_current_date_context, transcribe_audio
    from json_utils import normalize_json_like, extract_quoted_value
    from db import init_db, get_session, ChatMessage, UserSettings, AppSettings

# (/voice, /stt, /translate, /voices, /health) don't need it.
def get_user_id(
    x_user_id: Optional[str] = Header(default=None),
    x_app_id: Optional[str] = Header(default=None),
) -> str:
    """Returns a scoped identity string combining tenant (app) and end-user,
    e.g. "acme-corp::user-48213". This is what actually gets stored as
    user_id everywhere (ChatMessage, UserSettings) — every existing query
    that filters on user_id therefore already gets full per-app isolation
    for free, with no schema changes and no query changes required.

    x_app_id is optional for backward compatibility (older/solo frontends
    that don't send it yet) and falls back to a shared "default" tenant —
    matches CharacterBrain.js's own fallback behavior on the frontend.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    app_id = (x_app_id or "default").strip() or "default"
    return f"{app_id}::{x_user_id.strip()}"

def get_user_id_optional(
    x_user_id: Optional[str] = Header(default=None),
    x_app_id: Optional[str] = Header(default=None),
) -> Optional[str]:
    """Same identity-string construction as get_user_id(), but never raises.
    For routes like /settings where whether user_id is actually needed
    depends on scope (X-Settings-Scope) — scope=app never touches user_id
    at all, so requiring X-User-Id at the dependency level 400s app-scoped
    requests before the route body even gets to check scope. Callers that
    need user_id (scope=user) are responsible for raising themselves if
    this comes back None."""
    if not x_user_id or not x_user_id.strip():
        return None
    app_id = (x_app_id or "default").strip() or "default"
    return f"{app_id}::{x_user_id.strip()}"

def get_app_id(x_app_id: Optional[str] = Header(default=None)) -> str:
    """App-level identity only, no user component — used for AppSettings."""
    return (x_app_id or "default").strip() or "default"

def get_settings_group(x_settings_group: Optional[str] = Header(default=None)) -> str:
    """Second scoping dimension alongside app_id, for scope=app usage only
    (see get_settings_scope() below). Lets one app_id share settings per
    some sub-grouping (e.g. a "character" = scenario + avatar combo)
    instead of forcing every scope=app user of that app_id onto a single
    shared row. Defaults to "" when absent — matches AppSettings.settings_group's
    default, so existing scope=app usage with no group set keeps landing
    on the same row it always has. No trimming beyond that: unlike
    x_app_id/x_user_id this is allowed to be an arbitrary opaque string an
    integrator defines for their own use, not an identity that needs
    normalizing."""
    return x_settings_group or ""

def get_settings_scope(x_settings_scope: Optional[str] = Header(default=None)) -> str:
    """'app' = every user of this app-id shares one AppSettings row (opt-in).
    'user' (default when header is absent) = today's per-browser/UUID
    UserSettings isolation — unchanged for any client that doesn't send
    this header, e.g. the existing Vercel demo."""
    return "app" if (x_settings_scope or "").strip().lower() == "app" else "user"

DEV_LOGGING = os.environ.get("DEV_LOGGING", "false").strip().lower() in ("1", "true", "yes")


# 3. VOICE CATALOG
EN_VOICE = "en-US-JennyNeural"
JA_VOICE = "ja-JP-NanamiNeural"

def build_character_system(user_text: str, character_name: str = None, avatar_persona: str = None) -> str:
    active_character = character_name or "the selected avatar"
    avatar_specialty = avatar_persona or "a helpful, all-round assistant"
    return (
        f"You ARE {active_character}. This is not a role you are describing from the outside — "
        f"you think, speak, and answer in first person AS {active_character} for this whole conversation. "
        "Never break character, and never say you are an AI, a language model, or an assistant.\n"
        f"YOUR IDENTITY (this describes YOU, the avatar — NOT the user): {avatar_specialty}\n"
        "- If asked who you are or to introduce yourself, answer briefly using your identity above — "
        "your name alone is not an introduction.\n"
        "- Let your identity shape your tone, the examples you reach for, and the advice you give, "
        "even when the topic is general.\n"
        "- If a question falls clearly outside your identity, you may still help if you reasonably can, "
        "but say briefly that it's outside what you specialize in and, where it makes sense, steer the "
        "conversation back toward your area. This is a soft steer, not a refusal — stay warm and helpful.\n"
        "- CRITICAL: never attribute your own name, background, persona, or traits to the user. If asked "
        "what you know about the user (their name, background, preferences, etc.), answer only from what "
        "the user themselves has actually said in the conversation history — never from your own identity "
        "above. If the user hasn't told you anything about themselves yet, say so honestly instead of "
        "describing yourself back to them.\n\n"
        "Respond like a helpful, natural conversation partner. Be relaxed, clear, and human.\n"
    )

VOICE_MAP = {
    "en":           EN_VOICE,
    "en-US":        EN_VOICE,
    "en-US-Jenny":  "en-US-JennyNeural",
    "ja":           JA_VOICE,
    "ja-JP":        JA_VOICE,
    "ja-JP-Nanami": "ja-JP-NanamiNeural",
}

def resolve_voice(name: str, use_japanese: bool = False) -> str:
    if name and name in VOICE_MAP:
        return VOICE_MAP[name]
    if name and name.endswith("Neural"):
        return name
    return JA_VOICE if use_japanese else EN_VOICE

# 4. APP SETUP

app = FastAPI(title="AI Avatar Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-avatar-ui-ghost.vercel.app", "https://ai-dojo-prototype-ghost.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def _on_startup():
    init_db()
    for mw in app.user_middleware:
        if mw.cls.__name__ == "CORSMiddleware":
            print(f"🔍 CORS raw options: {mw.kwargs}", flush=True)

# ==========================================
# 5. BEHAVIOR JSON PARSING
# ==========================================
def _extract_behavior_json(raw: str) -> dict:
    """Best-effort parse of the LLM's {reply, expression, animation} JSON, tolerating
    the usual mistakes (single quotes, trailing commas, stray prose around the object)."""
    normalized = normalize_json_like(raw)
    try:
        data = json.loads(normalized)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except Exception:
        try:
            data = ast.literal_eval(normalized)
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0]
        except Exception:
            pass

    # Last resort: pull each field out with a tolerant quoted-value scan.
    fallback = {"reply": "", "expression": "neutral", "animation": "talk"}
    for key in ("reply", "expression", "animation"):
        extracted = extract_quoted_value(raw, key)
        if extracted is not None:
            fallback[key] = extracted.strip()

    if not fallback["reply"] and raw.strip():
        cleaned = raw.strip()
        key_match = re.search(r"\b(?:expression|animation|reply)\b\s*:", cleaned, flags=re.IGNORECASE)
        if key_match:
            cleaned = cleaned[:key_match.start()].rstrip(' \"\'.,;:')
        fallback["reply"] = cleaned or fallback["reply"]

    return fallback

# ── Offline / error fallback: lightweight keyword sentiment ────────────────
# This only fires when the LLM is unreachable or unconfigured — it is not a
# shortcut used on the normal request path, which always goes to the LLM.
def sentiment_behavior(text: str) -> dict:
    low = text.lower()
    expression, animation = "neutral", "talk"

    if any(w in low for w in ("thanks", "thank you", "appreciate")):
        expression, animation = "happy", "thankful"
    elif any(w in low for w in ("good", "happy", "yes", "perfect", "awesome")):
        expression, animation = "happy", "nod"
    elif any(w in low for w in ("bad", "sad", "sorry", "wrong", "fail", "error", "unfortunate")):
        expression, animation = "sad", "talk"
    elif "?" in text or any(w in low for w in ("why", "how", "what", "explain", "think")):
        expression, animation = "thinking", "explain"
    elif any(w in low for w in ("wow", "amazing", "incredible", "great", "fantastic", "scared", "fear")):
        expression, animation = "surprised", "nod"

    if any(w in low for w in ("hello", "hi ", "welcome", "konnichiwa", "こんにちは")):
        expression, animation = "happy", "greeting"

    return {"expression": expression, "animation": animation}

# A handful of natural-sounding lines instead of one exact string repeated on
# every failure — repeating the identical sentence verbatim reads as robotic
# and makes it obvious something's broken rather than just a hiccup.
_FALLBACK_REPLIES = [
    "Looks like I've lost my connection to the internet — I can't think right now. Please try again in a moment.",
    "Sorry, I seem to be offline — my connection dropped. Give it a moment and try again.",
    "I can't reach the internet right now, so I can't process that. Please try again shortly.",
    "It looks like my connection is down at the moment — try again in a bit once it's back.",
]
_last_fallback_reply = None

def pick_fallback_reply() -> str:
    """Picks a fallback line, avoiding an immediate repeat of the last one."""
    global _last_fallback_reply
    choices = [r for r in _FALLBACK_REPLIES if r != _last_fallback_reply] or _FALLBACK_REPLIES
    choice = random.choice(choices)
    _last_fallback_reply = choice
    return choice

def pick_rate_limit_reply(error_text: str) -> str:
    wait_match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
    if wait_match:
        minutes = int(wait_match.group(1) or 0)
        wait = f"about {minutes} minute{'s' if minutes != 1 else ''}" if minutes >= 1 else "a few seconds"
    else:
        wait = "a little while"
    return f"I've hit Groq's rate limit and need to wait {wait} before I can reply again."

# ── Viseme constants ──────────────────────────────────────────────────────
VOWEL_VISEMES = {"a": "aa", "e": "ee", "i": "ih", "o": "oh", "u": "ou"}
CONSONANT_VISEMES = {
    "m": "ou", "p": "ou", "b": "ou", "w": "ou",
    "f": "ih", "v": "ih", "s": "ih", "z": "ih", "c": "ih",
    "h": "aa", "k": "aa", "g": "aa",
    "r": "oh", "l": "oh",
    "t": "ee", "d": "ee", "n": "ee",
}
VISEME_LEAD_MS           = 70
DEFAULT_WORD_DURATION_MS = 220
MAX_VISEMES_PER_WORD     = 3

def normalize_word(word: str) -> str:
    return word.lower().strip(".,!?;:\"'()[]{}")


def word_to_viseme_sequence(word: str) -> list:
    if not word:
        return ["sil"]
    w = normalize_word(word)
    sequence = []
    for ch in w:
        if ch in VOWEL_VISEMES:
            vis = VOWEL_VISEMES[ch]
            if not sequence or sequence[-1] != vis:
                sequence.append(vis)
    if not sequence and w:
        first = w[0]
        if first in CONSONANT_VISEMES:
            sequence.append(CONSONANT_VISEMES[first])
    if not sequence:
        sequence.append("aa")
    return sequence[:MAX_VISEMES_PER_WORD]

async def generate_tts_with_visemes(text: str, voice: str, output_path: str, rate: str = "+0%") -> list:
    events = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            t_ms = chunk["offset"] // 10_000
            events.append({"t": t_ms, "text": chunk.get("text", "")})

    timeline = []
    for index, event in enumerate(events):
        start    = max(0, event["t"] - VISEME_LEAD_MS)
        end      = events[index + 1]["t"] if index + 1 < len(events) else event["t"] + DEFAULT_WORD_DURATION_MS
        duration = max(80, end - event["t"])
        visemes  = word_to_viseme_sequence(event["text"])
        if len(visemes) == 1:
            timeline.append({"t": start, "v": visemes[0]})
        else:
            step = duration // len(visemes)
            for idx, vis in enumerate(visemes):
                timeline.append({"t": start + idx * step, "v": vis})

    with open(output_path, "wb") as f:
        for c in chunks:
            f.write(c)

    if timeline:
        timeline.append({"t": timeline[-1]["t"] + 300, "v": "sil"})
    return timeline

async def safe_tts(text: str, voice: str, path: str, rate: str = "+0%") -> tuple:
    """Returns (visemes, generated). generated=False means no audio file was
    written for this call — either there was nothing worth saying, or the TTS
    request itself failed. Callers must key off `generated`, not
    os.path.exists(path): audio filenames are only unique within a server
    run (see next_audio_name), so a stale file from a previous run can
    already be sitting at that exact path and make a failed call look like
    it succeeded."""
    clean = (text or "").strip()
    if not clean or clean in ("...", "…", "."):
        return [], False
    try:
        visemes = await generate_tts_with_visemes(clean, voice, path, rate=rate)
        # Guard against a "successful" call that didn't actually produce a
        # readable file (e.g. edge-tts returned zero audio chunks) — better
        # to fall back to no-audio than hand back a URL that 404s.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"⚠️ TTS produced no audio file at {path}")
            return [], False
        return visemes, True
    except Exception as e:
        print(f"⚠️ TTS failed ({voice}): {e}")
        return [], False

# ── Audio naming ──────────────────────────────────────────────────────────
# UUID-based, not a shared counter — a counter is process-local (resets on
# every --reload restart) and isn't safe under concurrent /ask calls, since
# nothing prevents two overlapping requests from racing between "claim a
# name" and "finish writing the file for that name." A UUID makes every
# filename unique by construction, so there's nothing to race.
def next_audio_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}.mp3"

# ==========================================
# 6. THINK — always goes to the LLM
# ==========================================
VALID_EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "thinking", "relaxed", "angry", "scared"]
VALID_ANIMATIONS  = ["idle", "talk", "explain", "nod", "thankful", "greeting", "offline"]

_DIRECT_PAT = re.compile(
    r"\b(what is|what are|who is|how do|how does|how many|when|where|can you|"
    r"tell me|explain|define|give me|translate|what's|who's)\b", re.I
)

def _wrap_prompt(system_prompt: str, intent: str, character_name: str, teaching_mode: bool = False) -> str:
    date_context = _get_current_date_context()

    if teaching_mode:
        # Overrides the direct/general split below — a "how do I say X"
        # question matches _DIRECT_PAT and would otherwise get capped at
        # MAX 60 WORDS / "be concise", which is exactly why teaching
        # replies were coming back as a bare phrase with no instruction.
        mode = (
            "RESPONSE MODE — TEACHING ANSWER: Never just state a translation or phrase on its "
            "own — actually teach it. Every answer should include: (1) the word/phrase itself, "
            "properly formatted per the bracket rule below, (2) a short, natural note on how or "
            "when it's actually used (politeness level, context, common situation), and (3) one "
            "short example sentence using it naturally. Then invite the learner to try repeating "
            "it back, or ask a quick follow-up question to keep the lesson moving — don't just "
            "end on the fact. Keep the tone warm and conversational, like a real tutor talking, "
            "not a bullet-point dictionary entry. 60–100 WORDS.\n\n"
        )
    elif intent == "direct":
        mode = ("RESPONSE MODE — DIRECT ANSWER: Answer clearly and helpfully. "
                "Keep it concise and practical. MAX 60 WORDS — be concise.\n\n")
    else:
        mode = ("RESPONSE MODE — GENERAL GUIDANCE: Share practical insights with examples. "
                "Be warm, clear, and useful. 80–120 WORDS.\n\n")

    return date_context + mode + system_prompt

# Only appended when AskRequest.teaching_mode is true. Deliberately does
# NOT name which language is being taught or who the student is — that
# direction lives entirely in the avatar's own persona (see
# AvatarSources.js, e.g. "You teach Japanese to English-speaking
# learners"). This guide only adds the MECHANICAL formatting/tone rules
# that apply either direction, so the same guide works for a
# Japanese-teaching avatar and an English-teaching avatar without change.
_TEACHING_MODE_GUIDE = (
    "═══════════════════════════════════════════════════════\n"
    "TEACHING MODE\n"
    "═══════════════════════════════════════════════════════\n"
    "You are actively teaching a language, per your identity above. Your "
    "identity states which language you teach and which language your "
    "student speaks natively — follow that direction precisely, even "
    "though the rest of this system prompt and the conversation history "
    "are written in English. Do NOT default to writing mostly in English "
    "just because this instruction is in English.\n"
    "The EXPLANATORY portions of your reply (context, grammar notes, "
    "encouragement, small talk) should be written in your STUDENT's native "
    "language, not automatically in English. Concretely:\n"
    "- If you teach Japanese to an English-speaking student: write mostly "
    "in English, with Japanese words/phrases embedded in Japanese script.\n"
    "- If you teach English to a Japanese-speaking student: write mostly "
    "in Japanese, with English words/phrases embedded as actual English "
    "text.\n"
    "THIS SPLIT IS FIXED BY YOUR ROLE AND NEVER CHANGES BASED ON WHAT "
    "LANGUAGE THE STUDENT WRITES TO YOU IN. If the student writes to you "
    "in the language you're teaching — even a full sentence, even "
    "correctly — that is a PRACTICE ATTEMPT for you to acknowledge and "
    "gently correct or affirm, not a cue to switch your own reply into "
    "that language. Your reply's base language always stays the "
    "student's native language, no matter what language their message "
    "was written in.\n"
    "Every time you write a word or phrase in the language you're "
    "TEACHING (not the student's native language), IMMEDIATELY follow it "
    "with its reading/romanization in square brackets, e.g. "
    "ありがとうございます[arigatou gozaimasu] when teaching Japanese, or "
    "hello[ハロー] when teaching English to a Japanese speaker. Use this "
    "exact bracket format every time — it's read separately from the word "
    "itself, so never skip it and never use parentheses or another style "
    "of bracket for this purpose.\n"
    "Keep explanations short and natural — you're having a conversation, "
    "not reciting a textbook entry.\n"
    "═══════════════════════════════════════════════════════\n\n"
)

_MEMORY_CONTEXT = (
    "═══════════════════════════════════════════════════════\n"
    "CONVERSATION MEMORY — RULES\n"
    "═══════════════════════════════════════════════════════\n"
    "You are in an ONGOING conversation with the user. You MUST:\n"
    "1. REMEMBER what the user has told you about themselves (name, preferences, etc.) — "
    "using ONLY their own messages in the history below, never your own character persona.\n"
    "2. Reference previous messages when relevant\n"
    "3. NEVER say 'we just started talking' if you have conversation history below\n"
    "4. NEVER say you don't know the user's name if they already told you\n"
    "5. NEVER invent or hallucinate names the user never mentioned\n"
    "6. If asked 'what is my name?' or 'what do you know about me?', check the conversation "
    "history below first — and only the *user's own turns* in it, never your own persona/identity "
    "block above. Your name, background, and traits belong to you, not the user.\n"
    "═══════════════════════════════════════════════════════\n\n"
)


async def think(user_text: str, system_prompt: str, history: list,
                character_name: str = None, teaching_mode: bool = False) -> dict:
    if not ai_available():
        beh = sentiment_behavior(user_text)

        print("AI unavailable: GROQ_API_KEY is not set.")
        return {"reply": pick_fallback_reply(),
                "expression": beh["expression"], "animation": "offline", "_fallback": True}

    intent = "direct" if _DIRECT_PAT.search(user_text) or len(user_text.split()) <= 4 else "general"
    wrapped = _wrap_prompt(system_prompt, intent, character_name or "", teaching_mode=teaching_mode)

    full_system = (
        f"{_MEMORY_CONTEXT}"
        f"{_TEACHING_MODE_GUIDE if teaching_mode else ''}"
        f"{wrapped}\n\n"
        "You also direct a 3D avatar's face and body. Pick the expression and animation "
        "that actually match the moment — don't default to neutral/talk out of habit:\n"
        f"- expression, one of {VALID_EXPRESSIONS}\n"
        f"- animation, one of {VALID_ANIMATIONS}\n"
        "Use these exact strings — lowercase, spelled exactly as listed. Do NOT substitute a "
        "synonym (e.g. 'wave', 'smile', 'grateful' are NOT valid — use 'greeting' / 'thankful').\n\n"
        "GUIDE (use whichever fits the user's message, in priority order):\n"
        "- User is greeting you (hi, hello, hey, good morning, konnichiwa, etc.) "
        "→ expression 'happy', animation 'greeting'\n"
        "- User is thanking you or complimenting you (thanks, thank you, great job, nice, etc.) "
        "→ expression 'happy', animation 'thankful'\n"
        "- User is saying goodbye (bye, see you, take care) "
        "→ expression 'relaxed', animation 'greeting'\n"
        "- User agrees, confirms, or affirms something (yes, exactly, that's right, okay, got it, sounds good, sure) "
        "or shares good news / something positive happened "
        "→ expression 'happy', animation 'nod'\n"
        "- User shares bad news, an error, or something went wrong "
        "→ expression 'sad', animation 'talk'\n"
        "- User asks a question, or you're giving an explanation or answering directly "
        "→ expression 'thinking' or 'neutral', animation 'explain'\n"
        "- Ordinary back-and-forth conversation with none of the above "
        "→ expression 'neutral', animation 'talk'\n\n"
        "Note: 'think'/'thinking' is NOT a valid animation — that clip is reserved for the frontend's "
        "own loading state while it's waiting for a response, never for a finished reply. Use 'explain' "
        "instead when the mood calls for it; 'thinking' is only ever an expression, never an animation.\n\n"
        "EXAMPLES (copy this exact JSON shape and these exact field values for these cases):\n"
        'User: "hi" → {"reply": "Hey there! Great to see you.", "expression": "happy", "animation": "greeting"}\n'
        'User: "thanks so much!" → {"reply": "You\'re very welcome!", "expression": "happy", "animation": "thankful"}\n'
        'User: "yes, exactly right!" → {"reply": "Glad that lines up!", "expression": "happy", "animation": "nod"}\n\n'
        "Follow the RESPONSE MODE and LENGTH instructions above.\n"
        + (
            "REMINDER: your explanatory language is fixed by your role, not by "
            "what the student just typed — do not mirror the language of the "
            "message above.\n"
            if teaching_mode else ""
        ) +
        'Output ONLY JSON: {"reply": "<your response>", "expression": "<expr>", "animation": "<animation>"}'
    )

    messages = [{"role": "system", "content": full_system}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_text})

    if DEV_LOGGING:
        print(f"📨 Sending {len(messages)} messages to LLM ({len(history)} history messages)")

    # low effort + higher temperature: this is casual, in-character chat,
    # not a reasoning task — low effort keeps latency down and avoids the
    # model drifting into an "explaining my answer" tone, 0.85 (vs
    # call_llm's 0.7 default) adds warmth/variety to phrasing.
    try:
        # Teaching mode needs to track a role-based direction rule against
        # drift toward mirroring the student's input language — worth the
        # extra latency that "low" trades away.
        effort = "medium" if teaching_mode else "low"
        raw = await call_llm(messages, json_mode=True, temperature=0.85, reasoning_effort=effort)
        try:
            data = json.loads(normalize_json_like(raw))
        except Exception:
            data = _extract_behavior_json(raw)

        if not data.get("reply"):
            data = _extract_behavior_json(raw)

        reply      = str(data.get("reply", "")).strip()
        expression = str(data.get("expression", "neutral")).lower().strip()
        animation  = str(data.get("animation", "talk")).lower().strip()

        if not reply:
            reply = raw.strip() or "..."
        if expression not in VALID_EXPRESSIONS:
            expression = "neutral"
        if animation not in VALID_ANIMATIONS:
            animation = "talk"

        return {"reply": reply, "expression": expression, "animation": animation}

    except Exception as e:
        print(f"AI Think Error: {e}")
        beh = sentiment_behavior(user_text)
        error_text = str(e)
        if "rate_limit_exceeded" in error_text:
            reply = pick_rate_limit_reply(error_text)
        else:
            reply = pick_fallback_reply()
        return {"reply": reply,
                "expression": beh["expression"], "animation": "offline", "_fallback": True}

# ==========================================
# 7. REQUEST MODELS
# ==========================================
class AskRequest(BaseModel):
    text: str
    avatar_persona: Optional[str] = None
    character_name: Optional[str] = None
    voice_en: Optional[str] = None
    voice_ja: Optional[str] = None
    speak_language: Optional[str] = "en"
    timezone: Optional[str] = None
    # Set by the frontend for language-tutor avatars (e.g. Tokyo/Hikaru) —
    # switches /ask to a single mixed-language reply + single TTS pass
    # instead of the normal English reply + separate JA translation/track.
    teaching_mode: Optional[bool] = False

class VoiceRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    culture: Optional[str] = "en"

# ==========================================
# 8. ENDPOINTS
# ==========================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_enabled": ai_available(),
        "provider": "groq",
        "model": GROQ_MODEL,
        "memory_mode": "temporary_memory",
    }

@app.post("/ask")
async def ask_avatar(
    request: AskRequest,
    scope: str = Depends(get_settings_scope),
    app_id: str = Depends(get_app_id),
    settings_group: str = Depends(get_settings_group),
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    user_text = request.text.strip()
    if not user_text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    user_for_ai = (
        user_text if request.teaching_mode
        else (await translate_to_english(user_text) if is_japanese(user_text) else user_text)
    )

    system_prompt = build_character_system(
        user_for_ai,
        request.character_name,
        request.avatar_persona,
    )

    settings_row = session.get(AppSettings, (app_id, settings_group)) if scope == "app" else session.get(UserSettings, user_id)
    stored_response_language = settings_row.response_language if settings_row else None
    effective_language = request.speak_language
    if stored_response_language and (not effective_language or effective_language == "en"):
        effective_language = stored_response_language

    primary      = "ja" if effective_language == "ja" else "en"

    # Chat history is now per-user (X-User-Id) as well as per-avatar
    # (character_name) — a row belongs to exactly one user, and each user
    # sees only their own avatars' turns. Otherwise the model would see
    # (and get confused by, or leak) another user's or another avatar's
    # conversation.
    active_character_name = request.character_name or None
    own_turns = session.exec(
        select(ChatMessage)
        .where(ChatMessage.user_id == user_id)
        .where(ChatMessage.character_name == active_character_name)
        .order_by(ChatMessage.id)
    ).all()

    # The LLM's message schema only accepts {role, content} — anything else
    # gets a 400 — so strip each stored row down to that here rather than
    # passing the full rows through.
    history = [{"role": h.role, "content": h.content or ""} for h in own_turns[-20:]]

    behavior = await think(user_for_ai, system_prompt, history,
                           character_name=request.character_name,
                           teaching_mode=request.teaching_mode)
    reply_en = behavior.get("reply", "...")

    if request.teaching_mode:
        # The model already wrote one mixed-language reply — there's
        # nothing to translate, and only one audio track to generate.
        reply_ja = reply_en
        romanization = ""

        # Both set to the same resolved voice — teaching-mode avatars use
        # one Multilingual voice for both fields (see AvatarSources.js) —
        # and en_voice/ja_voice must exist here too since the final
        # response dict below references them unconditionally regardless
        # of which branch ran.
        en_voice = resolve_voice(request.voice_en, use_japanese=False)
        ja_voice = en_voice

        # Strip the [romaji] brackets before TTS so the voice doesn't say
        # each Japanese word twice (once in kana, once as literal roman
        # letters) — the bracketed text stays in reply_en/reply_ja for the
        # frontend to caption, just not in what's actually spoken.
        speech_text = re.sub(r"\[[^\]]*\]", "", reply_en)

        audio_name = next_audio_name("temp_teach")
        audio_path = os.path.join("static", audio_name)
        # Slower than normal conversational pace — a language learner
        # needs to actually catch the words, especially the Japanese
        # portions, not just hear a natural-speed sentence blur past.
        visemes, generated = await safe_tts(speech_text, en_voice, audio_path, rate="-20%")

        final_audio_url = f"/static/{audio_name}" if generated else ""
        audio_url_en = final_audio_url
        audio_url_ja = final_audio_url
        visemes_en = visemes
        visemes_ja = visemes
    else:
        translation = await translate_to_japanese(
            reply_en,
            character_name=request.character_name,
            persona=request.avatar_persona,
        )
        reply_ja = translation.get("japanese", "")
        romanization = translation.get("romanization", "")

        en_voice = resolve_voice(request.voice_en, use_japanese=False)
        ja_voice = resolve_voice(request.voice_ja, use_japanese=True)

        en_name = next_audio_name("temp_en")
        ja_name = next_audio_name("temp_ja")
        en_path = os.path.join("static", en_name)
        ja_path = os.path.join("static", ja_name)

        visemes_en, en_generated = await safe_tts(reply_en, en_voice, en_path)
        visemes_ja, ja_generated = await safe_tts(reply_ja, ja_voice, ja_path)

        audio_url_en = f"/static/{en_name}" if en_generated else ""
        audio_url_ja = f"/static/{ja_name}" if ja_generated else ""
        final_audio_url = audio_url_ja if primary == "ja" else audio_url_en

    # A fallback reply (LLM unavailable/erroring) isn't a real exchange — the
    # avatar didn't actually understand or respond to what the user said, so
    # persisting it would both clutter the chat history panel with canned
    # "having trouble thinking" lines once things are back to normal, and
    # feed a non-answer into the LLM's own memory of the conversation next
    # time. Only log turns where the AI genuinely replied.
    if not behavior.get("_fallback"):
        session.add(ChatMessage(
            user_id=user_id, character_name=active_character_name,
            role="user", content=user_for_ai, text=user_text,
        ))
        session.add(ChatMessage(
            user_id=user_id, character_name=active_character_name,
            role="assistant", content=reply_en, text_en=reply_en, text_ja=reply_ja,
        ))
        session.commit()

    animation = behavior.get("animation", "explain")
    return {
        "reply": reply_en, "translated_reply": reply_ja, "romanization": romanization,
        "expression": behavior.get("expression", "neutral"),
        "animation": animation,
        "audio_url_en": audio_url_en,
        "audio_url_ja": audio_url_ja,
        "audio_url": final_audio_url,
        "visemes_en": visemes_en, "visemes_ja": visemes_ja,
        "visemes": visemes_ja if primary == "ja" else visemes_en,
        "primary": primary,
        "voice": ja_voice if primary == "ja" else en_voice,
        "mode": "temporary",
    }

@app.post("/voice")
async def generate_voice(request: VoiceRequest):
    text = (request.text or "").strip()
    if not text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    is_ja = (request.culture or "en") == "ja"
    voice_name = request.voice or resolve_voice(None, use_japanese=is_ja)
    output_name = next_audio_name("temp_voice")
    output_path = os.path.join("static", output_name)

    visemes, generated = await safe_tts(text, voice_name, output_path)
    audio_url = f"/static/{output_name}" if generated else ""
    return {
        "audio_url": audio_url,
        "visemes": visemes,
        "voice": voice_name,
    }

@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...), language: Optional[str] = Form(None)):

    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse({"error": "Empty audio"}, status_code=400)

    lang_hint = language if language in ("en", "ja") else None

    try:
        text = await transcribe_audio(audio_bytes, filename=audio.filename or "audio.webm", language=lang_hint)
    except Exception as e:
        print(f"STT Error: {e}")
        return JSONResponse({"error": "Transcription failed"}, status_code=502)

    if not text:
        return JSONResponse({"error": "No speech detected"}, status_code=422)

    return {"text": text}

@app.post("/translate")
async def translate_text(text: str = Form(...), target: str = Form("ja")):
    if target == "en":
        return {"text": await translate_to_english(text), "romanization": ""}
    result = await translate_to_japanese(text)
    return {"text": result["japanese"], "romanization": result["romanization"]}

@app.get("/history")
def get_history(
    character_name: Optional[str] = Query(None),
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    query = select(ChatMessage).where(ChatMessage.user_id == user_id)
    if character_name is not None:
        query = query.where(ChatMessage.character_name == character_name)
    rows = session.exec(query.order_by(ChatMessage.id)).all()

    history = [
        {
            "role": h.role,
            "content": h.content,
            "text": h.text,
            "text_en": h.text_en,
            "text_ja": h.text_ja,
            "character_name": h.character_name,
            "time": h.time_iso(),   # was h.time.isoformat()
        }
        for h in rows
    ]
    return {"history": history}

@app.post("/reset")
def reset_conversation(
    character_name: Optional[str] = Query(None),
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    # Only clears this user's own history. Pass ?character_name=<avatar> to
    # clear just that avatar's turns; omitting it clears everything for
    # this user (matches the settings panel's "Clear history" button, which
    # doesn't scope by avatar today).
    query = select(ChatMessage).where(ChatMessage.user_id == user_id)
    if character_name is not None:
        query = query.where(ChatMessage.character_name == character_name)
    for row in session.exec(query).all():
        session.delete(row)
    session.commit()
    return {"status": "cleared", "mode": "sqlite"}


class SettingsRequest(BaseModel):
    ui_language: Optional[str] = None
    response_language: Optional[str] = None
    last_avatar: Optional[str] = None
    # Opaque blob from the frontend — keyed "instanceId::avatarId" — stored
    # and returned as-is, never inspected/validated here.
    persona_overrides: Optional[dict] = None

@app.get("/settings")
def get_settings(
    scope: str = Depends(get_settings_scope),
    app_id: str = Depends(get_app_id),
    settings_group: str = Depends(get_settings_group),
    user_id: Optional[str] = Depends(get_user_id_optional),
    session: Session = Depends(get_session),
):
    if scope == "app":
        row = session.get(AppSettings, (app_id, settings_group))
    else:
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing X-User-Id header")
        row = session.get(UserSettings, user_id)
    if row is None:
        return {"ui_language": "en", "response_language": "ja", "last_avatar": None, "persona_overrides": {}}
    try:
        overrides = json.loads(row.persona_overrides) if row.persona_overrides else {}
    except Exception:
        overrides = {}
    return {
        "ui_language": row.ui_language,
        "response_language": row.response_language,
        "last_avatar": row.last_avatar,
        "persona_overrides": overrides,
    }

@app.post("/settings")
def save_settings(
    request: SettingsRequest,
    scope: str = Depends(get_settings_scope),
    app_id: str = Depends(get_app_id),
    settings_group: str = Depends(get_settings_group),
    user_id: Optional[str] = Depends(get_user_id_optional),
    session: Session = Depends(get_session),
):
    if scope == "app":
        row = session.get(AppSettings, (app_id, settings_group))
        if row is None:
            row = AppSettings(app_id=app_id, settings_group=settings_group)
            session.add(row)
    else:
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing X-User-Id header")
        row = session.get(UserSettings, user_id)
        if row is None:
            row = UserSettings(user_id=user_id)
            session.add(row)

    if request.ui_language is not None:
        row.ui_language = request.ui_language
    if request.response_language is not None:
        row.response_language = request.response_language
    if request.last_avatar is not None:
        row.last_avatar = request.last_avatar
    if request.persona_overrides is not None:
        row.persona_overrides = json.dumps(request.persona_overrides)

    session.commit()
    session.refresh(row)
    try:
        overrides = json.loads(row.persona_overrides) if row.persona_overrides else {}
    except Exception:
        overrides = {}
    return {
        "ui_language": row.ui_language,
        "response_language": row.response_language,
        "last_avatar": row.last_avatar,
        "persona_overrides": overrides,
    }