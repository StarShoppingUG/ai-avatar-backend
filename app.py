import ast
import os
import random
import re
import json
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
    from .db import init_db, get_session, ChatMessage, UserSettings
except ImportError:
    from translation import translate_to_japanese, translate_to_english, is_japanese
    from ai import ai_available, call_llm, GROQ_MODEL, _get_current_date_context, transcribe_audio
    from json_utils import normalize_json_like, extract_quoted_value
    from db import init_db, get_session, ChatMessage, UserSettings

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
    allow_origins=["https://ai-avatar-ui-ghost.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def _on_startup():
    init_db()

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

async def generate_tts_with_visemes(text: str, voice: str, output_path: str) -> list:
    events = []
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
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

async def safe_tts(text: str, voice: str, path: str) -> tuple:
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
        visemes = await generate_tts_with_visemes(clean, voice, path)
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

def _wrap_prompt(system_prompt: str, intent: str, character_name: str) -> str:
    date_context = _get_current_date_context()

    if intent == "direct":
        mode = ("RESPONSE MODE — DIRECT ANSWER: Answer clearly and helpfully. "
                "Keep it concise and practical. MAX 60 WORDS — be concise.\n\n")
    else:
        mode = ("RESPONSE MODE — GENERAL GUIDANCE: Share practical insights with examples. "
                "Be warm, clear, and useful. 80–120 WORDS.\n\n")

    return date_context + mode + system_prompt

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
                character_name: str = None) -> dict:
    if not ai_available():
        beh = sentiment_behavior(user_text)
        # The missing-API-key detail is useful to whoever's running the
        # server, not to the person talking to the avatar — keep it in the
        # server log and give the user the same natural fallback line.
        print("AI unavailable: GROQ_API_KEY is not set.")
        return {"reply": pick_fallback_reply(),
                "expression": beh["expression"], "animation": "offline", "_fallback": True}

    intent = "direct" if _DIRECT_PAT.search(user_text) or len(user_text.split()) <= 4 else "general"
    wrapped = _wrap_prompt(system_prompt, intent, character_name or "")

    full_system = (
        f"{_MEMORY_CONTEXT}"
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
        'Output ONLY JSON: {"reply": "<your response>", "expression": "<expr>", "animation": "<animation>"}'
    )

    messages = [{"role": "system", "content": full_system}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": user_text})

    if DEV_LOGGING:
        print(f"📨 Sending {len(messages)} messages to LLM ({len(history)} history messages)")

    try:
        raw = await call_llm(messages, json_mode=True)
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
        return {"reply": pick_fallback_reply(),
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
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    user_text = request.text.strip()
    if not user_text:
        return JSONResponse({"error": "Empty input"}, status_code=400)

    user_for_ai = await translate_to_english(user_text) if is_japanese(user_text) else user_text

    system_prompt = build_character_system(
        user_for_ai,
        request.character_name,
        request.avatar_persona,
    )

    # `speak_language` on this request and the persisted `response_language`
    # in UserSettings are meant to represent the same choice, but nothing
    # previously tied them together — /ask only ever looked at whatever
    # speak_language happened to arrive on this one request, so a saved
    # response_language preference had no effect on the actual reply
    # unless the frontend also remembered to resend it every time. The
    # saved setting is now the source of truth here, same as ui_language
    # and last_avatar already are elsewhere; an explicit non-default
    # speak_language on the request can still override it for one-off cases.
    settings_row = session.get(UserSettings, user_id)
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
                           character_name=request.character_name)
    reply_en = behavior.get("reply", "...")

    translation = await translate_to_japanese(reply_en)
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
    """
    Server-side speech-to-text via Groq Whisper. Replaces reliance on the
    browser's built-in SpeechRecognition for the *final* transcript — the
    frontend still uses SpeechRecognition for live interim captions (it's
    free/instant), but sends the actual recorded audio here and swaps in
    this result before submitting, since Whisper is meaningfully more
    accurate (accents, background noise, non-English speech).
    """
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
    # All optional — POST /settings is a partial update; only the fields
    # present in the body get written, everything else on the row is left
    # as-is (see save_settings below).
    ui_language: Optional[str] = None
    response_language: Optional[str] = None
    last_avatar: Optional[str] = None

@app.get("/settings")
def get_settings(
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
    row = session.get(UserSettings, user_id)
    if row is None:
        # First time we've seen this user_id — hand back defaults without
        # writing a row yet; POST /settings creates it on first save.
        return {"ui_language": "en", "response_language": "ja", "last_avatar": None}
    return {
        "ui_language": row.ui_language,
        "response_language": row.response_language,
        "last_avatar": row.last_avatar,
    }

@app.post("/settings")
def save_settings(
    request: SettingsRequest,
    user_id: str = Depends(get_user_id),
    session: Session = Depends(get_session),
):
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

    session.commit()
    session.refresh(row)
    return {
        "ui_language": row.ui_language,
        "response_language": row.response_language,
        "last_avatar": row.last_avatar,
    }