import ast
import re
import json
import logging

try:
    from .ai import call_llm, ai_available, call_translation_llm, translation_ai_available
except ImportError:
    from ai import call_llm, ai_available, call_translation_llm, translation_ai_available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Accept Japanese punctuation and fullwidth digits as valid Japanese text for
# translation validation, since outputs like counting may produce １、２、３ or
# Japanese commas without hiragana/kanji.
_JP_PATTERN = re.compile(r"[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF10-\uFF19]")

_translation_cache: dict = {}

_MAX_CACHE_SIZE = 200
_MAX_TRANSLATE_CHARS = 1000

# Completely empty to stop false-positives on valid technical Japanese words like 機械学習
_SIMPLIFIED_ONLY_CHARS = set()

# Common legitimate katakana words
_COMMON_KATAKANA = {
    "コンピュータ",
    "コンピューター",
    "プログラム",
    "システム",
    "データ",
    "ネットワーク",
    "インターネット",
    "プラットフォーム",
    "パフォーマンス",
    "アーキテクチャ",
    "モデル",
    "フレームワーク",
    "クラウド",
    "サービス",
    "アメリカ",
    "イギリス",
    "フランス",
    "ドイツ",
    "日本",
    "中国",
    "テレビ",
    "ラジオ",
    "カメラ",
    "バス",
    "タクシー",
    "ホテル",
    "プログラミング",
    "ビジネス",
    "サイエンス",
    "アドバイス",
    "コーディング",
}

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def is_japanese(text: str) -> bool:
    if not text:
        return False
    return bool(_JP_PATTERN.search(text))


def _sanitize_text(text: str) -> str:
    """
    Remove invalid unicode while preserving valid Japanese.
    """
    if not text:
        return ""

    cleaned = []

    for char in text:
        cp = ord(char)

        if 0xD800 <= cp <= 0xDFFF:
            continue

        cleaned.append(char)

    text = "".join(cleaned)

    text = (
        text
        .encode("utf-8", "ignore")
        .decode("utf-8")
        .strip()
    )

    return text


def _strip_translation_label_prefix(text: str) -> str:
    """Strip common translation label prefixes from a line of text."""
    if not text:
        return text

    text = re.sub(
        r'(?i)^\s*(?:Japanese|日本語|Line\s*1)\s*[:\-]\s*',
        '',
        text,
    ).strip()

    text = re.sub(
        r'(?i)^\s*(?:Romaji|ローマ字|Reading|Line\s*2)\s*[:\-]\s*',
        '',
        text,
    ).strip()

    return text


def _is_valid_japanese(text: str) -> bool:
    """
    Validate translation without rejecting
    legitimate technical Japanese.
    """

    if not text:
        return False

    text = _sanitize_text(text)

    if not is_japanese(text):
        logger.warning(f"❌ Validation Failed: Text contains NO Japanese characters. Text: '{text}'")
        return False

    # reject pure ASCII
    if re.fullmatch(
        r"[A-Za-z0-9\s.,!?()'\":;/\\-]+",
        text,
    ):
        logger.warning(f"❌ Validation Failed: Text is pure ASCII/English. Text: '{text}'")
        return False

    # reject replacement characters
    if "\ufffd" in text:
        logger.warning(f"❌ Validation Failed: Text contains invalid replacement characters. Text: '{text}'")
        return False

    # reject simplified-only Chinese characters
    if any(ch in _SIMPLIFIED_ONLY_CHARS for ch in text):
        matched_chars = [ch for ch in text if ch in _SIMPLIFIED_ONLY_CHARS]
        logger.warning(f"❌ Validation Failed: Text contains Chinese-only characters {matched_chars}. Text: '{text}'")
        return False

    # reject absurd repetition
    if re.search(r"(.)\1{8,}", text):
        logger.warning(f"❌ Validation Failed: Text contains absurd character repetition. Text: '{text}'")
        return False

    return True


def _is_placeholder_text(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True

    if text in {"...", "…..", "……", "…", "-", "--", "---"}:
        return True

    if re.fullmatch(r"^[\.。…\-\s]{1,8}$", text):
        return True

    return False


# ---------------------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------------------

def _clean_llm_output(raw: str) -> str:
    """
    Clean malformed JSON returned by LLMs.
    """

    raw = raw.strip()
    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.replace("\u2018", "'")
    raw = raw.replace("\u2019", "'")
    raw = raw.replace("\u201c", '"')
    raw = raw.replace("\u201d", '"')

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    raw = raw.replace("\\n", " ")
    raw = raw.replace("\n", " ")

    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)

    return raw.strip()


def _normalize_json_like(raw: str) -> str:
    normalized = _clean_llm_output(raw)

    normalized = normalized.replace("\u2018", "'")
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"')
    normalized = normalized.replace("\u201d", '"')

    normalized = re.sub(
        r"(?<=\{|,)\s*'([^']+)'\s*:\s*",
        r'"\1": ',
        normalized,
    )
    normalized = re.sub(
        r":\s*'((?:[^'\\]|\\.)*)'(?=\s*[\},])",
        r': "\1"',
        normalized,
    )
    normalized = re.sub(
        r"(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:\s*",
        r'"\1": ',
        normalized,
    )
    normalized = re.sub(r",\s*([}\]])", r"\1", normalized)

    return normalized.strip()


def _extract_json_fallback(raw: str):

    japanese = ""
    romaji = ""

    m = re.search(
        r'"(?:japanese|translation|ja|text)"\s*:\s*"([^"]+)"',
        raw,
        re.IGNORECASE,
    )

    if m:
        japanese = m.group(1).strip()

    if not japanese:
        normalized = raw.replace("\r", "\n")
        for pattern in [
            r'(?i)(?:japanese|translation|ja|text)\s*[:=]\s*"([^"]+)"',
            r"(?i)(?:japanese|translation|ja|text)\s*[:=]\s*'([^']+)'",
            r'(?i)(?:japanese|translation|ja|text)\s*[:=]\s*([^\n,]+)',
        ]:
            m = re.search(pattern, normalized)
            if m:
                japanese = _strip_translation_label_prefix(m.group(1).strip())
                break

    if not japanese:
        jp_lines = [
            line.strip()
            for line in raw.splitlines()
            if is_japanese(line)
        ]

        if jp_lines:
            japanese = _strip_translation_label_prefix(max(jp_lines, key=len))

    m = re.search(
        r'"(?:romanization|romaji|reading)"\s*:\s*"([^"]+)"',
        raw,
        re.IGNORECASE,
    )

    if m:
        romaji = _strip_translation_label_prefix(m.group(1).strip())

    if not romaji:
        normalized = raw.replace("\r", "\n")
        for pattern in [
            r'(?i)(?:romanization|romaji|reading)\s*[:=]\s*"([^"]+)"',
            r"(?i)(?:romanization|romaji|reading)\s*[:=]\s*'([^']+)'",
            r'(?i)(?:romanization|romaji|reading)\s*[:=]\s*([^\n,]+)',
        ]:
            m = re.search(pattern, normalized)
            if m:
                romaji = _strip_translation_label_prefix(m.group(1).strip())
                break

    if not romaji:
        latin_lines = []

        for line in raw.splitlines():
            line = line.strip()
            if (
                len(line) > 3
                and not is_japanese(line)
                and re.search(r"[A-Za-z]", line)
            ):
                latin_lines.append(line)

        if latin_lines:
            romaji = _strip_translation_label_prefix(latin_lines[0])

    if japanese:
        return {
            "japanese": _sanitize_text(japanese),
            "romanization": _sanitize_text(romaji),
        }

    # Try plain key/value extraction without JSON formatting.
    normalized = raw.replace("\r", "\n")
    for pattern in [
        r'(?i)(?:japanese|translation|ja|text)\s*[:=]\s*"([^"]+)"',
        r"(?i)(?:japanese|translation|ja|text)\s*[:=]\s*'([^']+)'",
        r'(?i)(?:japanese|translation|ja|text)\s*[:=]\s*([^\n,]+)',
    ]:
        m = re.search(pattern, normalized)
        if m:
            japanese = m.group(1).strip()
            break

    for pattern in [
        r'(?i)(?:romanization|romaji|reading)\s*[:=]\s*"([^"]+)"',
        r"(?i)(?:romanization|romaji|reading)\s*[:=]\s*'([^']+)'",
        r'(?i)(?:romanization|romaji|reading)\s*[:=]\s*([^\n,]+)',
    ]:
        m = re.search(pattern, normalized)
        if m:
            romaji = m.group(1).strip()
            break

    if not japanese:
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if lines:
            if len(lines) >= 2 and is_japanese(lines[0]) and not is_japanese(lines[1]):
                japanese = lines[0]
                romaji = lines[1]
            elif is_japanese(lines[0]):
                japanese = lines[0]
            elif len(lines) >= 2 and is_japanese(lines[1]):
                japanese = lines[1]

    if not japanese:
        raw_match = re.search(r'([\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF10-\uFF19]+)', raw)
        if raw_match:
            japanese = raw_match.group(1).strip()
            suffix = raw[raw_match.end():].strip()
            if suffix:
                romaji = suffix.splitlines()[0].strip()

    if japanese:
        return {
            "japanese": _sanitize_text(japanese),
            "romanization": _sanitize_text(romaji),
        }

    return None


def _parse_translation_json(raw: str):

    cleaned = _normalize_json_like(raw)
    data = None

    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(cleaned)
            break
        except Exception:
            continue

    if data is None:
        return _extract_json_fallback(raw)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                data = item
                break

    if not isinstance(data, dict):
        return None

    japanese = (
        data.get("japanese")
        or data.get("translation")
        or data.get("ja")
        or data.get("text")
        or ""
    )

    romaji = (
        data.get("romanization")
        or data.get("romaji")
        or data.get("reading")
        or ""
    )

    japanese = _sanitize_text(japanese)
    romaji = _sanitize_text(romaji)

    if not japanese:
        return None

    return {
        "japanese": japanese,
        "romanization": romaji,
    }


# ---------------------------------------------------------------------
# FALLBACK TRANSLATION (NON-JSON)
# ---------------------------------------------------------------------

async def _translate_without_json(text: str, character_name: str = None, persona: str = None) -> dict | None:
    """
    Plain-text fallback when JSON mode fails.
    """

    try:
        context_block = ""
        if character_name or persona:
            context_lines = ["", "Context (for disambiguation only — translate ONLY the line below, do not translate this context):"]
            if character_name:
                context_lines.append(f"- Speaker: {character_name}")
            if persona:
                context_lines.append(f"- Speaker's persona: {persona}")
            context_block = "\n".join(context_lines) + "\n"

        prompt = (
            "You are a professional native Japanese translator.\n\n"
            "Translate the user's English text into NATURAL Japanese.\n"
            f"{context_block}\n"
            "Rules:\n"
            "- Preserve the exact meaning.\n"
            "- Use grammatically correct Japanese.\n"
            "- Use polite Japanese (です・ます).\n"
            "- Use common Kanji.\n"
            "- Keep proper nouns such as Python, ChatGPT, OpenAI and Docker unchanged when appropriate.\n"
            "- Never invent Japanese words.\n"
            "- Never explain your answer.\n\n"
            "Output EXACTLY TWO LINES.\n"
            "Line 1: Japanese\n"
            "Line 2: Romaji\n"
        )

        response = await call_translation_llm(
            [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            json_mode=False,
            temperature=0.2,
            reasoning_effort="low",
        )

        response = response.strip()
        if not response:
            return None

        parsed = _parse_translation_json(response)
        if parsed and parsed.get("japanese"):
            japanese = _sanitize_text(parsed["japanese"])
            romaji = _sanitize_text(parsed.get("romanization") or parsed.get("romaji") or parsed.get("reading") or "")
            if _is_valid_japanese(japanese):
                return {
                    "japanese": japanese,
                    "romanization": romaji,
                }

        lines = [
            x.strip()
            for x in response.splitlines()
            if x.strip()
        ]

        japanese = ""
        romaji = ""

        for line in lines:
            stripped = _strip_translation_label_prefix(line)
            if not japanese and is_japanese(stripped):
                japanese = stripped
                continue
            if japanese and not romaji:
                romaji = _strip_translation_label_prefix(line)
                break

        if not japanese:
            m = re.search(r'(?i)(?:Japanese|日本語)\s*[:=]\s*(.+)', response)
            if m:
                japanese = _strip_translation_label_prefix(m.group(1).strip())
        if not romaji:
            m = re.search(r'(?i)(?:Romaji|ローマ字)\s*[:=]\s*(.+)', response)
            if m:
                romaji = _strip_translation_label_prefix(m.group(1).strip())

        if not japanese:
            m = re.search(r'(?im)^line\s*1\s*[:\-]\s*(.+)$', response)
            if m:
                japanese = _strip_translation_label_prefix(m.group(1).strip())
        if not romaji:
            m = re.search(r'(?im)^line\s*2\s*[:\-]\s*(.+)$', response)
            if m:
                romaji = _strip_translation_label_prefix(m.group(1).strip())

        if not japanese:
            raw_match = re.search(r'([\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF10-\uFF19]+)', response)
            if raw_match:
                japanese = raw_match.group(1).strip()
                suffix = response[raw_match.end():].strip()
                if suffix and not romaji:
                    romaji = _strip_translation_label_prefix(suffix.splitlines()[0].strip())

        japanese = _sanitize_text(japanese)
        romaji = _sanitize_text(romaji)

        if not japanese:
            return None

        if not _is_valid_japanese(japanese):
            logger.warning(
                "Fallback translation failed validation."
            )
            return None

        return {
            "japanese": japanese,
            "romanization": romaji,
        }

    except Exception as e:
        logger.exception(
            "Fallback translation failed: %s",
            e,
        )
        return None


# ---------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------

def _get_from_cache(text: str, character_name: str = None):
    key = f"en_ja:{character_name or ''}:{text[:_MAX_TRANSLATE_CHARS]}"
    return _translation_cache.get(key)


def _save_to_cache(text: str, result: dict, character_name: str = None):
    if len(_translation_cache) >= _MAX_CACHE_SIZE:
        oldest = next(iter(_translation_cache))
        del _translation_cache[oldest]

    key = f"en_ja:{character_name or ''}:{text[:_MAX_TRANSLATE_CHARS]}"
    _translation_cache[key] = result


# ---------------------------------------------------------------------
# MAIN TRANSLATOR
# ---------------------------------------------------------------------

async def translate_to_japanese(text: str, character_name: str = None, persona: str = None) -> dict:

    if not translation_ai_available():
        return {
            "japanese": text,
            "romanization": "",
        }

    if not text.strip():
        return {
            "japanese": "",
            "romanization": "",
        }

    if _is_placeholder_text(text):
        return {
            "japanese": text,
            "romanization": "",
        }

    cached = _get_from_cache(text, character_name)
    if cached:
        return cached

    # Context block, only included when the caller has it — without this,
    # the translator sees nothing but a bare sentence, which is a common
    # source of drift on ambiguous pronouns or tone-dependent phrasing
    # (e.g. a line's politeness level or implied subject depending on who's
    # speaking to whom). Optional and additive: omitting character_name/
    # persona degrades gracefully to the old context-free behavior.
    context_block = ""
    if character_name or persona:
        context_lines = ["", "Context (for disambiguation only — translate ONLY the line below, do not translate this context):"]
        if character_name:
            context_lines.append(f"- Speaker: {character_name}")
        if persona:
            context_lines.append(f"- Speaker's persona: {persona}")
        context_block = "\n".join(context_lines) + "\n"

    prompt = f"""
You are a PROFESSIONAL NATIVE JAPANESE TRANSLATOR.

Translate the user's English into natural Japanese.
{context_block}
Requirements
- Preserve the meaning exactly.
- Maintain absolute factual accuracy with political and institutional titles (e.g., ensure "President" is translated as 大統領 and "Prime Minister" as 首相/総理大臣—never mix them up).
- Use correct Japanese grammar and particles.
- Use polite Japanese (です・ます).
- Use common Japanese vocabulary.
- Never invent Japanese words.
- Never mix Chinese into the translation.
- Proper nouns such as Python, OpenAI, Linux, Docker, ChatGPT may remain unchanged.
- Return ONLY valid JSON.

Example
{{
 "japanese":"私は毎日日本語を勉強しています。",
 "romanization":"Watashi wa mainichi nihongo o benkyou shiteimasu."
}}
"""

    for attempt, json_mode_flag in enumerate((False, True), start=1):
        try:
            raw = await call_translation_llm(
                [
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                json_mode=json_mode_flag,
                # Low, not zero — leaves a little room for natural phrasing
                # while prioritizing fidelity over variety. Was implicitly
                # 0.7 (ai.py's old hardcoded default), which is tuned for
                # creative dialogue, not translation.
                temperature=0.2,
                reasoning_effort="low",
            )

            parsed = _parse_translation_json(raw)
            if not parsed:
                if re.search(r"forgot to include the text|provide the English text|send me the text|text you'd like me to translate|no input text|nothing to translate", raw, re.IGNORECASE):
                    logger.warning(
                        "Attempt %d: Translator indicated no input text was provided. Returning original text.",
                        attempt,
                    )
                    return {
                        "japanese": text,
                        "romanization": "",
                    }
                logger.warning(
                    "Attempt %d: JSON parse failed. Raw translation response: %s",
                    attempt,
                    raw[:500],
                )
                continue

            japanese = parsed.get("japanese")
            if not japanese:
                logger.warning(
                    "Attempt %d: Parsed translation response did not contain Japanese. Raw: %s",
                    attempt,
                    raw[:500],
                )
                continue

            japanese = _sanitize_text(japanese)
            romaji = _sanitize_text(
                parsed.get("romanization")
                or parsed.get("romaji")
                or parsed.get("reading")
                or ""
            )

            if not _is_valid_japanese(japanese):
                logger.warning(
                    "Attempt %d: Validation failed. Japanese text: %s",
                    attempt,
                    japanese[:200],
                )
                continue

            _save_to_cache(text, {
                "japanese": japanese,
                "romanization": romaji,
            }, character_name)
            logger.info("Translation succeeded.")
            return {
                "japanese": japanese,
                "romanization": romaji,
            }

        except Exception as e:
            logger.exception(
                "Translation attempt %d failed: %s",
                attempt,
                e,
            )

    logger.info("Trying fallback translator...")
    fallback = await _translate_without_json(text, character_name, persona)

    if fallback:
        _save_to_cache(text, fallback, character_name)
        return fallback

    # When translation fails entirely, return the original English text as the Japanese fallback,
    # so the UI still has something safe to speak and render.
    logger.error("Translation failed completely.")
    return {
        "japanese": text,
        "romanization": "",
    }


# ---------------------------------------------------------------------
# JAPANESE -> ENGLISH
# ---------------------------------------------------------------------

async def translate_to_english(text: str) -> str:
    """
    Translate Japanese to English.
    Returns clean English text only.
    """

    if not translation_ai_available():
        return text

    if not text or not text.strip():
        return ""

    cache_key = f"ja_en:{text[:_MAX_TRANSLATE_CHARS]}"
    cached = _translation_cache.get(cache_key)

    if cached:
        logger.debug("JA->EN cache hit.")
        return cached

    prompt = """
You are a professional English translator.

Translate Japanese into fluent natural English.

Rules
- Preserve the original meaning.
- Use grammatically correct English.
- Do not summarize.
- Do not explain.
- Do not add information.
- Return ONLY the English translation.
"""

    try:
        result = await call_translation_llm(
            [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            json_mode=False,
            temperature=0.2,
            reasoning_effort="low",
        )

        english = result.strip()
        english = english.strip('"')
        english = _sanitize_text(english)

        if not english:
            return text

        if len(_translation_cache) >= _MAX_CACHE_SIZE:
            oldest = next(iter(_translation_cache))
            del _translation_cache[oldest]

        _translation_cache[cache_key] = english
        logger.info("JA->EN translation successful.")
        return english

    except Exception as e:
        logger.exception(
            "JA->EN translation failed: %s",
            e,
        )
        return text