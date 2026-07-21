"""
json_utils.py — shared helpers for cleaning up and parsing the loosely-formed
JSON an LLM sometimes returns (smart quotes, single quotes, trailing commas,
stray prose around the object, code fences, etc).

Used by backend.py to parse the {reply, expression, animation} behavior
object. translation.py has its own variant tuned specifically for the
{japanese, romanization} shape and is left as-is.
"""
import re

_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
}


def strip_fences(raw: str) -> str:
    clean = raw.strip()
    for fence in ("```json", "```"):
        clean = clean.replace(fence, "")
    return clean.strip()


def normalize_json_like(raw: str) -> str:
    """Best-effort repair of common LLM JSON mistakes into parseable JSON text."""
    normalized = strip_fences(raw)
    for smart, plain in _QUOTE_MAP.items():
        normalized = normalized.replace(smart, plain)

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start != -1 and end > start:
        normalized = normalized[start:end + 1]

    # 'key': -> "key":
    normalized = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:\s*", r'"\1": ', normalized)

    # : 'value' -> : "value"  (preserving apostrophes/escapes inside the value)
    def _replace_single_quoted_value(match):
        raw_val = match.group(1).replace('\\', '\\\\').replace('"', '\\"')
        return f': "{raw_val}"'

    normalized = re.sub(
        r":\s*'((?:\\.|[^'])*?)'\s*(?=[,\}])",
        _replace_single_quoted_value,
        normalized,
        flags=re.S,
    )

    # bare_key: -> "bare_key":
    normalized = re.sub(r"(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:\s*", r'"\1": ', normalized)

    # trailing commas
    normalized = re.sub(r",\s*\}", "}", normalized)
    normalized = re.sub(r",\s*\]", "]", normalized)
    return normalized.strip()


def extract_quoted_value(text: str, field_name: str) -> str | None:
    """Pull out `"field_name": "<value>"` (or single-quoted) from raw, malformed text."""
    field_pattern = re.compile(rf"['\"]{re.escape(field_name)}['\"]\s*:\s*", re.IGNORECASE)
    match = field_pattern.search(text)
    if not match:
        return None

    index = match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] not in ("'", '"'):
        return None

    quote = text[index]
    index += 1
    value_chars = []
    escaped = False
    while index < len(text):
        ch = text[index]
        index += 1
        if escaped:
            value_chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == quote:
            return ''.join(value_chars)
        value_chars.append(ch)
    return None