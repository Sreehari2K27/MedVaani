"""Offline spoken-language identification for the mirroring layer.

The Prisma REST endpoint requires a fixed ``language_code`` and returns no
detected-language field, so the language of a turn is inferred from the
transcript itself. This is deliberately standard-library-only and
deterministic: Unicode script blocks identify the Indic languages directly,
Devanagari is disambiguated between Hindi and Marathi with function words, and
Latin text is treated as English unless romanised Hindi markers are present
(which is what Prisma's English model emits for Hindi speech).
"""

from __future__ import annotations

import re
from typing import Optional

from .catalog import SUPPORTED_LANGUAGES, normalize_language

# Non-Devanagari Indic script blocks -> BCP-47 code.
_SCRIPT_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0980, 0x09FF, "bn-IN"),  # Bengali
    (0x0A00, 0x0A7F, "pa-IN"),  # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gu-IN"),  # Gujarati
    (0x0B80, 0x0BFF, "ta-IN"),  # Tamil
    (0x0C00, 0x0C7F, "te-IN"),  # Telugu
    (0x0C80, 0x0CFF, "kn-IN"),  # Kannada
    (0x0D00, 0x0D7F, "ml-IN"),  # Malayalam
)

_DEVANAGARI_RANGE = (0x0900, 0x097F)

# Devanagari function words. Marathi and Hindi share the script, so the
# decision is made on markers that are common in one but not the other.
_MARATHI_MARKERS = ("आहे", "आणि", "मला", "नाही", "तुम्ही", "होते", "झाले", "पण", "खूप")
_HINDI_MARKERS = ("है", "और", "मुझे", "नहीं", "आप", "मैं", "हुआ", "बहुत", "लेकिन", "दर्द")

# Romanised Hindi words Prisma's English model tends to emit for Hindi audio.
_ROMANIZED_HINDI = frozenset(
    {
        "mujhe", "dard", "nahi", "nahin", "kyun", "matlab", "thoda", "bahut",
        "hua", "huaa", "mera", "meri", "hai", "hain", "kya", "kaise", "kyu",
        "theek", "accha", "bhai", "se", "kal", "aaj", "raha", "rahi", "gaya",
        "gayi", "bahot", "zada", "jyada", "takleef", "taklif", "dawai", "dawa",
        # Conversational vocabulary the AE extraction now understands
        # (affirmations, clarifications, outcome/status words).
        "haan", "han", "ji", "hoon", "main", "khud", "gambhir", "sahmat",
        "bilkul", "dobara", "shayad", "behtar", "yaad",
    }
)
# Unambiguous enough that a single occurrence implies Hindi, not English.
# Only words that never appear in English are listed here.
_STRONG_ROMANIZED_HINDI = frozenset(
    {
        "mujhe", "dard", "nahi", "nahin", "kyun", "matlab", "takleef", "taklif",
        "dawai", "haan", "khud", "gambhir", "sahmat", "bilkul", "dobara",
        "shayad", "behtar",
    }
)

# Latin replies shorter than this many word tokens carry no reliable language
# evidence (names, "yes", "myself", "haan" before it was marked): they keep
# the conversation's current language instead of reverting an established
# session (e.g. Hindi) to English. Full English sentences are unaffected.
_MIN_ENGLISH_TOKENS = 5

_LETTERS = re.compile(r"[A-Za-z]")
_WORDS = re.compile(r"[A-Za-z]+")


def _count_scripts(text: str) -> tuple[int, dict[str, int]]:
    devanagari = 0
    others: dict[str, int] = {}
    for char in text:
        codepoint = ord(char)
        if _DEVANAGARI_RANGE[0] <= codepoint <= _DEVANAGARI_RANGE[1]:
            devanagari += 1
            continue
        for low, high, code in _SCRIPT_BLOCKS:
            if low <= codepoint <= high:
                others[code] = others.get(code, 0) + 1
                break
    return devanagari, others


def _devanagari_language(text: str) -> str:
    marathi = sum(text.count(marker) for marker in _MARATHI_MARKERS)
    hindi = sum(text.count(marker) for marker in _HINDI_MARKERS)
    return "mr-IN" if marathi > hindi else "hi-IN"


def _latin_language(text: str) -> Optional[str]:
    tokens = [token.lower() for token in _WORDS.findall(text)]
    if not tokens:
        return None
    markers = {token for token in tokens if token in _ROMANIZED_HINDI}
    if len(markers) >= 2 or any(token in _STRONG_ROMANIZED_HINDI for token in tokens):
        return "hi-IN"
    # Short marker-less fragments ("myself", "yes that is fine", a name) are
    # not evidence of English: keep whatever language the session established.
    if len(tokens) < _MIN_ENGLISH_TOKENS:
        return None
    return "en-IN"


def detect_language(
    text: str,
    provider_language: str | None = None,
) -> Optional[str]:
    """Identify the language of ``text``.

    A provider-supplied language, when present and supported, always wins (the
    documented "reliable provider metadata" path). Otherwise the transcript is
    inspected. Returns ``None`` when nothing identifiable is present so the
    caller can keep the conversation's current language.
    """
    provider = normalize_language(provider_language)
    if provider is not None:
        return provider

    if not text or not text.strip():
        return None
    if not _LETTERS.search(text) and not _count_scripts(text)[0] and not _count_scripts(text)[1]:
        return None

    devanagari, others = _count_scripts(text)
    if others:
        code = max(others, key=lambda key: others[key])
        # A distinct Indic script is decisive unless Devanagari clearly dominates.
        if others[code] >= devanagari:
            return code
    if devanagari:
        return _devanagari_language(text)
    return _latin_language(text)


def is_supported(language: str | None) -> bool:
    return normalize_language(language) is not None


__all__ = ["detect_language", "is_supported", "SUPPORTED_LANGUAGES"]
