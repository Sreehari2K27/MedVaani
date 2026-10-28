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

# Devanagari digit words shared by the hi-IN/mr-IN pair plus the Devanagari
# transliterations STT emits for spoken English digits. Devanagari is the
# only script two supported locales share, so a digit-only utterance (a
# dictated phone number) carries no evidence that can tell Hindi from
# Marathi - and a marker-less tie-break would force Hindi onto a Marathi
# session. Kept module-local: language identification never imports the
# provider layer, and phone extraction is a different layer entirely.
_DEVANAGARI_DIGIT_WORDS = frozenset(
    {
        # Hindi
        "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "छः", "छे",
        "सात", "आठ", "नौ", "शून्य",
        # Marathi
        "दोन", "पाच", "सहा", "नऊ",
        # Devanagari transliterations of spoken English digits
        "वन", "नाइन", "टू", "थ्री", "फोर", "फ़ोर", "फाइव", "सिक्स",
        "सेवन", "एट", "ऐट", "जीरो", "ज़ीरो",
    }
)

# Devanagari digit glyphs (०-९): a run of them is digit content too.
_DEVANAGARI_DIGIT_GLYPHS = re.compile(r"^[\u0966-\u096F]+$")
_DEVANAGARI_TOKENS = re.compile(r"[\u0900-\u097F]+")

# Tokens that never identify a language by themselves: number words, single
# letter-name spellings and the field-spelling cues around them. A phone
# number, email address or identifier dictated in English words inside a
# Malayalam (or any other) session is field content, not a language switch,
# so such replies keep the conversation's established language. A genuine
# English sentence keeps enough content tokens after this filter to detect.
_WEAK_EVIDENCE_TOKENS = frozenset(
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "oh",
        # Spelled-email / identifier cues (the same cues the AE email
        # capture understands).
        "at", "the", "rate", "dot", "underscore", "hyphen",
    }
)

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
    # Weak-evidence tokens (number words, single letters, spelling cues) are
    # removed before the English threshold is applied: dictating a phone
    # number or email address in English words is field content, never a
    # language switch, so it keeps the session's established language.
    content = [
        token
        for token in tokens
        if len(token) > 1 and token not in _WEAK_EVIDENCE_TOKENS
    ]
    if len(content) < _MIN_ENGLISH_TOKENS:
        return None
    return "en-IN"


def _devanagari_digit_words_only(text: str) -> bool:
    """Whether a Devanagari utterance consists entirely of digit words (or
    digit glyphs): dictated-number content with no evidence that can tell
    the shared-script Hindi/Marathi pair apart."""
    tokens = _DEVANAGARI_TOKENS.findall(text)
    if not tokens:
        return False
    for token in tokens:
        if _DEVANAGARI_DIGIT_GLYPHS.match(token):
            continue
        if token in _DEVANAGARI_DIGIT_WORDS:
            continue
        return False
    return True


def _devanagari_marker_present(text: str) -> bool:
    """Whether the text carries a Hindi or Marathi function-word marker.

    Devanagari is the only script two supported locales share, so an
    utterance with no distinguishing marker (short confirmations such as
    "होय"/"बरोबर आहे"-style fragments, names, misrecognitions) cannot tell
    the pair apart and is weak evidence: the caller keeps the session's
    established language instead of forcing the marker-less tie-break.
    """
    for marker in _MARATHI_MARKERS:
        if marker in text:
            return True
    for marker in _HINDI_MARKERS:
        if marker in text:
            return True
    return False


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
        if _devanagari_digit_words_only(text):
            # Digit-only Devanagari (a dictated number): no hi/mr evidence,
            # so the caller keeps the session's established language.
            return None
        if not _devanagari_marker_present(text):
            # Marker-less Devanagari (short confirmations, names,
            # misrecognitions): the hi/mr tie-break would force Hindi onto
            # an established Marathi session, so treat it as weak evidence.
            return None
        return _devanagari_language(text)
    return _latin_language(text)


def is_supported(language: str | None) -> bool:
    return normalize_language(language) is not None


__all__ = ["detect_language", "is_supported", "SUPPORTED_LANGUAGES"]
