"""Offline spoken-language mirroring helpers.

Pure, standard-library-only utilities used by the voice endpoint to detect the
patient's language, persist it, pick a Timbre voice and produce a translated
spoken form of the assistant's English reply.
"""

from __future__ import annotations

from .catalog import (
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    SUPPORTED_LANGUAGES,
    VOICE_FOR_LANGUAGE,
    normalize_language,
    supports_language,
    voice_for,
)
from .identification import detect_language
from .translation import TranslationResult, translate, translation_available

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_VOICE",
    "SUPPORTED_LANGUAGES",
    "VOICE_FOR_LANGUAGE",
    "TranslationResult",
    "detect_language",
    "normalize_language",
    "supports_language",
    "translate",
    "translation_available",
    "voice_for",
]
