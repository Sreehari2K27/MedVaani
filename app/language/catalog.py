"""Language and voice catalog shared by the mirroring layer.

Single place that maps a BCP-47 language code to a Gnani Timbre voice and
declares which languages this prototype can actually speak. It adds no new
provider dependency: the codes are the ones the Prisma/Timbre adapters already
accept.
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "en-IN"
DEFAULT_VOICE = "Kaveri"

# Languages this layer understands (mirrors Prisma STT + Timbre TTS support).
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
)

# One default Timbre voice per language (documented "available voices" panel).
VOICE_FOR_LANGUAGE: dict[str, str] = {
    "bn-IN": "Kirra",
    "en-IN": "Kaveri",
    "gu-IN": "Falak",
    "hi-IN": "Nalini",
    "kn-IN": "Saanvi",
    "ml-IN": "Reshma",
    "mr-IN": "Zahira",
    "pa-IN": "Mehuli",
    "ta-IN": "Asmita",
    "te-IN": "Suhana",
}


def supports_language(language: str | None) -> bool:
    """Whether ``language`` is one of the BCP-47 codes handled here."""
    return isinstance(language, str) and language in SUPPORTED_LANGUAGES


def normalize_language(language: str | None) -> str | None:
    """Coerce a loose provider/endpoint value to a supported BCP-47 code.

    Accepts a bare ISO-639-1 code (``"hi"``) and the ``en-IN`` style. Returns
    ``None`` when the value is not one we can speak.
    """
    if not isinstance(language, str) or not language.strip():
        return None
    candidate = language.strip()
    if candidate in SUPPORTED_LANGUAGES:
        return candidate
    base = candidate.split("-", 1)[0].lower()
    for code in SUPPORTED_LANGUAGES:
        if code.split("-", 1)[0] == base:
            return code
    return None


def voice_for(language: str | None) -> str:
    """Default Timbre voice for ``language`` (falls back to English)."""
    if isinstance(language, str) and language in VOICE_FOR_LANGUAGE:
        return VOICE_FOR_LANGUAGE[language]
    return DEFAULT_VOICE
