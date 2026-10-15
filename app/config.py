"""Environment configuration for MedVaani.

Secrets are read from the process environment, optionally populated from a
.env file at the repository root. The .env file is git-ignored; only the
placeholder in .env.example is ever committed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)

GNANI_API_KEY_ENV_VAR = "GNANI_API_KEY"

MISTRAL_API_KEY_ENV_VAR = "MISTRAL_API_KEY"
MISTRAL_MODEL_ENV_VAR = "MISTRAL_MODEL"
MISTRAL_TIMEOUT_ENV_VAR = "MISTRAL_TIMEOUT_SECONDS"
MISTRAL_AE_ENABLED_ENV_VAR = "MISTRAL_AE_ENABLED"
MISTRAL_MI_ENABLED_ENV_VAR = "MISTRAL_MI_ENABLED"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
DEFAULT_MISTRAL_TIMEOUT_SECONDS = 10.0
_FLAG_TRUTHY = frozenset({"1", "true", "yes", "on"})


def get_gnani_api_key() -> str | None:
    """Return the configured Gnani API key, or None when it is not set."""
    value = os.getenv(GNANI_API_KEY_ENV_VAR, "")
    return value.strip() or None


def get_mistral_api_key() -> str | None:
    """Return the configured Mistral API key, or None when it is not set."""
    value = os.getenv(MISTRAL_API_KEY_ENV_VAR, "")
    return value.strip() or None


def get_mistral_model() -> str:
    """Return the configured Mistral model name (never a secret)."""
    value = os.getenv(MISTRAL_MODEL_ENV_VAR, "")
    return value.strip() or DEFAULT_MISTRAL_MODEL


def get_mistral_timeout_seconds() -> float:
    """Return the Mistral request timeout in seconds (invalid values -> default)."""
    value = os.getenv(MISTRAL_TIMEOUT_ENV_VAR, "")
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MISTRAL_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_MISTRAL_TIMEOUT_SECONDS
    return timeout


def _mistral_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _FLAG_TRUTHY


def mistral_ae_enabled() -> bool:
    """Mistral AE interpretation is opt-in: key present AND flag on (default off)."""
    return bool(get_mistral_api_key()) and _mistral_flag(MISTRAL_AE_ENABLED_ENV_VAR)


def mistral_mi_enabled() -> bool:
    """Mistral MI/RAG answering is opt-in: key present AND flag on (default off)."""
    return bool(get_mistral_api_key()) and _mistral_flag(MISTRAL_MI_ENABLED_ENV_VAR)
