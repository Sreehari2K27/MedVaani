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


def get_gnani_api_key() -> str | None:
    """Return the configured Gnani API key, or None when it is not set."""
    value = os.getenv(GNANI_API_KEY_ENV_VAR, "")
    return value.strip() or None
