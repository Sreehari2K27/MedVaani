"""Mistral LLM provider (isolated implementation)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .base import LLMProvider, LLMResult


class MistralLLMProvider:
    """Mistral provider - isolated, no real calls in tests by default."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        """Interpret message. Raises if API key missing (clean handling)."""
        if not self.api_key:
            raise ValueError("Mistral API key is missing. Set MISTRAL_API_KEY environment variable.")

        # In production this would call Mistral API. For this task,
        # we don't make real calls; tests may mock this method.
        return LLMResult(
            intent="unknown",
            extracted_fields={},
            missing_information=[],
        )
