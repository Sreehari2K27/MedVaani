"""Composite LLM provider with a deterministic fallback.

Keeps the AE conversation alive when a primary (future) LLM provider fails:
timeouts, network errors, payload-parsing/validation failures, or an invalid
result object. Any ordinary exception from the primary is treated as a
provider failure and the configured fallback answers the turn instead.

This is not a retry mechanism and adds no dependencies: one primary attempt,
then the fallback. Fallback failures propagate unchanged so real bugs are
never hidden. ``last_error`` exposes the recorded primary failure for tests
and callers. The deterministic fallback is not multilingual: on primary
failure, non-English turns degrade exactly as they do without a primary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import LLMProvider, LLMResult


def _result_shape_is_valid(result: Any) -> bool:
    if not isinstance(result, LLMResult):
        return False
    if not isinstance(result.intent, str):
        return False
    if not isinstance(result.extracted_fields, dict):
        return False
    if not isinstance(result.events, list):
        return False
    return True


class FallbackLLMProvider:
    """Try the primary provider once; on failure use the fallback provider."""

    def __init__(
        self,
        primary: Optional[LLMProvider] = None,
        fallback: Optional[LLMProvider] = None,
    ) -> None:
        if fallback is None:
            raise ValueError("FallbackLLMProvider requires a fallback provider.")
        self._primary = primary
        self._fallback = fallback
        self.last_error: Optional[Exception] = None

    def interpret(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        if self._primary is not None:
            try:
                result = self._primary.interpret(message, context)
                if not _result_shape_is_valid(result):
                    raise ValueError(
                        "Primary LLM provider returned an invalid result object."
                    )
                return result
            except Exception as exc:
                self.last_error = exc
        return self._fallback.interpret(message, context)


__all__ = ["FallbackLLMProvider"]
