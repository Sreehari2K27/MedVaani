"""Focused tests for the composite fallback LLM provider."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from app.providers.base import LLMResult
from app.providers.fallback import FallbackLLMProvider


class _PrimaryOk:
    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        self.calls += 1
        return LLMResult(
            intent="ae_report", extracted_fields={"product": "Lantus SoloStar"}
        )


class _PrimaryRaises:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        self.calls += 1
        raise self.exc


class _PrimaryInvalidObject:
    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> Any:
        return {"intent": "ae_report"}  # dict, not LLMResult


class _PrimaryInvalidShape:
    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        return LLMResult(intent="ae_report", extracted_fields="not-a-dict")  # type: ignore[arg-type]


class _Fallback:
    def __init__(self, exc: Exception | None = None) -> None:
        self.calls = 0
        self.exc = exc

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return LLMResult(intent="unknown")


def test_healthy_primary_is_used_and_fallback_untouched():
    primary, fallback = _PrimaryOk(), _Fallback()
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    result = provider.interpret("I took Lantus and have a rash.")

    assert result.intent == "ae_report"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert provider.last_error is None


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("boom"), ValueError("bad input"), TimeoutError("timed out")],
)
def test_failing_primary_falls_back_and_records_error(exc):
    primary, fallback = _PrimaryRaises(exc), _Fallback()
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    result = provider.interpret("anything")

    assert result.intent == "unknown"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert provider.last_error is exc


def test_invalid_primary_object_falls_back():
    fallback = _Fallback()
    provider = FallbackLLMProvider(primary=_PrimaryInvalidObject(), fallback=fallback)

    result = provider.interpret("anything")

    assert result.intent == "unknown"
    assert fallback.calls == 1
    assert isinstance(provider.last_error, ValueError)


def test_invalid_primary_result_shape_falls_back():
    fallback = _Fallback()
    provider = FallbackLLMProvider(primary=_PrimaryInvalidShape(), fallback=fallback)

    result = provider.interpret("anything")

    assert result.intent == "unknown"
    assert fallback.calls == 1
    assert isinstance(provider.last_error, ValueError)


def test_primary_none_delegates_directly_to_fallback():
    fallback = _Fallback()
    provider = FallbackLLMProvider(primary=None, fallback=fallback)

    result = provider.interpret("anything")

    assert result.intent == "unknown"
    assert fallback.calls == 1
    assert provider.last_error is None


def test_fallback_failure_propagates():
    failing_fallback = _Fallback(exc=RuntimeError("fallback down"))
    provider = FallbackLLMProvider(primary=_PrimaryRaises(ValueError("x")), fallback=failing_fallback)

    with pytest.raises(RuntimeError, match="fallback down"):
        provider.interpret("anything")
    assert failing_fallback.calls == 1


def test_fallback_failure_propagates_without_primary():
    failing_fallback = _Fallback(exc=RuntimeError("fallback down"))
    provider = FallbackLLMProvider(primary=None, fallback=failing_fallback)

    with pytest.raises(RuntimeError, match="fallback down"):
        provider.interpret("anything")


def test_fallback_is_required():
    with pytest.raises(ValueError, match="fallback"):
        FallbackLLMProvider(primary=None, fallback=None)
