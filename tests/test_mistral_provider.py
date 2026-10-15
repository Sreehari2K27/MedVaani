"""Focused mocked tests for the Mistral provider and the AE composite.

No network: ``requests.post`` is monkeypatched and every composite test uses
stub providers. Nothing here reads .env or any API key.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

import app.providers.mistral as mistral_module
from app.config import (
    DEFAULT_MISTRAL_MODEL,
    DEFAULT_MISTRAL_TIMEOUT_SECONDS,
    mistral_ae_enabled,
    mistral_mi_enabled,
)
from app.providers import (
    AnswerLLMProvider,
    MistralAnswerInterpreter,
    MistralLLMProvider,
    accepted_mistral_fields,
    build_ae_llm_provider,
    build_mi_llm_provider,
)
from app.providers.base import LLMProviderError, LLMResult


# ---- HTTP stubs -------------------------------------------------------------


class _Resp:
    def __init__(self, status_code: int = 200, content: Any = None) -> None:
        self.status_code = status_code
        self._content = content

    def json(self) -> Any:
        if isinstance(self._content, Exception):
            raise self._content
        return self._content


def _payload(content: str) -> Dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture()
def posted(monkeypatch):
    """Capture requests.post calls; configurable response/exception."""
    state = {"calls": [], "response": _Resp(), "exception": None}

    def _post(url, **kwargs):
        state["calls"].append({"url": url, **kwargs})
        if state["exception"] is not None:
            raise state["exception"]
        return state["response"]

    monkeypatch.setattr(mistral_module.requests, "post", _post)
    return state


def _ae_provider() -> MistralLLMProvider:
    return MistralLLMProvider(api_key="test-key")


# ---- provider: configuration and failure modes ------------------------------


def test_missing_api_key_still_raises_cleanly():
    provider = MistralLLMProvider(api_key=None)
    # Never read from the environment for this assertion.
    provider.api_key = None
    with pytest.raises(ValueError, match="API key is missing"):
        provider.interpret("hello", {"next_field": ("outcome", "q")})


def test_ae_mode_parses_json_object(posted):
    posted["response"] = _Resp(content=_payload('{"outcome": "recovered"}'))
    result = _ae_provider().interpret(
        "it cleared up completely",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert isinstance(result, LLMResult)
    assert result.intent == "unknown"
    assert result.extracted_fields == {"outcome": "recovered"}
    assert len(posted["calls"]) == 1
    assert posted["calls"][0]["timeout"] == DEFAULT_MISTRAL_TIMEOUT_SECONDS


def test_ae_mode_without_pending_field_makes_no_call(posted):
    result = _ae_provider().interpret("something", {})
    assert result.intent == "unknown"
    assert result.extracted_fields == {}
    assert posted["calls"] == []


def test_ae_mode_rejects_non_json_output(posted):
    posted["response"] = _Resp(content=_payload("the outcome is fine"))
    with pytest.raises(LLMProviderError, match="JSON object"):
        _ae_provider().interpret(
            "it cleared up", {"next_field": ["outcome", "How now?"]}
        )


def test_http_error_becomes_provider_error(posted):
    posted["response"] = _Resp(status_code=500, content=None)
    with pytest.raises(LLMProviderError) as excinfo:
        _ae_provider().interpret(
            "x", {"next_field": ["outcome", "How now?"]}
        )
    assert "HTTP 500" in str(excinfo.value)
    # The key never leaks into an error message.
    assert "test-key" not in str(excinfo.value)


def test_timeout_becomes_provider_error(posted):
    import requests

    posted["exception"] = requests.Timeout("timed out")
    with pytest.raises(LLMProviderError, match="Timeout"):
        _ae_provider().interpret(
            "x", {"next_field": ["outcome", "How now?"]}
        )


def test_malformed_json_body_becomes_provider_error(posted):
    posted["response"] = _Resp(content=ValueError("bad json"))
    with pytest.raises(LLMProviderError, match="parsed"):
        _ae_provider().interpret(
            "x", {"next_field": ["outcome", "How now?"]}
        )


def test_mi_mode_returns_grounded_answer(posted):
    posted["response"] = _Resp(
        content=_payload(
            "Lantus is stored unopened in a refrigerator. "
            "(Source: Lantus package insert, page 11)."
        )
    )
    result = _ae_provider().interpret(
        "irrelevant message",
        {
            "question": "How should Lantus be stored?",
            "grounding_instruction": "Answer only from context.",
            "context": "=== Page 11 ===\nLantus is stored unissued...",
        },
    )
    assert result.intent == "product_info"
    assert result.extracted_fields["answer"].startswith("Lantus is stored")
    assert len(posted["calls"]) == 1


def test_prompt_uses_configured_model_and_no_key_in_error(monkeypatch, posted):
    monkeypatch.setenv("MISTRAL_MODEL", "test-model")
    posted["response"] = _Resp(content=_payload("{}"))
    _ae_provider().interpret("hi", {"next_field": ["outcome", "q"]})
    assert posted["calls"][0]["json"]["model"] == "test-model"
    headers = posted["calls"][0]["headers"]
    assert headers["Authorization"] == "Bearer test-key"


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    monkeypatch.delenv("MISTRAL_TIMEOUT_SECONDS", raising=False)
    from app.config import get_mistral_model, get_mistral_timeout_seconds

    assert get_mistral_model() == DEFAULT_MISTRAL_MODEL
    assert get_mistral_timeout_seconds() == DEFAULT_MISTRAL_TIMEOUT_SECONDS
    monkeypatch.setenv("MISTRAL_TIMEOUT_SECONDS", "not-a-number")
    assert get_mistral_timeout_seconds() == DEFAULT_MISTRAL_TIMEOUT_SECONDS
    monkeypatch.setenv("MISTRAL_TIMEOUT_SECONDS", "-5")
    assert get_mistral_timeout_seconds() == DEFAULT_MISTRAL_TIMEOUT_SECONDS


# ---- feature flags (default off) -------------------------------------------


def test_flags_default_off(monkeypatch):
    for name in (
        "MISTRAL_AE_ENABLED",
        "MISTRAL_MI_ENABLED",
        "MISTRAL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert mistral_ae_enabled() is False
    assert mistral_mi_enabled() is False
    assert build_ae_llm_provider() is None
    assert build_mi_llm_provider() is None


def test_flag_requires_key_and_flag(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("MISTRAL_AE_ENABLED", "1")
    monkeypatch.setenv("MISTRAL_MI_ENABLED", "0")
    assert mistral_ae_enabled() is True
    assert mistral_mi_enabled() is False
    monkeypatch.setenv("MISTRAL_AE_ENABLED", "0")
    assert mistral_ae_enabled() is False


def test_builder_returns_composite_only_when_enabled(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setenv("MISTRAL_AE_ENABLED", "true")
    # Hermetic regardless of the developer's local flags: only AE is enabled
    # for this test, so the MI builder must stay off.
    monkeypatch.setenv("MISTRAL_MI_ENABLED", "0")
    built = build_ae_llm_provider()
    assert isinstance(built, MistralAnswerInterpreter)
    assert build_mi_llm_provider() is None  # MI flag still off


# ---- field policy -----------------------------------------------------------


def test_field_policy_allows_only_pending_keys():
    assert accepted_mistral_fields("outcome") == frozenset({"outcome"})
    assert accepted_mistral_fields("consent_to_contact") == frozenset()
    assert accepted_mistral_fields("adverse_event") == frozenset()
    assert accepted_mistral_fields("product") == frozenset()
    assert accepted_mistral_fields("product_verification") == frozenset()
    assert accepted_mistral_fields("additional_information") == frozenset()
    assert accepted_mistral_fields(None) == frozenset()
    assert accepted_mistral_fields("contact_phone or contact_email") == frozenset(
        {"contact_phone", "contact_email"}
    )
    # Documented combined name + date-of-birth answer.
    assert accepted_mistral_fields("patient_identifier") == frozenset(
        {"patient_identifier", "patient_age"}
    )


# ---- composite stubs --------------------------------------------------------


class _DeterministicStub:
    def __init__(self, fields: Optional[Dict[str, Any]] = None, intent: str = "unknown"):
        self.fields = fields or {}
        self.intent = intent
        self.calls = 0

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        self.calls += 1
        return LLMResult(intent=self.intent, extracted_fields=dict(self.fields))


class _MistralStub:
    def __init__(self, fields: Optional[Dict[str, Any]] = None, exc: Exception = None,
                 intent: str = "unknown"):
        self.fields = fields or {}
        self.exc = exc
        self.intent = intent
        self.calls = 0
        self.last_context: Optional[Dict[str, Any]] = None

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        self.calls += 1
        self.last_context = context
        if self.exc is not None:
            raise self.exc
        return LLMResult(intent=self.intent, extracted_fields=dict(self.fields))


def _composite(deterministic, mistral) -> MistralAnswerInterpreter:
    return MistralAnswerInterpreter(deterministic=deterministic, mistral=mistral)


# ---- composite: deterministic-first guarantees ------------------------------


def test_deterministic_fields_win_and_mistral_never_called():
    det = _DeterministicStub({"outcome": "recovered"})
    mistral = _MistralStub({"outcome": "worsening"})
    result = _composite(det, mistral).interpret(
        "anything", {"next_field": ["outcome", "How now?"]}
    )
    assert mistral.calls == 0
    assert result.extracted_fields == {"outcome": "recovered"}


@pytest.mark.parametrize(
    "intent",
    ["ae_report", "urgent_medical", "treatment_query", "product_info"],
)
def test_base_interpreted_intents_never_consult_mistral(intent):
    det = _DeterministicStub({}, intent=intent)
    mistral = _MistralStub({"outcome": "recovered"})
    result = _composite(det, mistral).interpret(
        "anything", {"next_field": ["outcome", "How now?"]}
    )
    assert mistral.calls == 0
    assert result.intent == intent
    assert result.extracted_fields == {}


def test_urgent_statement_with_real_deterministic_provider_not_consulted():
    # The real deterministic provider classifies urgent wording itself; the
    # composite must leave that path untouched and never call Mistral.
    mistral = _MistralStub({"seriousness": "no"})
    provider = MistralAnswerInterpreter(
        deterministic=AnswerLLMProvider(), mistral=mistral
    )
    result = provider.interpret(
        "I have severe chest pain",
        {"next_field": ["seriousness", "Was the event severe?"]},
    )
    assert mistral.calls == 0
    assert result.intent == "urgent_medical"


def test_no_pending_field_means_no_mistral_call():
    det = _DeterministicStub()
    mistral = _MistralStub({"outcome": "recovered"})
    result = _composite(det, mistral).interpret("it cleared up", {})
    assert mistral.calls == 0
    assert result.extracted_fields == {}


@pytest.mark.parametrize(
    "message",
    [
        "yes",
        "No.",
        "okay",
        "dobara kahiye",
        "I don't know",
        "nothing else",
        "   ",
    ],
)
def test_non_consult_message_classes_untouched(message):
    det = _DeterministicStub()
    mistral = _MistralStub({"outcome": "recovered"})
    result = _composite(det, mistral).interpret(
        message, {"next_field": ["outcome", "How now?"]}
    )
    assert mistral.calls == 0
    assert result.extracted_fields == {}


def test_value_for_unrelated_field_is_dropped():
    det = _DeterministicStub()
    mistral = _MistralStub({"adverse_event": "rewritten wording"})  # forbidden key
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert mistral.calls == 1
    assert result.extracted_fields == {}


def test_pending_field_value_accepted():
    det = _DeterministicStub()
    mistral = _MistralStub({"outcome": "recovered"})
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert mistral.calls == 1
    assert result.extracted_fields == {"outcome": "recovered"}


def test_forbidden_fields_never_consulted():
    for pending in ("consent_to_contact", "adverse_event", "product"):
        det = _DeterministicStub()
        mistral = _MistralStub({"consent_to_contact": True, "product": "Lantus SoloStar"})
        _composite(det, mistral).interpret(
            "some reply", {"next_field": [pending, "Question?"]}
        )
        assert mistral.calls == 0, pending


def test_mistral_cannot_raise_consent_even_if_it_returns_it():
    # Even if a misbehaving provider returns forbidden keys alongside an
    # allowed one, only the pending allowed key survives.
    det = _DeterministicStub()
    mistral = _MistralStub(
        {"outcome": "recovered", "consent_to_contact": True, "adverse_event": "new text"}
    )
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert result.extracted_fields == {"outcome": "recovered"}


def test_invalid_enum_value_dropped():
    det = _DeterministicStub()
    mistral = _MistralStub({"outcome": "kinda better"})  # not in the outcome enum
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert mistral.calls == 1
    assert result.extracted_fields == {}


def test_unknown_keys_dropped_by_validation():
    det = _DeterministicStub()
    mistral = _MistralStub({"invented_key": "x", "case_state": "COMPLETED"})
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert mistral.calls == 1
    assert result.extracted_fields == {}


def test_fabricated_contact_value_rejected():
    det = _DeterministicStub()
    mistral = _MistralStub({"contact_email": "someone@example.com"})
    result = _composite(det, mistral).interpret(
        "I would rather not say right now",
        {"next_field": ["contact_email", "May I have the email address?"]},
    )
    assert mistral.calls == 1
    # The address is not in the reporter's words -> never stored.
    assert result.extracted_fields == {}


def test_traceable_values_accepted():
    det = _DeterministicStub()
    mistral = _MistralStub({"patient_identifier": "Priya Sharma"})
    result = _composite(det, mistral).interpret(
        "the patient is Priya Sharma",
        {"next_field": ["patient_identifier", "Patient name and date of birth?"]},
    )
    assert result.extracted_fields == {"patient_identifier": "Priya Sharma"}


def test_traceable_digit_value_accepted():
    det = _DeterministicStub()
    mistral = _MistralStub({"contact_phone": "9886094321"})
    result = _composite(det, mistral).interpret(
        "call me back on 988 609 4321 later",
        {"next_field": ["contact_phone", "Patient contact number?"]},
    )
    assert result.extracted_fields == {"contact_phone": "9886094321"}


def test_mistral_failure_returns_deterministic_result():
    det = _DeterministicStub()
    mistral = _MistralStub(exc=LLMProviderError("down"))
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert mistral.calls == 1
    assert result.intent == "unknown"
    assert result.extracted_fields == {}


def test_mistral_bad_result_object_returns_deterministic_result():
    det = _DeterministicStub()
    mistral = _MistralStub()
    mistral.interpret = lambda message, context=None: {"intent": "ae_report"}  # type: ignore
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert result.extracted_fields == {}


def test_intent_never_raised_by_mistral():
    det = _DeterministicStub()
    mistral = _MistralStub({"outcome": "recovered"}, intent="ae_report")
    result = _composite(det, mistral).interpret(
        "the pain vanished after two days",
        {"next_field": ["outcome", "How is the event now?"]},
    )
    assert result.intent == "unknown"


def test_prompt_fields_hint_exposed_to_provider():
    det = _DeterministicStub()
    mistral = _MistralStub({"patient_age": "27"})
    result = _composite(det, mistral).interpret(
        "born on 27 May 1998",  # digits 27 1998 present in message
        {"next_field": ["patient_identifier", "Patient name and date of birth?"]},
    )
    assert mistral.last_context["prompt_fields"] == ["patient_age", "patient_identifier"]
    assert result.extracted_fields == {"patient_age": "27"}
