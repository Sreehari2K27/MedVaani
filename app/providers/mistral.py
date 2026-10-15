"""Mistral LLM provider (one shared client for AE and MI/RAG).

Two prompt modes behind the single ``interpret`` surface:

- AE mode (``context["next_field"]`` present): interprets one pending
  question's reply into a JSON object of extracted fields. The result is
  advisory only - intent stays ``"unknown"`` and the deterministic composite
  (``mistral_composite.MistralAnswerInterpreter``) decides what, if anything,
  is accepted.
- MI/RAG mode (``context["grounding_instruction"]`` present): answers using
  only the retrieved reference content and returns ``intent="product_info"``
  with the answer in ``extracted_fields["answer"]`` for the existing grounding
  gate in ``MIAnswerService``.

No logging of prompts, responses, question text or keys - request diagnostics
are metadata only (model, HTTP status, error type, elapsed ms). No network
calls in tests - callers mock ``requests.post`` or this provider's
``interpret``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from app.config import (
    get_mistral_api_key,
    get_mistral_model,
    get_mistral_timeout_seconds,
    mistral_ae_enabled,
    mistral_mi_enabled,
)

from .base import LLMProviderError, LLMResult
from .validation import ENUM_FIELDS

_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Request diagnostics only: model, HTTP status, error type and elapsed time.
# Never prompts, responses, question/answer text, or credentials.
logger = logging.getLogger(__name__)

_AE_SYSTEM = (
    "You interpret exactly one reply in a structured adverse-event intake "
    "interview. The interview has one pending question; the reply text is the "
    "answer to THAT question and to nothing else. "
    "Reply with ONLY a JSON object of the form {{\"<field_key>\": <value>}} and "
    "no prose, no markdown. "
    "Rules: "
    "(1) Use a field key exactly from this list: {fields}. "
    "(2) If the reply does not clearly answer the pending question, asks to "
    "repeat, expresses uncertainty, or answers a different question, reply "
    "with exactly {{}}. "
    "(3) Never invent, guess, or complete any information that is not plainly "
    "stated in the reply. "
    "(4) Never decide consent, never rewrite the reporter's own description of "
    "the event, and never confirm or change the medication. "
    "{allowed}"
)

_MI_SYSTEM = (
    "You answer questions about the medicine Lantus SoloStar using ONLY the "
    "supplied reference content. Do not use outside medical knowledge. "
    "Do not give personalized dosing, diagnosis, treatment recommendations, or "
    "medication-change advice. "
    "Keep the answer concise and include an inline citation of the form "
    "(Source: Lantus package insert, page <N>) drawn from the supplied "
    "content. If the supplied content does not support an answer, reply with "
    "exactly: Sufficient information is not available in the provided product "
    "reference."
)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``text`` as one JSON object; None when it is not one."""
    work = _FENCE.sub("", (text or "").strip()).strip()
    if not work:
        return None
    try:
        parsed = json.loads(work)
    except json.JSONDecodeError:
        start, end = work.find("{"), work.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(work[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _allowed_values_hint(field_keys: List[str]) -> str:
    hints = []
    for key in field_keys:
        values = ENUM_FIELDS.get(key)
        if values:
            hints.append(f'For "{key}" use only one of: {", ".join(sorted(values))}.')
    return " ".join(hints)


class MistralLLMProvider:
    """Mistral provider - isolated; never called unless explicitly enabled."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key if api_key is not None else get_mistral_api_key()

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        """Interpret message. Raises if API key missing (clean handling)."""
        if not self.api_key:
            raise ValueError("Mistral API key is missing. Set MISTRAL_API_KEY environment variable.")
        context = context or {}
        if "grounding_instruction" in context and "question" in context:
            return self._interpret_grounded(message, context)
        return self._interpret_pending(message, context)

    # -- internals ---------------------------------------------------------

    def _chat(self, system: str, user: str) -> str:
        """One chat-completions request; raises LLMProviderError on failure."""
        model = get_mistral_model()
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = requests.post(
                _API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=get_mistral_timeout_seconds(),
            )
        except requests.RequestException as exc:
            # Never include the request body or key in the log or error.
            logger.warning(
                "mistral_request_failed transport=%s duration_ms=%d model=%s",
                type(exc).__name__,
                elapsed_ms(),
                model,
            )
            raise LLMProviderError(
                f"Mistral request failed ({type(exc).__name__})."
            ) from exc
        if response.status_code != 200:
            logger.warning(
                "mistral_request_failed status=%d duration_ms=%d model=%s",
                response.status_code,
                elapsed_ms(),
                model,
            )
            raise LLMProviderError(
                f"Mistral request failed with HTTP {response.status_code}."
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "mistral_response_unparsable duration_ms=%d model=%s",
                elapsed_ms(),
                model,
            )
            raise LLMProviderError("Mistral response could not be parsed.") from exc
        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "mistral_response_empty duration_ms=%d model=%s",
                elapsed_ms(),
                model,
            )
            raise LLMProviderError("Mistral returned an empty response.")
        logger.debug(
            "mistral_request_ok status=200 duration_ms=%d model=%s",
            elapsed_ms(),
            model,
        )
        return content.strip()

    def _interpret_pending(self, message: str, context: Dict[str, Any]) -> LLMResult:
        """AE mode: extract fields for the single pending question only."""
        pending = context.get("next_field")
        if (
            not isinstance(pending, (list, tuple))
            or len(pending) < 2
            or not str(pending[0]).strip()
        ):
            # No pending question: nothing to interpret against; never guess.
            return LLMResult(intent="unknown", extracted_fields={})

        field_keys = [
            key.strip()
            for key in str(pending[0]).split(" or ")
            if key.strip()
        ]
        hint = context.get("prompt_fields")
        if isinstance(hint, (list, tuple)) and hint:
            field_keys = [str(key) for key in hint]
        question = str(pending[1])

        system = _AE_SYSTEM.format(
            fields=", ".join(field_keys),
            allowed=_allowed_values_hint(field_keys),
        )
        user = (
            f"Pending question: {question}\n"
            f"Field key(s): {', '.join(field_keys)}\n"
            f"Reply: {message}"
        )
        content = self._chat(system, user)
        parsed = _parse_json_object(content)
        if parsed is None:
            raise LLMProviderError("Mistral AE output was not a JSON object.")
        # Strict shape: only simple field values leave this provider.
        fields: Dict[str, Any] = {}
        for key, value in parsed.items():
            if isinstance(value, (str, bool, int, float)) or value is None:
                fields[str(key)] = value
        return LLMResult(intent="unknown", extracted_fields=fields)

    def _interpret_grounded(self, message: str, context: Dict[str, Any]) -> LLMResult:
        """MI mode: grounded answer over the retrieved reference content."""
        question = str(context.get("question") or "")
        grounding = str(context.get("grounding_instruction") or "")
        reference = str(context.get("context") or "")
        system = _MI_SYSTEM
        user = f"{grounding}\n\nQuestion: {question}\n\nContext: {reference}"
        content = self._chat(system, user)
        return LLMResult(
            intent="product_info",
            extracted_fields={"answer": content},
        )


def build_ae_llm_provider(deterministic=None):
    """Deterministic-first AE provider with Mistral assistance, or None.

    Returns ``None`` while ``MISTRAL_AE_ENABLED``/``MISTRAL_API_KEY`` say the
    feature is off, so every caller keeps today's deterministic wiring.
    """
    if not mistral_ae_enabled():
        return None
    from .mistral_composite import MistralAnswerInterpreter

    return MistralAnswerInterpreter(deterministic=deterministic, mistral=MistralLLMProvider())


def build_mi_llm_provider():
    """Mistral provider for MI/RAG, or None while the MI feature flag is off."""
    if not mistral_mi_enabled():
        return None
    return MistralLLMProvider()


__all__ = [
    "MistralLLMProvider",
    "build_ae_llm_provider",
    "build_mi_llm_provider",
]
