"""Deterministic-first AE interpreter with optional Mistral assistance.

The deterministic interpreter (``AnswerLLMProvider``) is always authoritative:
it runs first and its result is returned unchanged whenever it understood the
turn. Mistral is consulted only when ALL of the following hold:

- the deterministic interpreter returned ``intent="unknown"`` with no
  extracted fields (it could not confidently interpret the reply),
- a pending interview question exists in the turn context,
- the reply is not a bare yes/no, a repeat/confusion request, an explicit
  "don't know", or a plain non-answer (all handled deterministically),
- the pending question maps to at least one Mistral-acceptable field.

Mistral output is advisory and double-gated before anything reaches the
workflow: every field passes the existing ``sanitize_extracted_fields``
validation, must belong to the pending question's field allow-list, and
free-text values must be traceable to the reporter's own words (nothing is
invented). Consent decisions, adverse-event rewrites and product answers are
never accepted from Mistral, the intent is never raised above ``"unknown"``,
and any provider failure returns the deterministic result untouched.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .answer import (
    AnswerLLMProvider,
    _CLARIFY_REPLY,
    _NON_ANSWER,
    _SIMPLE_YESNO,
    _UNKNOWN_ANSWER,
)
from .base import LLMProvider, LLMResult
from .validation import ENUM_FIELDS, sanitize_extracted_fields

# Fields the AE workflow maps from an LLM result (mirrors
# app.workflow.service.conversational._field_map_from_llm). Only these can
# ever be applied, so unknown keys are dead on arrival.
_MAPPABLE_FIELDS = frozenset(
    {
        "product",
        "adverse_event",
        "onset_date",
        "seriousness",
        "outcome",
        "patient_identifier",
        "patient_age",
        "reporter_role",
        "reporter_name",
        "reporter_contact_phone",
        "contact_phone",
        "contact_email",
        "preferred_contact_method",
        "consent_to_contact",
        "medical_care",
        "dose_and_use",
        "prescribing_physician",
        "concomitant_medications",
    }
)

# Never accepted from Mistral, regardless of the pending question:
# - consent must come from the deterministic consent path (never inferred),
# - the reporter's adverse-event wording is never rewritten by a model,
# - the product is only ever captured/verified deterministically (the
#   medication question and Step 4 verification both re-ask on ambiguity).
_MISTRAL_FORBIDDEN_FIELDS = frozenset(
    {"consent_to_contact", "adverse_event", "product"}
)

# Documented multi-field combinations: while the patient name/DOB question is
# pending, one reply may legitimately carry the name AND the date of birth;
# the combined "phone or email" question may yield either key.
_COMBINATION_EXTRA_KEYS = {
    "patient_identifier": frozenset({"patient_age"}),
}


def accepted_mistral_fields(pending_key: Any) -> frozenset:
    """Mistral-acceptable field keys for the pending question, or empty set."""
    keys = {part.strip() for part in str(pending_key or "").split(" or ") if part.strip()}
    if not keys:
        return frozenset()
    for key in list(keys):
        keys |= _COMBINATION_EXTRA_KEYS.get(key, frozenset())
    return (frozenset(keys) & _MAPPABLE_FIELDS) - _MISTRAL_FORBIDDEN_FIELDS


def _value_traceable(field_key: str, value: Any, message: str) -> bool:
    """Whether a free-text Mistral value is grounded in the reporter's words.

    Enum-valued fields are pure interpretations over a constrained vocabulary
    and need no traceability check. Everything else must be recoverable from
    the reply itself (verbatim substring, its digits, or all of its tokens),
    so a model can never invent a name, contact detail, age or date.
    """
    if field_key in ENUM_FIELDS:
        return True
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    text = message or ""
    lowered = text.lower()
    if candidate.lower() in lowered:
        return True
    digits = "".join(re.findall(r"\d+", candidate))
    if digits and digits in "".join(re.findall(r"\d+", text)):
        return True
    tokens = re.findall(r"[a-z0-9]{3,}", candidate.lower())
    return bool(tokens) and all(token in lowered for token in tokens)


class MistralAnswerInterpreter:
    """Deterministic-first ``interpret`` with gated Mistral assistance."""

    def __init__(
        self,
        deterministic: Optional[LLMProvider] = None,
        mistral: Optional[LLMProvider] = None,
    ) -> None:
        if deterministic is None:
            deterministic = AnswerLLMProvider()
        if mistral is None:
            from .mistral import MistralLLMProvider

            mistral = MistralLLMProvider()
        self._deterministic = deterministic
        self._mistral = mistral

    # -- gate helpers ------------------------------------------------------

    @staticmethod
    def _pending_keys(message: str, base: LLMResult, context: Dict[str, Any]) -> frozenset:
        """Field keys Mistral may answer for this turn, or empty when barred."""
        if not (message or "").strip():
            return frozenset()
        # A turn the deterministic interpreter already understood is never
        # re-interpreted (preserves every currently working response path).
        if base.intent != "unknown" or base.extracted_fields:
            return frozenset()
        stripped = message.strip()
        if (
            _CLARIFY_REPLY.search(stripped)
            or _UNKNOWN_ANSWER.search(stripped)
            or _NON_ANSWER.match(stripped)
            # Bare yes/no replies are already handled by the deterministic
            # pending-field parser and by every confirmation/verification step;
            # Mistral never sees them.
            or _SIMPLE_YESNO.match(stripped)
        ):
            return frozenset()
        pending = context.get("next_field")
        if not isinstance(pending, (list, tuple)) or len(pending) < 2:
            return frozenset()
        return accepted_mistral_fields(pending[0])

    # -- public surface ----------------------------------------------------

    def interpret(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> LLMResult:
        context = context or {}
        base = self._deterministic.interpret(message, context)

        accepted = self._pending_keys(message, base, context)
        if not accepted:
            return base

        try:
            assisted_context = dict(context)
            assisted_context["prompt_fields"] = sorted(accepted)
            assisted = self._mistral.interpret(message, assisted_context)
        except Exception:
            # Any provider failure (key, timeout, network, parse) degrades to
            # today's deterministic behaviour for this turn.
            return base
        if not isinstance(assisted, LLMResult):
            return base

        cleaned, _dropped = sanitize_extracted_fields(assisted.extracted_fields)
        allowed: Dict[str, Any] = {
            key: value
            for key, value in cleaned.items()
            if key in accepted and _value_traceable(key, value, message)
        }
        if not allowed:
            return base

        merged = dict(base.extracted_fields or {})
        merged.update(allowed)
        base.extracted_fields = merged
        # Intent stays exactly as the deterministic interpreter returned it
        # ("unknown" for answer turns); state transitions remain the workflow's.
        return base


__all__ = [
    "MistralAnswerInterpreter",
    "accepted_mistral_fields",
]
