"""Structured-output validation for LLM provider results.

Per the master context (section 25), the backend must never trust free-form
LLM output blindly. This module is the single gate between a provider's raw
payload and the ``LLMResult`` the AE workflow consumes:

- only the fields the workflow maps are accepted (unknown keys are dropped),
- each field is type-checked; enum-valued fields are checked against the
  engine's allowed domains (``ALLOWED_FIELD_VALUES``),
- ``dob_age_conflict`` must be a ``{"dob": str, "age": str}`` dict,
- the intent vocabulary is constrained to what the workflow understands.

Invalid individual fields are dropped and reported; only a wholly unusable
payload raises ``LLMValidationError`` so a composite provider can fall back.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.workflow.ae_engine import ALLOWED_FIELD_VALUES

from .base import LLMResult, LLMValidationError

# The extractable AE fields, mirroring the workflow's mapping in
# app.workflow.service.conversational._field_map_from_llm. Types follow
# AEStateEngine.update() validation.
STRING_FIELDS: frozenset[str] = frozenset(
    {
        "product",
        "adverse_event",
        "onset_date",
        "patient_identifier",
        "patient_age",
        "reporter_name",
        "contact_phone",
        "contact_email",
        "dose_and_use",
        "prescribing_physician",
    }
)

ENUM_FIELDS: dict[str, frozenset[str]] = {
    key: ALLOWED_FIELD_VALUES[key]
    for key in (
        "seriousness",
        "outcome",
        "reporter_role",
        "preferred_contact_method",
        "medical_care",
    )
}

BOOL_FIELDS: frozenset[str] = frozenset({"consent_to_contact"})
LIST_FIELDS: frozenset[str] = frozenset({"concomitant_medications"})

ALLOWED_INTENTS: frozenset[str] = frozenset({"unknown", "ae_report", "product_info"})

_DOB_AGE_CONFLICT_KEYS = ("dob", "age")


def sanitize_intent(value: Any) -> str:
    """Constrain an intent to the workflow's vocabulary (else ``unknown``)."""
    if isinstance(value, str) and value in ALLOWED_INTENTS:
        return value
    return "unknown"


def _valid_dob_age_conflict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(value.get(key), str) and value.get(key).strip()
        for key in _DOB_AGE_CONFLICT_KEYS
    )


def sanitize_extracted_fields(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate one extracted-fields mapping.

    Returns ``(clean_fields, dropped_field_names)``. Invalid individual
    fields (wrong type, out-of-domain enum, unknown key) are dropped and
    reported rather than raised. ``None`` is treated as an empty mapping;
    any other non-mapping raises ``LLMValidationError``.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        raise LLMValidationError(
            f"extracted_fields must be a mapping, got {type(raw).__name__}."
        )

    clean: dict[str, Any] = {}
    dropped: list[str] = []

    for key, value in raw.items():
        if value is None:
            continue
        if key == "dob_age_conflict":
            if _valid_dob_age_conflict(value):
                clean[key] = dict(value)
            else:
                dropped.append(key)
            continue
        if key in ENUM_FIELDS:
            if isinstance(value, str) and value in ENUM_FIELDS[key]:
                clean[key] = value
            else:
                dropped.append(key)
            continue
        if key in BOOL_FIELDS:
            if isinstance(value, bool):
                clean[key] = value
            else:
                dropped.append(key)
            continue
        if key in LIST_FIELDS:
            if isinstance(value, list) and all(
                isinstance(item, str) and item.strip() for item in value
            ):
                clean[key] = list(value)
            else:
                dropped.append(key)
            continue
        if key in STRING_FIELDS:
            if isinstance(value, str) and value.strip():
                clean[key] = value
            else:
                dropped.append(key)
            continue
        dropped.append(key)

    return clean, dropped


def validated_llm_result(payload: Mapping[str, Any]) -> LLMResult:
    """Convert a raw provider payload into a validated ``LLMResult``.

    Raises ``LLMValidationError`` when the payload is not a usable mapping.
    Invalid individual fields are dropped; the intent is constrained to the
    workflow vocabulary.
    """
    if not isinstance(payload, Mapping):
        raise LLMValidationError(
            f"LLM payload must be a mapping, got {type(payload).__name__}."
        )

    fields, _dropped = sanitize_extracted_fields(payload.get("extracted_fields"))

    events = payload.get("events")
    clean_events = (
        [item for item in events if isinstance(item, str) and item.strip()]
        if isinstance(events, list)
        else []
    )

    return LLMResult(
        intent=sanitize_intent(payload.get("intent")),
        extracted_fields=fields,
        events=clean_events,
        raw=dict(payload),
    )


__all__ = [
    "ALLOWED_INTENTS",
    "BOOL_FIELDS",
    "ENUM_FIELDS",
    "LIST_FIELDS",
    "STRING_FIELDS",
    "sanitize_extracted_fields",
    "sanitize_intent",
    "validated_llm_result",
]
