"""Focused tests for structured LLM-output validation."""
from __future__ import annotations

import pytest

from app.providers.base import LLMResult, LLMValidationError
from app.providers.validation import (
    sanitize_extracted_fields,
    sanitize_intent,
    validated_llm_result,
)
from app.workflow.service.conversational import _field_map_from_llm

VALID_FIELDS = {
    "product": "Lantus SoloStar",
    "adverse_event": "rash at the injection site",
    "onset_date": "yesterday",
    "seriousness": "no",
    "outcome": "recovering",
    "patient_identifier": "Demo Patient",
    "patient_age": "45",
    "reporter_role": "self",
    "reporter_name": "Rahul Sharma",
    "contact_phone": "9876543210",
    "contact_email": "demo@example.com",
    "preferred_contact_method": "phone",
    "consent_to_contact": True,
    "medical_care": "yes",
    "dose_and_use": "once daily at bedtime",
    "prescribing_physician": "Dr. Nair",
    "concomitant_medications": ["metformin"],
}


def test_valid_fields_pass_unchanged():
    clean, dropped = sanitize_extracted_fields(VALID_FIELDS)
    assert clean == VALID_FIELDS
    assert dropped == []


def test_none_and_empty_mapping_yield_empty_result():
    assert sanitize_extracted_fields(None) == ({}, [])
    assert sanitize_extracted_fields({}) == ({}, [])


def test_extra_keys_are_dropped_and_reported():
    clean, dropped = sanitize_extracted_fields(
        {"product": "Lantus SoloStar", "hallucinated_field": "x"}
    )
    assert clean == {"product": "Lantus SoloStar"}
    assert dropped == ["hallucinated_field"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("product", 42),
        ("patient_age", 45),
        ("adverse_event", ""),
        ("consent_to_contact", "yes"),
        ("concomitant_medications", "metformin"),
        ("concomitant_medications", [1, 2]),
        ("seriousness", "definitely"),
        ("outcome", "fine"),
        ("reporter_role", "neighbor"),
        ("medical_care", "maybe"),
        ("preferred_contact_method", "fax"),
    ],
)
def test_invalid_field_values_are_dropped_and_reported(key, value):
    clean, dropped = sanitize_extracted_fields({"reporter_name": "Anchor", key: value})
    assert clean == {"reporter_name": "Anchor"}
    assert dropped == [key]


def test_none_valued_fields_are_ignored_not_dropped():
    clean, dropped = sanitize_extracted_fields({"product": None, "seriousness": "no"})
    assert clean == {"seriousness": "no"}
    assert dropped == []


def test_dob_age_conflict_valid_shape_is_kept():
    conflict = {"dob": "1990-01-01", "age": "35"}
    clean, dropped = sanitize_extracted_fields({"dob_age_conflict": conflict})
    assert clean == {"dob_age_conflict": conflict}
    assert dropped == []


@pytest.mark.parametrize(
    "value",
    [
        "1990-01-01",
        {"dob": "1990-01-01"},
        {"dob": "1990-01-01", "age": 35},
        {"dob": " ", "age": "35"},
    ],
)
def test_dob_age_conflict_invalid_shapes_are_dropped(value):
    clean, dropped = sanitize_extracted_fields({"dob_age_conflict": value})
    assert clean == {}
    assert dropped == ["dob_age_conflict"]


def test_non_mapping_extracted_fields_raises():
    with pytest.raises(LLMValidationError):
        sanitize_extracted_fields(["product"])
    with pytest.raises(LLMValidationError):
        sanitize_extracted_fields("product")


def test_invalid_envelope_payload_raises():
    with pytest.raises(LLMValidationError):
        validated_llm_result("not a mapping")  # type: ignore[arg-type]
    with pytest.raises(LLMValidationError):
        validated_llm_result(["extracted_fields"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("ae_report", "ae_report"),
        ("product_info", "product_info"),
        ("unknown", "unknown"),
        ("diagnosis", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
        (42, "unknown"),
    ],
)
def test_intent_vocabulary_is_constrained(intent, expected):
    assert sanitize_intent(intent) == expected


def test_validated_llm_result_maps_payload():
    result = validated_llm_result(
        {
            "intent": "ae_report",
            "extracted_fields": {"product": "Lantus SoloStar", "junk": 1},
            "events": ["rash", 7, ""],
        }
    )
    assert isinstance(result, LLMResult)
    assert result.intent == "ae_report"
    assert result.extracted_fields == {"product": "Lantus SoloStar"}
    assert result.events == ["rash"]
    assert result.raw is not None


def test_whitelist_parity_with_workflow_field_mapping():
    # The validator's accepted field set must exactly match the keys the
    # workflow maps from an LLMResult, so no field can be silently lost
    # (or smuggled past the workflow) when a real LLM provider arrives.
    clean, dropped = sanitize_extracted_fields(VALID_FIELDS)
    workflow_keys = set(
        _field_map_from_llm(LLMResult(extracted_fields=VALID_FIELDS))
    )
    assert set(clean) == workflow_keys
    assert dropped == []
    assert len(workflow_keys) == 17
