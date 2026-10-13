"""Focused unit tests for the deterministic AE workflow/state engine.

All tests are offline and exercise the engine with fictional/demo data only.
No LLM, Prisma, Timbre, FastAPI, or network is involved.
"""

from __future__ import annotations

import json

import pytest

from app.workflow.ae_engine import (
    AEStateEngine,
    AECaseIncompleteError,
    AECaseValidationError,
    WorkflowState,
)

REQUIRED_FIRST_FIELDS = {
    "product",
    "adverse_event",
    "patient_identifier",
    "patient_age",
    "reporter_role",
    "consent_to_contact",
    "onset_date",
    "seriousness",
    "outcome",
}


def make_engine(case_id: str = "AE-test-0001-demo") -> AEStateEngine:
    return AEStateEngine.create_case(case_id=case_id)


def interview_patient(engine: AEStateEngine) -> AEStateEngine:
    """Drive a normal interview to the completion-ready CASE_REVIEW state."""
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Headache with blurred vision (demo)")
    engine.update("patient_identifier", "PT-101d")
    engine.update("patient_age", "45")
    engine.update("reporter_role", "self")
    engine.update("contact_email", "demo.patient@example.com")
    engine.update("consent_to_contact", True)
    engine.update("onset_date", "2026-09-30")
    engine.update("seriousness", "no")
    engine.update("outcome", "recovering")
    return engine


def test_new_case_starts_in_created_state() -> None:
    engine = make_engine()
    assert engine.current_state is WorkflowState.CASE_CREATED
    assert engine.case.case_id == "AE-test-0001-demo"
    assert engine.case.priority == "normal"


def test_normal_progression_to_completion() -> None:
    engine = make_engine()

    engine.update("product", "Lantus (fictional)")
    assert engine.current_state is WorkflowState.PRODUCT_IDENTIFIED

    engine.update("adverse_event", "Headache with blurred vision (demo)")
    assert engine.current_state is WorkflowState.EVENT_IDENTIFIED

    # Reporter identity is established before patient-specific questions.
    engine.update("reporter_role", "self")
    assert engine.current_state is WorkflowState.EVENT_IDENTIFIED

    engine.update("patient_identifier", "PT-101d")
    assert engine.current_state is WorkflowState.REPORTER_IDENTIFIED

    engine.update("patient_age", "45")
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED

    # Consent is captured before any contact detail is requested.
    engine.update("consent_to_contact", True)
    assert engine.current_state is WorkflowState.CONSENT_CAPTURED

    engine.update("contact_email", "demo.patient@example.com")
    assert engine.current_state is WorkflowState.CONTACT_CAPTURED

    engine.update("onset_date", "2026-09-30")
    assert engine.current_state is WorkflowState.EVENT_TIMING_CAPTURED

    engine.update("seriousness", "no")
    assert engine.current_state is WorkflowState.SERIOUSNESS_SCREENED

    engine.update("outcome", "recovering")
    # OUTCOME_CAPTURED -> OPTIONAL_DETAILS -> CASE_REVIEW are traversed in one
    # auto-advance because nothing is required between outcome and review.
    assert engine.current_state is WorkflowState.CASE_REVIEW
    assert engine.missing_information() == []
    assert engine.next_field() is None
    assert engine.can_complete() is True

    engine.complete()
    assert engine.current_state is WorkflowState.CASE_COMPLETED
    assert engine.to_dict()["report_complete"] is True


def test_missing_required_information_at_creation() -> None:
    engine = make_engine()
    missing = set(engine.missing_information())
    assert missing == REQUIRED_FIRST_FIELDS
    field, question = engine.next_field()  # type: ignore[misc]
    assert field == "product"
    assert question


def test_only_reports_first_missing_field_as_next_question() -> None:
    engine = make_engine()
    engine.update("product", "Lantus (fictional)")
    field, _ = engine.next_field()  # type: ignore[misc]
    assert field == "adverse_event"


def test_reporter_is_the_patient() -> None:
    engine = make_engine()
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Headache (demo)")
    engine.update("patient_identifier", "PT-101d")
    engine.update("patient_age", "45")

    engine.update("reporter_role", "self")
    # Patient identity is already present, so the case advances past reporter
    # and patient identification in one recompute.
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED
    assert "reporter_name" not in engine.missing_information()
    assert engine.to_dict()["reporter"]["same_as_patient"] is True
    assert engine.case.reporter_name is None

    engine.update("consent_to_contact", True)
    engine.update("contact_email", "demo.patient@example.com")
    engine.update("onset_date", "2026-09-30")
    engine.update("seriousness", "no")
    engine.update("outcome", "recovering")
    assert engine.can_complete() is True


def test_reporter_is_someone_else_requires_name() -> None:
    engine = make_engine()
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Headache (demo)")
    engine.update("patient_identifier", "PT-101d")
    engine.update("patient_age", "45")

    engine.update("reporter_role", "someone_else")
    # The reporter is not fully identified until a name is captured, and the
    # reporter precedes patient identification, so the workflow has not yet
    # advanced past EVENT_IDENTIFIED.
    assert engine.current_state is WorkflowState.EVENT_IDENTIFIED
    assert "reporter_name" in engine.missing_information()
    assert engine.to_dict()["reporter"]["same_as_patient"] is False
    field, _ = engine.next_field()  # type: ignore[misc]
    assert field == "reporter_name"

    engine.update("reporter_name", "Dr. Demo Observer")
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED
    assert "reporter_name" not in engine.missing_information()
    assert engine.case.reporter_name == "Dr. Demo Observer"


def test_switching_reporter_to_self_clears_duplicate_name() -> None:
    engine = make_engine()
    engine.update("reporter_role", "someone_else")
    engine.update("reporter_name", "Dr. Demo Observer")
    engine.update("reporter_role", "self")
    assert engine.case.reporter_name is None
    assert "reporter_name" not in engine.missing_information()


def test_consent_tracking_is_explicit_and_required() -> None:
    engine = make_engine()
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Headache (demo)")
    engine.update("patient_identifier", "PT-101d")
    engine.update("patient_age", "45")
    engine.update("reporter_role", "self")
    engine.update("contact_email", "demo.patient@example.com")

    assert engine.case.consent_to_contact is None
    assert "consent_to_contact" in engine.missing_information()
    assert engine.can_complete() is False

    engine.update("consent_to_contact", True)
    assert engine.case.consent_to_contact is True
    assert "consent_to_contact" not in engine.missing_information()


def test_incomplete_case_cannot_complete() -> None:
    engine = make_engine()
    engine.update("product", "Lantus (fictional)")
    with pytest.raises(AECaseIncompleteError):
        engine.complete()
    assert engine.current_state is not WorkflowState.CASE_COMPLETED

    missing_outcome = interview_patient(make_engine())
    missing_outcome.case.outcome = None
    assert missing_outcome.can_complete() is False
    with pytest.raises(AECaseIncompleteError):
        missing_outcome.complete()
    assert missing_outcome.current_state is not WorkflowState.CASE_COMPLETED


def test_seriousness_supports_unknown_until_captured() -> None:
    engine = interview_patient(make_engine())
    engine.update("seriousness", "unknown")
    assert engine.case.seriousness == "unknown"
    assert "seriousness" in engine.missing_information()
    assert engine.can_complete() is False
    assert "seriousness" == engine.next_field()[0]  # type: ignore[index]

    engine.update("seriousness", "no")
    assert engine.can_complete() is True


def test_all_workflow_stages_are_supported() -> None:
    expected = [
        WorkflowState.CASE_CREATED,
        WorkflowState.PRODUCT_IDENTIFIED,
        WorkflowState.EVENT_IDENTIFIED,
        WorkflowState.PATIENT_IDENTIFIED,
        WorkflowState.REPORTER_IDENTIFIED,
        WorkflowState.CONTACT_CAPTURED,
        WorkflowState.CONSENT_CAPTURED,
        WorkflowState.EVENT_TIMING_CAPTURED,
        WorkflowState.SERIOUSNESS_SCREENED,
        WorkflowState.OUTCOME_CAPTURED,
        WorkflowState.OPTIONAL_DETAILS,
        WorkflowState.CASE_REVIEW,
        WorkflowState.CASE_COMPLETED,
        WorkflowState.HUMAN_REVIEW,
    ]
    assert list(WorkflowState) == expected


def test_urgent_transition_to_human_review() -> None:
    engine = interview_patient(make_engine())
    assert engine.can_complete() is True

    engine.mark_urgent(reason="Potential hospitalization reported by caller (demo)")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.priority == "high"
    assert engine.case.human_review_reason == (
        "Potential hospitalization reported by caller (demo)"
    )
    # Collected information is preserved.
    assert engine.case.product == "Lantus (fictional)"

    # Questioning is paused.
    assert engine.next_field() is None
    assert engine.can_complete() is False
    with pytest.raises(AECaseIncompleteError):
        engine.complete()


def test_updates_after_human_review_preserve_data_and_state() -> None:
    engine = interview_patient(make_engine())
    engine.mark_urgent(reason="demo")
    engine.update("outcome", "recovered")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.priority == "high"
    assert engine.case.outcome == "recovered"
    assert engine.case.status is WorkflowState.HUMAN_REVIEW


def test_optional_details_are_not_required() -> None:
    engine = make_engine()
    engine.update("concomitant_medications", ["Metformin (fictional)"])
    assert engine.case.concomitant_medications == ["Metformin (fictional)"]
    assert "concomitant_medications" not in engine.missing_information()


def test_serialization_is_json_ready() -> None:
    engine = interview_patient(make_engine())
    engine.update("concomitant_medications", ["Aspirin 75mg (fictional)"])
    payload = engine.to_dict()

    assert payload["case_id"] == "AE-test-0001-demo"
    assert payload["status"] == "CASE_REVIEW"
    assert payload["priority"] == "normal"
    assert payload["report_complete"] is False
    assert payload["product"] == "Lantus (fictional)"
    assert payload["patient"] == {"identifier": "PT-101d", "age": "45", "sex": None}
    assert payload["reporter"] == {
        "role": "self",
        "name": None,
        "same_as_patient": True,
    }
    assert payload["contact"]["consent_to_contact"] is True
    assert payload["seriousness"] == "no"
    assert payload["outcome"] == "recovering"
    assert payload["concomitant_medications"] == ["Aspirin 75mg (fictional)"]
    assert payload["missing_information"] == []
    assert payload["next_question"] is None
    assert payload["human_review_reason"] is None

    raw = json.dumps(payload, sort_keys=True)  # must not raise
    assert '"case_id"' in raw


def test_validation_rejects_bad_values() -> None:
    engine = make_engine()
    with pytest.raises(AECaseValidationError):
        engine.update("unknown_field", "x")
    with pytest.raises(AECaseValidationError):
        engine.update("case_id", "hijack")
    with pytest.raises(AECaseValidationError):
        engine.update("seriousness", "very")
    with pytest.raises(AECaseValidationError):
        engine.update("outcome", "miracled")
    with pytest.raises(AECaseValidationError):
        engine.update("consent_to_contact", "yes")
    with pytest.raises(AECaseValidationError):
        engine.update("consent_to_contact", None)
    with pytest.raises(AECaseValidationError):
        engine.update("reporter_role", "spouse")
    with pytest.raises(AECaseValidationError):
        engine.update("product", "")
    with pytest.raises(AECaseValidationError):
        engine.update("concomitant_medications", "Aspirin")