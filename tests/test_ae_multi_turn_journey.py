"""Regression tests for the complete multi-turn AE reporting journey.

Every test drives a scripted conversation through the production
``AEStateEngine`` and ``ConversationalAEWorkflow`` using the offline harness,
with no database, speech, or hosted-model service involved.
"""
from __future__ import annotations

import ast
from pathlib import Path

import scripts.ae_journey_harness as harness_module
from app.providers.base import LLMResult
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import ConversationalAEWorkflow, _field_map_from_llm
from scripts.ae_journey_harness import (
    SCRIPTED_JOURNEY,
    URGENT_JOURNEY,
    URGENT_STATEMENT,
    TurnRecord,
    build_harness,
)


REQUIRED_TURN_FIELDS = (
    "user_input",
    "intent",
    "current_state",
    "extracted_information",
    "next_question",
    "priority",
    "human_review_required",
    "completed",
)


def test_initial_report_is_classified_as_ae_and_asks_for_reporter_role():
    harness = build_harness("AE-journey-001")
    first = harness.run_turn(SCRIPTED_JOURNEY[0])

    assert first.intent == "ae_report"
    assert first.current_state == WorkflowState.EVENT_IDENTIFIED.value
    assert first.extracted_information["product"]
    assert first.extracted_information["adverse_event"]
    assert first.next_question is not None
    assert first.next_question[0] == "reporter_role"
    assert first.priority == "normal"
    assert first.human_review_required is False
    assert first.completed is False


def test_reporter_role_then_patient_identification_captures_identity():
    harness = build_harness("AE-journey-002")
    harness.run_turn(SCRIPTED_JOURNEY[0])
    second = harness.run_turn(SCRIPTED_JOURNEY[1])

    # Reporter role is established before patient-specific questions.
    assert second.extracted_information == {"reporter_role": "self"}
    assert second.next_question is not None
    assert second.next_question[0] == "patient_identifier"

    third = harness.run_turn(SCRIPTED_JOURNEY[2])
    assert third.extracted_information == {
        "patient_identifier": "Demo Patient 001",
        "patient_age": "45",
    }
    assert harness.engine.case.patient_identifier == "Demo Patient 001"
    assert harness.engine.case.patient_age == "45"
    assert harness.engine.current_state is WorkflowState.PATIENT_IDENTIFIED
    assert third.next_question is not None
    assert third.next_question[0] == "consent_to_contact"


def test_reporter_and_contact_answers_reach_the_engine():
    harness = build_harness("AE-journey-003")
    harness.run_journey(SCRIPTED_JOURNEY[:7])

    case = harness.engine.case
    assert case.reporter_role == "self"
    assert case.reporter_name is None
    assert case.preferred_contact_method == "phone"
    assert case.contact_phone == "9999990000"
    assert case.consent_to_contact is True
    assert case.onset_date == "It started this morning"
    assert harness.engine.current_state is WorkflowState.EVENT_TIMING_CAPTURED
    assert harness.engine.missing_information() == ["seriousness", "outcome"]


def test_field_map_forwards_reporter_and_contact_fields_to_the_engine():
    result = LLMResult(
        intent="ae_report",
        extracted_fields={
            "reporter_role": "self",
            "reporter_name": "Demo Reporter",
            "contact_phone": "9999990000",
            "contact_email": "demo.patient@example.com",
            "consent_to_contact": True,
            "unmapped_noise": "dropped",
        },
    )

    assert _field_map_from_llm(result) == {
        "reporter_role": "self",
        "reporter_name": "Demo Reporter",
        "contact_phone": "9999990000",
        "contact_email": "demo.patient@example.com",
        "consent_to_contact": True,
    }


def test_normal_journey_walks_every_workflow_state_to_case_completed():
    harness = build_harness("AE-journey-004")
    records = harness.run_journey(SCRIPTED_JOURNEY)

    assert [record.current_state for record in records] == [
        WorkflowState.EVENT_IDENTIFIED.value,
        WorkflowState.EVENT_IDENTIFIED.value,
        WorkflowState.PATIENT_IDENTIFIED.value,
        WorkflowState.CONSENT_CAPTURED.value,
        WorkflowState.CONSENT_CAPTURED.value,
        WorkflowState.CONTACT_CAPTURED.value,
        WorkflowState.EVENT_TIMING_CAPTURED.value,
        WorkflowState.SERIOUSNESS_SCREENED.value,
        WorkflowState.CASE_COMPLETED.value,
    ]

    final = records[-1]
    assert final.completed is True
    assert final.current_state == WorkflowState.CASE_COMPLETED.value
    assert final.priority == "normal"
    assert final.human_review_required is False
    assert final.next_question is None
    assert harness.engine.current_state is WorkflowState.CASE_COMPLETED
    assert harness.engine.missing_information() == []
    assert harness.engine.to_dict()["report_complete"] is True


def test_normal_journey_captures_every_required_report_field():
    harness = build_harness("AE-journey-005")
    harness.run_journey(SCRIPTED_JOURNEY)

    snapshot = harness.engine.to_dict()
    assert snapshot["status"] == WorkflowState.CASE_COMPLETED.value
    assert snapshot["product"]
    assert snapshot["adverse_event"]
    assert snapshot["patient"]["identifier"] == "Demo Patient 001"
    assert snapshot["patient"]["age"] == "45"
    assert snapshot["reporter"]["role"] == "self"
    assert snapshot["reporter"]["same_as_patient"] is True
    assert snapshot["contact"]["preferred_method"] == "phone"
    assert snapshot["contact"]["phone"] == "9999990000"
    assert snapshot["contact"]["consent_to_contact"] is True
    assert snapshot["onset_date"] == "It started this morning"
    assert snapshot["seriousness"] == "no"
    assert snapshot["medical_care"] == "no"
    assert snapshot["outcome"] == "not_recovered"
    assert snapshot["next_question"] is None


def test_negated_seriousness_answer_does_not_escalate_the_normal_journey():
    harness = build_harness("AE-journey-006")
    records = harness.run_journey(SCRIPTED_JOURNEY)

    for record in records:
        assert record.safety_status != "urgent_life_threatening"
        assert record.human_review_required is False
        assert record.priority == "normal"

    seriousness_turn = records[7]
    assert seriousness_turn.user_input == SCRIPTED_JOURNEY[7]
    # Seeking (or not seeking) medical care is tracked separately from the
    # seriousness claim, so "did not go to the hospital" does not set a
    # seriousness of "no".
    assert seriousness_turn.extracted_information == {
        "seriousness": "no",
        "medical_care": "no",
    }
    assert harness.engine.case.human_review_reason is None


def test_urgent_journey_escalates_to_human_review_and_high_priority():
    harness = build_harness("AE-journey-007")
    records = harness.run_journey(URGENT_JOURNEY)

    escalation = next(record for record in records if record.user_input == URGENT_STATEMENT)
    assert escalation.safety_status == "urgent_life_threatening"
    assert escalation.current_state == WorkflowState.HUMAN_REVIEW.value
    assert escalation.priority == "high"
    assert escalation.human_review_required is True
    assert escalation.next_question is None
    assert escalation.completed is False

    assert harness.engine.current_state is WorkflowState.HUMAN_REVIEW
    assert harness.engine.case.priority == "high"
    assert harness.engine.case.human_review_reason


def test_urgent_journey_preserves_information_and_prevents_completion():
    harness = build_harness("AE-journey-008")
    records = harness.run_journey(URGENT_JOURNEY)

    final = records[-1]
    assert final.current_state == WorkflowState.HUMAN_REVIEW.value
    assert final.human_review_required is True
    assert final.completed is False
    assert final.next_question is None
    assert harness.engine.can_complete() is False

    snapshot = harness.engine.to_dict()
    assert snapshot["report_complete"] is False
    assert snapshot["status"] == WorkflowState.HUMAN_REVIEW.value
    assert snapshot["product"]
    assert snapshot["adverse_event"]
    assert snapshot["patient"]["identifier"] == "Demo Patient 001"
    assert snapshot["patient"]["age"] == "45"
    assert snapshot["reporter"]["role"] == "self"
    assert snapshot["contact"]["phone"] == "9999990000"
    assert snapshot["outcome"] == "not_recovered"


def test_every_turn_record_exposes_the_required_reported_fields():
    harness = build_harness("AE-journey-009")
    records = harness.run_journey(SCRIPTED_JOURNEY)

    assert len(records) == len(SCRIPTED_JOURNEY)
    declared = set(TurnRecord.__dataclass_fields__)
    for record in records:
        for field_name in REQUIRED_TURN_FIELDS:
            assert field_name in declared
        rendered = record.render()
        for field_name in REQUIRED_TURN_FIELDS:
            assert field_name in rendered or field_name.replace("_", " ") in rendered
        assert record.user_input


def test_harness_reuses_production_classes_and_imports_no_external_modules():
    harness = build_harness("AE-journey-010")

    assert isinstance(harness.engine, AEStateEngine)
    assert isinstance(harness.workflow, ConversationalAEWorkflow)
    assert isinstance(harness.workflow.engine, AEStateEngine)

    source = Path(harness_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = (
        "app.services.prisma",
        "app.services.timbre",
        "app.providers.mistral",
        "requests",
        "httpx",
        "urllib",
        "socket",
    )
    for module in forbidden:
        assert not any(name == module or name.startswith(module + ".") for name in imported), module
