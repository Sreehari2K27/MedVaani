"""Focused regression tests for ambiguous seriousness answers.

An ambiguous claim such as "it is serious for me but I don't ..." must never
be stored as seriousness="no". The workflow asks one concise clarification; if
the answer is still unresolvable the case records "unknown" and pauses for
qualified human review. Priority/urgency semantics are untouched: only the
separate urgent-escalation path raises priority.
"""
from __future__ import annotations

import pytest

from app.providers.answer import AnswerLLMProvider, _seriousness_value
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    _FIELD_CLARIFICATIONS,
)

AMBIGUOUS = "it is serious for me but I don't take it lightly"
JOURNEY_SERIOUSNESS = (
    "I did not go to the hospital and I am not experiencing a life-threatening reaction"
)


def _engine_at_seriousness(case_id: str) -> AEStateEngine:
    """Engine with everything captured; the seriousness question is next."""
    engine = AEStateEngine.create_case(case_id)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "severe headache and vomiting")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "45")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    engine.update("contact_phone", "9999990000")
    engine.update("onset_date", "It started this morning")
    assert engine.next_field()[0] == "seriousness"
    return engine


def _workflow_at_seriousness(case_id: str):
    """Workflow whose previous turn asked the seriousness question."""
    engine = _engine_at_seriousness(case_id)
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("It started this morning")  # onset; asks seriousness
    assert engine.next_field()[0] == "seriousness"
    return engine, workflow


# ---- extraction: never guess "no" ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        AMBIGUOUS,
        "It is serious for me but I don't want to overreact",
    ],
)
def test_ambiguous_seriousness_is_never_no(text):
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["seriousness", "Would you consider this problem serious?"]}
    )
    # Nothing is stored: the workflow asks a clarification instead of guessing.
    assert "seriousness" not in result.extracted_fields


def test_seriousness_value_classifies_ambiguity():
    assert _seriousness_value(AMBIGUOUS) == "ambiguous"


@pytest.mark.parametrize(
    "text",
    [
        "it is not serious",
        "no, it is not serious",
        JOURNEY_SERIOUSNESS,
    ],
)
def test_direct_denial_still_captures_no(text):
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["seriousness", "Would you consider this problem serious?"]}
    )
    assert result.extracted_fields["seriousness"] == "no"


def test_plain_yes_answer_still_captures_yes():
    result = AnswerLLMProvider().interpret(
        "Yes, it is serious",
        {"next_field": ["seriousness", "Would you consider this problem serious?"]},
    )
    assert result.extracted_fields["seriousness"] == "yes"


def test_explicit_uncertainty_still_stores_unknown():
    result = AnswerLLMProvider().interpret(
        "I am not sure",
        {"next_field": ["seriousness", "Would you consider this problem serious?"]},
    )
    assert result.extracted_fields["seriousness"] == "unknown"


# ---- workflow: one clarification, then review fallback ---------------------


def test_first_ambiguous_answer_asks_one_concise_clarification():
    engine, workflow = _workflow_at_seriousness("AE-ser-ambig-1")

    result = workflow.process_turn(AMBIGUOUS)

    assert result.next_question == ["seriousness", _FIELD_CLARIFICATIONS["seriousness"]]
    assert engine.case.seriousness != "no"
    assert result.human_review_required is False
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW


def test_clear_answer_after_clarification_still_advances():
    engine, workflow = _workflow_at_seriousness("AE-ser-ambig-2")
    workflow.process_turn(AMBIGUOUS)

    result = workflow.process_turn("Yes, it is serious")

    assert engine.case.seriousness == "yes"
    assert result.next_question is not None
    assert result.next_question[0] == "outcome"
    assert result.human_review_required is False


def test_second_ambiguous_answer_records_unknown_and_flags_review():
    engine, workflow = _workflow_at_seriousness("AE-ser-ambig-3")
    workflow.process_turn(AMBIGUOUS)

    result = workflow.process_turn(AMBIGUOUS)

    assert engine.case.seriousness == "unknown"
    assert result.human_review_required is True
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    # Not an urgency escalation: priority stays normal.
    assert engine.case.priority == "normal"
    assert result.priority == "normal"
    assert result.completed is False
    # Questioning stops; collected information is preserved.
    assert result.next_question is None
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == "severe headache and vomiting"
    assert engine.case.human_review_reason is not None


def test_review_flag_blocks_auto_completion():
    engine, workflow = _workflow_at_seriousness("AE-ser-ambig-4")
    workflow.process_turn(AMBIGUOUS)
    workflow.process_turn(AMBIGUOUS)

    assert engine.can_complete() is False
    result = workflow.process_turn("I guess no")
    assert result.completed is False
    assert engine.current_state is WorkflowState.HUMAN_REVIEW


# ---- urgent escalation and safety are unchanged ----------------------------


def test_urgent_escalation_still_owns_high_priority():
    engine = AEStateEngine.create_case("AE-ser-urgent")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("I am having severe difficulty breathing")

    assert result.safety_status == "urgent_life_threatening"
    assert result.priority == "high"
    assert engine.case.human_review_reason


def test_journey_seriousness_phrase_is_still_no():
    engine, workflow = _workflow_at_seriousness("AE-ser-journey")
    workflow.process_turn(JOURNEY_SERIOUSNESS)

    assert engine.case.seriousness == "no"
    assert engine.current_state is WorkflowState.SERIOUSNESS_SCREENED
