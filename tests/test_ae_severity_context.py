"""Regression tests for premature severity/urgency from a symptom description.

A fresh AE session whose event description contains words like "serious" or
"severe" must never be recorded as a severity answer, must never enter the
urgent branch, and must never complete a report. Seriousness is only ever an
answer to the severity question at the approved step. All data is synthetic.
"""
from __future__ import annotations

from app.providers.answer import AnswerLLMProvider
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    CLOSING_MESSAGE,
    INTRODUCTION_MESSAGE,
    SCOPE_VERIFICATION_QUESTION,
    SEVERITY_URGENT_MESSAGE,
    VOICE_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)

DESCRIPTION = "I am facing serious stomach ache after taking this medication"


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine,
        AnswerLLMProvider(),
        confirm_fields=VOICE_CONFIRMATION_FIELDS,
        ask_additional_information=True,
    )
    return engine, workflow


# ---- 1 + 3. no premature urgency or seriousness from a description ---------


def test_fresh_session_with_serious_description_does_not_escalate_or_complete():
    engine, workflow = _workflow("sv-fresh")

    result = workflow.process_turn(DESCRIPTION)

    assert result.completed is False
    assert result.safety_status == "ok"
    assert result.human_review_required is False
    assert result.priority == "normal"
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert engine.current_state is not WorkflowState.CASE_COMPLETED
    assert "immediate medical attention" not in result.response_text


def test_event_description_is_preserved_and_medication_step_follows():
    engine, workflow = _workflow("sv-event-medication")

    result = workflow.process_turn(DESCRIPTION)

    # Verbatim description, and the approved Step 3 question follows.
    assert engine.case.adverse_event == DESCRIPTION
    assert result.next_question is not None
    assert result.next_question[0] == "product"
    assert "To help us record your report correctly, could you tell me the name" in (
        result.response_text
    )

    verified = workflow.process_turn("Lantus")
    assert verified.next_question is not None
    assert verified.next_question[0] == "product_verification"
    assert verified.next_question[1] == SCOPE_VERIFICATION_QUESTION


def test_severe_and_serious_words_never_populate_an_unasked_severity_answer():
    for case_id, text in (
        ("sv-serious-word", "I am facing serious stomach ache after taking Lantus"),
        ("sv-severe-word", "severe headache after taking this medication"),
    ):
        engine, workflow = _workflow(case_id)

        result = workflow.process_turn(text)

        # The description words never become a seriousness claim on the case
        # and never trigger the urgent branch.
        assert engine.case.seriousness == "unknown"
        assert result.priority == "normal"
        assert result.safety_status == "ok"


# ---- 2. the approved urgent branch still works ------------------------------


def test_explicit_urgent_answer_at_the_severity_step_still_escalates():
    engine, workflow = _workflow("sv-urgent-step")
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")  # Step 4
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Beta, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    workflow.process_turn("no")  # consent declined -> severity next

    urgent = workflow.process_turn("Yes, it was severe")

    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    assert urgent.next_question is None
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert "immediate medical attention" in urgent.response_text


def test_urgent_symptom_interruption_is_unaffected():
    engine, workflow = _workflow("sv-urgent-symptom")

    result = workflow.process_turn("I am having severe difficulty breathing")

    assert result.safety_status == "urgent_life_threatening"
    assert result.human_review_required is True
    assert engine.current_state is WorkflowState.HUMAN_REVIEW


# ---- 3. normal collection cannot be bypassed --------------------------------


def test_normal_journey_still_requires_every_step_before_the_report():
    engine, workflow = _workflow("sv-full-journey")
    workflow.process_turn("I have a headache after taking Lantus")
    assert "seriousness" in engine.missing_information()
    assert "contact_phone" in engine.missing_information()
    assert "consent_to_contact" in engine.missing_information()
    assert engine.can_complete() is False

    workflow.process_turn("yes")  # Step 4
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Beta, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    workflow.process_turn("no")  # consent declined
    workflow.process_turn("No, it was not severe")
    asked = workflow.process_turn("It started yesterday")

    # The closing question is asked; the report completes only after it.
    assert asked.completed is False
    assert "Is there anything else you'd like to share" in asked.response_text

    final = workflow.process_turn("no")

    # Identity/consent safeguards hold and the report completes only at the end.
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.consent_to_contact is False
    assert final.completed is True
    assert engine.current_state is WorkflowState.CASE_COMPLETED


# ---- 4. urgent and closing message integrity --------------------------------


def test_urgent_and_closing_messages_render_exactly_once():
    engine, workflow = _workflow("sv-messages")
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Beta, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    workflow.process_turn("no")

    urgent = workflow.process_turn("Yes, it was severe")

    # One urgent instruction, one closing, no duplicated introduction, and
    # no post-documentation question anywhere. The follow-up promise stays
    # conditional (only when consent was already given), never asserted.
    assert urgent.response_text.count("Please seek immediate medical attention") == 1
    assert urgent.response_text.count(SEVERITY_URGENT_MESSAGE) == 1
    assert urgent.response_text.count(CLOSING_MESSAGE) == 1
    assert urgent.response_text.count(INTRODUCTION_MESSAGE) <= 1
    assert "How are you feeling now" not in urgent.response_text
    assert "submitted" not in urgent.response_text.lower()


def test_normal_completion_closing_renders_exactly_once():
    engine, workflow = _workflow("sv-closing")
    workflow.process_turn("Hello")  # opening
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Beta, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    workflow.process_turn("no")
    workflow.process_turn("No, it was not severe")
    workflow.process_turn("It started yesterday")
    final = workflow.process_turn("no")

    assert final.completed is True
    assert final.response_text.count(CLOSING_MESSAGE) == 1
    assert INTRODUCTION_MESSAGE not in final.response_text
    assert final.response_text.count("immediate medical attention") == 0


# ---- 7. consent, identity and human-review safeguards ----------------------


def test_unanswered_consent_and_identity_safeguards_survive_the_fix():
    engine, workflow = _workflow("sv-safeguards")
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")

    collected = workflow.process_turn("Alpha Beta, 05-01-1990").collected_information

    # The name is only a candidate until confirmed; consent stays unanswered
    # (never silently "No").
    assert collected["patient_identifier"] is None
    assert collected["consent_to_contact"] is None
    assert engine.can_complete() is False
