"""Regression tests: premature report after the initial Hinglish event description.

Repro transcript: after the approved introduction the person says
``dubara davai lene ke baad mera bahut pet dard karne laga hai`` — the
severity-bearing abdominal wording is the INITIAL event description of a
fresh case. It must be captured verbatim and taken through the approved
interview (medication next), never intercepted as an in-progress emergency.

Safeguards that must NOT change:
- standalone safety classification of abdominal severity (no case) stays
  urgent, exactly as ``test_safety_layer`` pins it,
- abdominal severity once wording is recorded on the case still interrupts,
- unambiguous first-turn emergencies (breathing, death, …) still interrupt,
- completion still requires every required field; consent is never invented.

All synthetic: deterministic providers, mocked STT/TTS, no live calls.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.answer import AnswerLLMProvider
from app.safety import SafetyLayer, SafetySignal
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import ConversationalAEWorkflow

SENTENCE = "dubara davai lene ke baad mera bahut pet dard karne laga hai"
HEADACHE = "I took Lantus and have a headache."
EMERGENCY_CUE = "medical emergency"


def _fresh_workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    return engine, workflow


# ---- the exact supplied transcript ------------------------------------------


def test_exact_hinglish_initial_description_continues_normal_workflow():
    engine, workflow = _fresh_workflow("AE-premature-1")

    result = workflow.process_turn(SENTENCE)

    # Not an emergency, no escalation, no report-worthy completion.
    assert result.safety_status == "ok"
    assert result.human_review_required is False
    assert result.completed is False
    assert result.priority == "normal"
    assert result.current_state not in ("HUMAN_REVIEW", "CASE_COMPLETED")
    assert EMERGENCY_CUE not in result.response_text.lower()

    # Approved flow continues at the medication step (expected step 2).
    assert result.next_question is not None
    assert result.next_question[0] == "product"
    assert "medication" in result.next_question[1].lower()

    # The original wording is verbatim; nothing else was invented.
    assert engine.case.adverse_event == SENTENCE
    assert engine.case.consent_to_contact is None
    assert engine.case.seriousness not in ("yes", "no")


def test_exact_sentence_via_text_endpoint_produces_no_premature_report():
    app = create_app(prisma=object(), timbre=object())
    client = TestClient(app)
    resp = client.post(
        "/text/conversation",
        data={"session_id": "AE-premature-endpoint", "message": SENTENCE},
    )
    assert resp.status_code == 200
    body = resp.json()

    # The observed regression: report + consent "Not provided" + emergency
    # warning + conversation end — none of that may happen here.
    assert body["completed"] is False
    assert body["human_review_required"] is False
    assert body["safety_status"] == "ok"
    assert body["priority"] == "normal"
    assert EMERGENCY_CUE not in body["response_text"].lower()

    # The interview continues and required fields are still outstanding.
    assert body["next_question"] is not None
    assert body["next_question"][0] == "product"
    assert body["missing_information"]
    assert "consent_to_contact" in body["missing_information"]

    # Event description preserved verbatim; consent untouched (not shown).
    collected = body["ae"]["collected_information"]
    assert collected["adverse_event"] == SENTENCE
    assert collected["consent_to_contact"] is None


def test_standalone_abdominal_severity_classification_is_unchanged():
    # No-case classification (routing, standalone safety checks) keeps the
    # approved semantics pinned by tests/test_safety_layer.py.
    result = SafetyLayer.classify_input(SENTENCE)
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True


# ---- non-urgent abdominal-pain reports keep flowing --------------------------


def test_non_urgent_abdominal_pain_first_turn_continues_normal_workflow():
    for case_id, text in (
        ("AE-premature-mild", "mere ko halka pet dard hai"),
        ("AE-premature-deva", "बहुत पेट दर्द हो रहा है"),
    ):
        engine, workflow = _fresh_workflow(case_id)

        result = workflow.process_turn(text)

        assert result.safety_status == "ok", text
        assert result.human_review_required is False, text
        assert result.completed is False, text
        assert EMERGENCY_CUE not in result.response_text.lower(), text
        assert result.next_question is not None, text
        # Wording preserved verbatim, in whichever field the deterministic
        # interpreter recorded it.
        recorded = engine.case.adverse_event or engine.case.unclassified_initial_input
        assert recorded == text, text


# ---- safeguards that must not weaken ----------------------------------------


def test_first_turn_unambiguous_emergency_still_interrupts():
    engine, workflow = _fresh_workflow("AE-premature-emergency")

    result = workflow.process_turn("I am having severe difficulty breathing")

    assert result.safety_status == "urgent_life_threatening"
    assert result.human_review_required is True
    assert result.priority == "high"
    assert result.current_state == "HUMAN_REVIEW"
    response = result.response_text.lower()
    assert "emergency" in response or "immediate" in response
    assert result.next_question is None


def test_abdominal_severity_mid_interview_still_interrupts():
    engine, workflow = _fresh_workflow("AE-premature-mid")

    first = workflow.process_turn(HEADACHE)
    assert first.safety_status == "ok"
    original = engine.case.adverse_event
    assert original == HEADACHE

    urgent = workflow.process_turn(
        "kal davai lene ke baad mujhe aaj bahut pet dard kar raha hai"
    )
    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    # Original description preserved through the escalation.
    assert original in engine.case.adverse_event


# ---- completion and wording invariants on the normal flow -------------------


def test_incomplete_required_fields_never_produce_a_completed_report():
    engine, workflow = _fresh_workflow("AE-premature-incomplete")

    first = workflow.process_turn(SENTENCE)
    assert first.completed is False
    assert engine.can_complete() is False

    second = workflow.process_turn("Lantus")
    assert second.completed is False
    assert second.current_state != "CASE_COMPLETED"
    assert engine.can_complete() is False
    # Consent, severity and the rest are still required.
    assert "consent_to_contact" in second.missing_information
    assert "seriousness" in second.missing_information


def test_original_event_wording_unchanged_by_followup_turns():
    engine, workflow = _fresh_workflow("AE-premature-wording")

    workflow.process_turn(SENTENCE)
    assert engine.case.adverse_event == SENTENCE

    workflow.process_turn("Lantus")
    assert engine.case.adverse_event == SENTENCE

    workflow.process_turn("It started yesterday")
    assert engine.case.adverse_event == SENTENCE
