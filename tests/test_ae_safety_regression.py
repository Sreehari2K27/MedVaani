"""Regression tests for AE safety behavior."""
from __future__ import annotations

from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import ConversationalAEWorkflow
from app.providers.mock import MockLLMProvider
from app.safety import SafetyLayer


def test_normal_ae_includes_physician_consultation():
    engine = AEStateEngine.create_case("ae-safety-1")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn("I took Lantus and have a headache.")
    response = res.response_text.lower()
    assert "physician" in response or "healthcare professional" in response
    assert "consult" in response


def test_normal_ae_report_includes_physician_consultation():
    engine = AEStateEngine.create_case("ae-safety-2")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn("I want to report an adverse event after taking Lantus - severe headache")
    response = res.response_text.lower()
    assert "physician" in response or "healthcare professional" in response


def test_urgent_ae_escalates_to_immediate_attention():
    result = SafetyLayer.classify_input("I am having severe difficulty breathing after taking Lantus")
    assert result.signal.name == "URGENT_LIFE_THREATENING"
    assert result.should_interrupt_ae is True
    assert result.mark_high_priority is True
    assert "emergency" in result.message.lower() or "immediate" in result.message.lower()


def test_physician_consultation_warning_is_shown_once_per_session():
    engine = AEStateEngine.create_case("ae-safety-warn-once")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())

    # First routine AE turn: the general warning is present.
    first = wf.process_turn("I took Lantus and have a headache.")
    first_response = first.response_text.lower()
    assert "physician" in first_response or "healthcare professional" in first_response
    assert "consult" in first_response

    # Subsequent routine AE turns: warning must not repeat.
    for follow_up in (
        "The headache started yesterday morning.",
        "I am 45 years old, patient ID PT-001.",
        "My phone number is 9999990000.",
    ):
        res = wf.process_turn(follow_up)
        response = res.response_text.lower()
        assert "physician" not in response
        assert "healthcare professional" not in response


def test_new_ae_session_receives_its_own_physician_consultation_warning():
    first_engine = AEStateEngine.create_case("ae-safety-warn-session-1")
    first_wf = ConversationalAEWorkflow(first_engine, MockLLMProvider())
    first_wf.process_turn("I took Lantus and have a headache.")
    first_follow_up = first_wf.process_turn("The headache started yesterday morning.")
    assert "physician" not in first_follow_up.response_text.lower()

    # A brand new AE session must show the warning again on its first AE turn.
    second_engine = AEStateEngine.create_case("ae-safety-warn-session-2")
    second_wf = ConversationalAEWorkflow(second_engine, MockLLMProvider())
    second_first = second_wf.process_turn("I took Lantus and have a headache.")
    response = second_first.response_text.lower()
    assert "physician" in response or "healthcare professional" in response
    assert "consult" in response


def test_urgent_safety_message_still_appears_after_warning_suppressed():
    engine = AEStateEngine.create_case("ae-safety-urgent-after-warn")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())

    # Consume the general AE warning first.
    wf.process_turn("I took Lantus and have a headache.")

    # An urgent message must still surface the urgent safety message.
    urgent = wf.process_turn("After taking Lantus I am having severe difficulty breathing.")
    urgent_response = urgent.response_text.lower()
    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert "emergency" in urgent_response or "immediate" in urgent_response


def test_treatment_request_blocked_with_no_dose_recommendation():
    from app.workflow.ae_engine import AEStateEngine
    from app.workflow.service.conversational import ConversationalAEWorkflow
    from app.providers.mock import MockLLMProvider
    engine = AEStateEngine.create_case("ae-safety-3")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn("My sugar is 250, should I increase my Lantus dose?")
    response = res.response_text.lower()
    assert "healthcare professional" in response
    assert "increase" not in response or "cannot" in response  # must not recommend
    assert "dose" not in response.split("cannot")[0].split("should")[0] if "cannot" in response else True  # safety
