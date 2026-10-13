"""Regression tests for AE intent classification."""
from __future__ import annotations

from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import ConversationalAEWorkflow
from app.providers.mock import MockLLMProvider


def test_exact_transcript_classified_as_ae():
    transcript = "hi yesterday i had taken your lantus medication however today i am facing severe headache and vomiting i want to report this adverse event"
    engine = AEStateEngine.create_case("reg-001")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn(transcript)
    assert res.intent == "ae_report"


def test_want_to_report_ae_variant():
    transcript = "I want to report an adverse event after taking Lantus"
    engine = AEStateEngine.create_case("reg-002")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn(transcript)
    assert res.intent == "ae_report"


def test_side_effect_variant():
    transcript = "I want to report a side effect - I have a headache after taking Lantus"
    engine = AEStateEngine.create_case("reg-003")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn(transcript)
    assert res.intent == "ae_report"


def test_reaction_after_taking_variant():
    transcript = "I had a reaction after taking Lantus - severe headache and vomiting"
    engine = AEStateEngine.create_case("reg-004")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn(transcript)
    assert res.intent == "ae_report"


def test_severe_headache_vomiting_variant():
    transcript = "I am having severe headache and vomiting after taking Lantus"
    engine = AEStateEngine.create_case("reg-005")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    res = wf.process_turn(transcript)
    assert res.intent == "ae_report"
