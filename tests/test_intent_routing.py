"""Tests for deterministic AE/MI intent routing."""
from __future__ import annotations

import pytest

from app.routing import route_transcript


def route(text):
    return route_transcript(text)


def test_product_info_question_routes_to_mi():
    assert route("What is Lantus used for?").workflow == "mi"


def test_side_effects_question_routes_to_mi():
    decision = route("What are the side effects of Lantus?")
    assert decision.workflow == "mi"
    assert decision.intent == "product_info"


def test_storage_question_routes_to_mi():
    assert route("How should Lantus be stored?").workflow == "mi"


def test_refrigeration_question_routes_to_mi():
    assert route("Does Lantus need refrigeration?").workflow == "mi"


def test_starting_dose_question_routes_to_mi():
    assert route("What is the recommended starting dose?").workflow == "mi"


def test_how_does_lantus_work_routes_to_mi():
    assert route("How does Lantus work?").workflow == "mi"


def test_personalized_dosing_question_routes_to_mi_with_refusal():
    decision = route("How many units should I take tonight?")
    assert decision.workflow == "mi"
    assert decision.intent == "treatment_query"
    assert decision.reason == "mi_guardrail_blocked_question"


@pytest.mark.parametrize(
    "text",
    [
        "What side effects does Lantus have?",
        "Does Lantus have side effects?",
    ],
)
def test_side_effect_question_variants_route_to_mi(text):
    assert route(text).workflow == "mi"


def test_urgent_utterance_routes_to_ae_even_with_mi_words():
    decision = route("I am having severe difficulty breathing, what is Lantus used for?")
    assert decision.workflow == "ae"
    assert decision.intent == "urgent_medical"


def test_ae_report_statement_routes_to_ae():
    decision = route("I want to report a side effect after taking Lantus")
    assert decision.workflow == "ae"
    assert decision.reason == "safety_ae_reporting"


def test_ae_report_with_event_routes_to_ae():
    text = "I took Lantus yesterday and had a severe headache and vomiting"
    decision = route(text)
    assert decision.workflow == "ae"
    assert decision.intent == "ae_report"


def test_treatment_request_via_safety_routes_to_ae_bounded():
    decision = route("What dose should I take tonight?")
    assert decision.workflow == "ae"
    assert decision.reason == "safety_treatment_bounded"


def test_chat_like_utterance_defaults_to_ae():
    decision = route("My name is John")
    assert decision.workflow == "ae"
    assert decision.reason == "default_ae_workflow"


def test_empty_input_defaults_to_ae():
    assert route("").workflow == "ae"
    assert route("   ").workflow == "ae"


def test_missed_dose_question_stays_in_ae_workflow():
    decision = route("What should I do if I miss a dose?")
    assert decision.workflow == "ae"


def test_dosing_request_blocked_by_guardrail_not_safety():
    decision = route("What is the dose calculation for Lantus?")
    assert decision.workflow == "mi"
    assert decision.intent == "treatment_query"