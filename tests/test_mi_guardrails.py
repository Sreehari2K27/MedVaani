"""Focused tests for MI guardrails."""
from __future__ import annotations

from app.mi.guardrails import MIGuardrail, MIGHybridService


def test_normal_product_information_question_allowed():
    result = MIGuardrail.check("What is Lantus used for?")
    assert result.allowed is True
    assert result.requires_retrieval is True


def test_storage_question_allowed():
    result = MIGuardrail.check("What are the storage conditions?")
    assert result.allowed is True


def test_stored_question_allowed():
    result = MIGuardrail.check("How should Lantus be stored?")
    assert result.allowed is True


def test_refrigeration_question_allowed():
    result = MIGuardrail.check("Does Lantus need refrigeration?")
    assert result.allowed is True
    assert result.requires_retrieval is True


def test_recommended_starting_dose_question_allowed():
    result = MIGuardrail.check("What is the recommended starting dose?")
    assert result.allowed is True


def test_adverse_reaction_product_information_allowed():
    result = MIGuardrail.check("What are the common adverse reactions listed in the package insert?")
    assert result.allowed is True


def test_personalized_dosing_request_blocked():
    result = MIGuardrail.check("My blood sugar is 250. How many units of Lantus should I take?")
    assert result.allowed is False
    assert "dosing" in result.response_text.lower() or "healthcare professional" in result.response_text.lower()


def test_dose_calculation_request_blocked():
    result = MIGuardrail.check("What dose should I take tonight?")
    assert result.allowed is False


def test_medication_stop_request_blocked():
    result = MIGuardrail.check("Should I stop taking Lantus?")
    assert result.allowed is False


def test_medication_change_request_blocked():
    result = MIGuardrail.check("Should I switch my insulin?")
    assert result.allowed is False


def test_personalized_treatment_request_blocked():
    result = MIGuardrail.check("My sugar is high, what should I do?")
    assert result.allowed is False


def test_unsupported_out_of_scope_request_bounded():
    result = MIGuardrail.check("nonexistentmedicationconceptxyz123")
    assert result.allowed is False
    assert result.response_text


def test_blocked_requests_do_not_call_llm():
    service = MIGHybridService()
    res = service.handle("How many units of Lantus should I take?")
    assert res.allowed is False
    assert res.status == "blocked"


def test_allowed_requests_still_use_retrieval_and_grounded_answer_generation():
    service = MIGHybridService()
    res = service.handle("What is Lantus used for?", top_k=2)
    assert res.allowed is True
    assert res.status == "allowed"
    assert res.answer
    assert res.source_page_numbers
    assert res.grounded is True
    assert res.supported is True


def test_blocked_results_report_not_grounded():
    service = MIGHybridService()
    res = service.handle("How many units of Lantus should I take?", top_k=2)
    assert res.allowed is False
    assert res.grounded is False
    assert res.supported is False
    assert res.source_page_numbers == []
