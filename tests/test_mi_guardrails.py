"""Focused tests for MI guardrails."""
from __future__ import annotations

from app.mi.guardrails import MIGuardrail, MIGHybridService, MIRule


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


def test_mechanism_of_action_questions_allowed():
    """General mechanism questions are product information (the same
    recognition SafetyLayer already applies for routing)."""
    for question in (
        "How does Lantus SoloStar work?",
        "What is the mechanism of action of Lantus?",
    ):
        result = MIGuardrail.check(question)
        assert result.allowed is True, question
        assert result.rule is MIRule.ALLOWED_PRODUCT_INFO
        assert result.requires_retrieval is True


def test_manufacturer_instruction_questions_allowed():
    """General device/leaflet how-to questions reach retrieval, including the
    audited pen-use topics."""
    for question in (
        "How do I use the Lantus SoloStar pen?",
        "How do I attach a new needle to the SoloStar pen?",
        "How do I perform the safety test before injecting?",
        "How do I remove and dispose of the needle after injection?",
        "How do I check the insulin before using my Lantus SoloStar pen?",
        "How do I select the dose and inject with the SoloStar pen?",
    ):
        result = MIGuardrail.check(question)
        assert result.allowed is True, question
        assert result.rule is MIRule.ALLOWED_PRODUCT_INFO


def test_observed_fallback_questions_allowed():
    """The three audited browser questions that used to fall back as out of
    scope must reach the grounded retrieval answer path."""
    for question in (
        "What is the active ingredient in Lantus SoloStar?",
        "Can I reuse the needle on my Lantus SoloStar pen?",
        "Can Lantus SoloStar cure diabetes permanently?",
    ):
        result = MIGuardrail.check(question)
        assert result.allowed is True, question
        assert result.rule is MIRule.ALLOWED_PRODUCT_INFO, question
        assert result.category == "product_info", question
        assert result.requires_retrieval is True, question
        assert result.response_text == ""  # no fallback string


def test_hybrid_service_does_not_fall_back_on_observed_questions():
    """End-to-end through the guardrail service: the audited questions are
    routed to the answer service instead of returning the scope fallback."""
    service = MIGHybridService()
    for question in (
        "What is the active ingredient in Lantus SoloStar?",
        "Can I reuse the needle on my Lantus SoloStar pen?",
        "Can Lantus SoloStar cure diabetes permanently?",
    ):
        result = service.handle(question)
        assert result.status == "allowed", question
        assert result.allowed is True, question
        assert "I can only answer general product information" not in result.answer


def test_new_terms_stay_question_shaped():
    """Statements (not questions) mentioning the new terms stay out of scope:
    no unrestricted keyword allow was added."""
    for statement in (
        "My doctor told me reuse of the needle is bad",
        "I want a cure for this",
        "The composition is printed on the leaflet",
    ):
        result = MIGuardrail.check(statement)
        assert result.allowed is False, statement
        assert result.rule is MIRule.OUT_OF_SCOPE, statement


def test_mixed_wording_with_new_stay_blocked_by_safety_checks():
    """Personalized intents phrased with the new terminology are still caught
    by the SAFETY checks that run before the product-info allow."""
    blocked_cases = (
        (
            "Should I stop taking Lantus and find a cure instead?",
            MIRule.BLOCKED_MEDICATION_CHANGE,
        ),
        (
            "What treatment should I take for a permanent cure?",
            MIRule.BLOCKED_PERSONALIZED_TREATMENT,
        ),
        (
            "How many units should I take to reuse my pen safely?",
            MIRule.BLOCKED_PERSONALIZED_DOSING,
        ),
        (
            # Pre-existing precedence: the "what should I do … sugar" idiom is
            # classified as personalized dosing before the treatment-rule
            # class is ever consulted.
            "What should I do to cure my high sugar for me?",
            MIRule.BLOCKED_PERSONALIZED_DOSING,
        ),
    )
    for question, expected_rule in blocked_cases:
        result = MIGuardrail.check(question)
        assert result.allowed is False, question
        assert result.rule is expected_rule, question


def test_unsafe_intents_still_block_before_product_info_rule():
    """The new product-info clauses never override the earlier safety checks:
    personalized dosing, medication change and personalized treatment are
    evaluated first and stay blocked."""
    blocked_cases = (
        ("How many units should I take tonight?", MIRule.BLOCKED_PERSONALIZED_DOSING),
        (
            "My blood sugar is 250. How many units of Lantus should I take?",
            MIRule.BLOCKED_PERSONALIZED_DOSING,
        ),
        ("Should I stop taking Lantus?", MIRule.BLOCKED_MEDICATION_CHANGE),
        ("Should I switch my insulin?", MIRule.BLOCKED_MEDICATION_CHANGE),
        ("My sugar is high, what should I do?", MIRule.BLOCKED_PERSONALIZED_TREATMENT),
    )
    for question, rule in blocked_cases:
        result = MIGuardrail.check(question)
        assert result.allowed is False, question
        assert result.rule is rule, question


def test_arbitrary_what_should_i_do_stays_blocked():
    """Scope stays limited: open-ended advice requests are not product info."""
    result = MIGuardrail.check("What should I do if I miss a dose?")
    assert result.allowed is False


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
