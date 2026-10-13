"""Integration tests for conversational AE workflow."""
from __future__ import annotations

from app.providers.base import LLMResult
from app.providers.mock import MockLLMProvider
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import ConversationalAEWorkflow


def test_normal_ae_message_updates_case():
    engine = AEStateEngine.create_case("AE-conv-001-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("I took Lantus yesterday and since this morning I have a headache.")
    # Case should be updated based on extraction
    assert result.case_id == "AE-conv-001-demo"
    assert result.intent == "ae_report"
    assert result.safety_status == "ok"


def test_extracted_ae_information_is_preserved():
    engine = AEStateEngine.create_case("AE-conv-002-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("I developed a headache after taking Lantus.")
    assert result.extracted_information is not None
    assert "adverse_event" in result.extracted_information or "headache" in str(result.extracted_information).lower()
    assert result.collected_information is not None


def test_next_missing_field_is_returned():
    engine = AEStateEngine.create_case("AE-conv-003-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("I developed a headache after taking Lantus.")
    # Should return next missing field info
    assert result.missing_information is not None
    assert len(result.missing_information) > 0
    assert result.response_text  # has some response/question


def test_urgent_message_triggers_safety_interruption_high_priority_human_review():
    engine = AEStateEngine.create_case("AE-conv-004-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("After taking Lantus I am having severe difficulty breathing.")
    assert result.safety_status == "urgent_life_threatening"
    assert result.priority == "high"
    assert result.human_review_required is True
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert result.response_text  # contains safety instruction


def test_personalized_treatment_request_is_safely_bounded():
    engine = AEStateEngine.create_case("AE-conv-005-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("My sugar is 250. Should I increase my Lantus dose?")
    assert result.intent == "treatment_query"
    assert result.safety_status == "treatment_request_blocked"
    assert "healthcare professional" in result.response_text.lower()
    assert result.human_review_required is False


def test_product_information_request_is_routed_to_mi_rag_intent():
    engine = AEStateEngine.create_case("AE-conv-006-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("What is Lantus SoloStar?")
    assert result.intent == "product_info"
    assert result.safety_status == "product_info_routed_to_mi_rag"
    assert "mi/rag" in result.response_text.lower()


def test_case_remains_persistent_across_multiple_conversational_turns():
    engine = AEStateEngine.create_case("AE-conv-007-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    # Turn 1
    result1 = workflow.process_turn("I took Lantus and have a headache.")
    first_case_id = result1.case_id

    # Turn 2 - add more info
    workflow.process_turn("The headache started yesterday morning.")
    result2 = workflow.process_turn("I am 45 years old, patient ID PT-001.")
    
    assert result2.case_id == first_case_id
    assert result2.collected_information is not None


def test_incomplete_case_is_not_marked_completed():
    engine = AEStateEngine.create_case("AE-conv-008-demo")
    llm = MockLLMProvider()
    workflow = ConversationalAEWorkflow(engine, llm)

    result = workflow.process_turn("I developed a headache after taking Lantus.")
    assert result.completed is False
    assert engine.current_state is not WorkflowState.CASE_COMPLETED


def _fill_all_required_fields(engine: AEStateEngine) -> None:
    for field, value in (
        ("product", "Lantus (fictional)"),
        ("adverse_event", "Headache (demo)"),
        ("patient_identifier", "PT-001"),
        ("patient_age", "45"),
        ("reporter_role", "self"),
        ("contact_phone", "9999990000"),
        ("consent_to_contact", True),
        ("onset_date", "yesterday"),
        ("seriousness", "no"),
        ("outcome", "recovering"),
    ):
        engine.update(field, value)


def test_workflow_finalizes_case_once_required_fields_are_satisfied():
    engine = AEStateEngine.create_case("AE-conv-009-demo")
    workflow = ConversationalAEWorkflow(engine, MockLLMProvider())
    _fill_all_required_fields(engine)
    assert engine.current_state is WorkflowState.CASE_REVIEW
    assert engine.can_complete() is True

    result = workflow.process_turn("Thank you, that is all the information.")

    assert engine.current_state is WorkflowState.CASE_COMPLETED
    assert result.completed is True
    assert result.current_state == WorkflowState.CASE_COMPLETED.value
    assert result.next_question is None
    assert result.human_review_required is False


def test_workflow_does_not_finalize_a_case_under_human_review():
    engine = AEStateEngine.create_case("AE-conv-010-demo")
    workflow = ConversationalAEWorkflow(engine, MockLLMProvider())
    _fill_all_required_fields(engine)
    engine.mark_urgent(reason="escalated for manual review")

    result = workflow.process_turn("Thank you, that is all the information.")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert result.completed is False
    assert result.human_review_required is True
    assert result.priority == "high"


# ---- LLM product-info intent gating -----------------------------------------


class _ProductInfoLLM:
    """Fake provider that always claims a product-information intent."""

    def interpret(self, message, context=None):
        return LLMResult(
            intent="product_info",
            extracted_fields={"dose_and_use": "once daily at bedtime"},
        )


class _NeutralLLM:
    """Fake provider with no interpretation signals."""

    def interpret(self, message, context=None):
        return LLMResult(intent="unknown")


def test_llm_product_info_intent_does_not_divert_an_active_report():
    engine = AEStateEngine.create_case("AE-gate-001-demo")
    workflow = ConversationalAEWorkflow(engine, _ProductInfoLLM())
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash at the injection site")

    result = workflow.process_turn("Does Lantus cause rash?")

    # The unverified LLM intent is demoted; the interview continues.
    assert result.intent == "unknown"
    assert result.safety_status == "ok"
    # The turn's legitimate extracted field is still applied.
    assert engine.case.dose_and_use == "once daily at bedtime"


def test_llm_product_info_intent_still_honoured_on_an_empty_case():
    engine = AEStateEngine.create_case("AE-gate-002-demo")
    workflow = ConversationalAEWorkflow(engine, _ProductInfoLLM())

    result = workflow.process_turn("What is Lantus SoloStar?")

    assert result.intent == "product_info"
    assert result.safety_status == "product_info_routed_to_mi_rag"
    assert "mi/rag" in result.response_text.lower()


def test_deterministic_safety_product_info_routing_is_preserved():
    engine = AEStateEngine.create_case("AE-gate-003-demo")
    workflow = ConversationalAEWorkflow(engine, _NeutralLLM())

    result = workflow.process_turn("How does Lantus work?")

    # The deterministic safety layer alone routes this turn, regardless of
    # the LLM's (neutral) interpretation.
    assert result.safety_status == "product_info_routed_to_mi_rag"
    assert "mi/rag" in result.response_text.lower()


def test_urgent_escalation_is_preserved_despite_llm_product_info_intent():
    engine = AEStateEngine.create_case("AE-gate-004-demo")
    workflow = ConversationalAEWorkflow(engine, _ProductInfoLLM())
    engine.update("product", "Lantus SoloStar")

    result = workflow.process_turn(
        "I am having severe chest pain and cannot breathe."
    )

    assert result.safety_status == "urgent_life_threatening"
    assert result.human_review_required is True
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
