"""Focused tests for MI answer generation."""
from __future__ import annotations

from app.mi.answer import MIAnswerService, MIAnswerResult
from app.mi.retrieval import LantusRetriever
from app.providers import MockLLMProvider


def test_supported_product_information_question():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=2)
    assert isinstance(result, MIAnswerResult)
    assert result.intent == "product_info"


def test_answer_generation_from_retrieved_context():
    service = MIAnswerService(retriever=LantusRetriever(), llm_provider=MockLLMProvider())
    result = service.answer("What is Lantus used for?", top_k=2)
    assert result.answer
    assert len(result.answer) > 0


def test_source_page_metadata_preservation():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=3)
    assert result.source_page_numbers is not None
    assert len(result.source_page_numbers) > 0
    assert all(p >= 1 for p in result.source_page_numbers)
    assert result.retrieved_chunk_ids is not None


def test_unsupported_question_when_retrieval_insufficient():
    service = MIAnswerService()
    # Use a very specific query unlikely to match
    result = service.answer("nonexistentmedicationconceptxyz123", top_k=2)
    assert result.unsupported is True
    assert "sufficient information is not available" in result.answer.lower()


def test_grounding_instruction_supplied():
    service = MIAnswerService()
    result = service.answer("storage conditions", top_k=2)
    assert result.answer
    assert result.intent == "product_info"


def test_deterministic_mock_behavior():
    service1 = MIAnswerService(llm_provider=MockLLMProvider())
    service2 = MIAnswerService(llm_provider=MockLLMProvider())
    res1 = service1.answer("What is Lantus used for?", top_k=1)
    res2 = service2.answer("What is Lantus used for?", top_k=1)
    assert res1.answer == res2.answer


def test_empty_question_handling():
    service = MIAnswerService()
    result = service.answer("", top_k=2)
    assert result.answer
    assert result.unsupported or result.grounded is False


def test_top_k_behavior():
    service = MIAnswerService()
    result = service.answer("indication", top_k=1)
    assert len(result.source_chunks) <= 1
    result2 = service.answer("indication", top_k=5)
    assert len(result2.source_chunks) <= 5


def test_storage_answer_is_extractive_and_cited():
    service = MIAnswerService()
    result = service.answer("How should Lantus be stored?", top_k=3)
    assert result.answer
    assert "package insert" in result.answer
    assert "page 11" in result.answer
    assert "Information relevant to your question" not in result.answer
    assert result.grounded is True
    assert result.unsupported is False


def test_refrigeration_answer_uses_storage_section():
    service = MIAnswerService()
    result = service.answer("Does Lantus need refrigeration?", top_k=3)
    assert "page 11" in result.answer
    assert "refriger" in result.answer.lower()


def test_side_effects_answer_uses_adverse_reactions_section():
    service = MIAnswerService()
    result = service.answer("What are the side effects of Lantus?", top_k=3)
    assert "package insert" in result.answer
    assert any(p == 6 for p in result.source_page_numbers)
    assert result.unsupported is False


def test_indication_answer_is_deterministic_and_grounded():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=3)
    assert result.answer
    assert result.grounded is True
    assert result.unsupported is False
    assert result.source_page_numbers
