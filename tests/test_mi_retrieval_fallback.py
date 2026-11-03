"""Regression tests for MI retrieval under dependency degradation.

Background (deployment incident): scikit-learn was missing from
requirements.txt, so the Render deployment silently degraded to a keyword
fallback that only matched the WHOLE query as an exact substring - zero
chunks for every natural-language question - and MI answered every question
with the insufficient-information message. These tests pin the fixed
behavior:

- scikit-learn is declared as a dependency (deploy parity).
- TF-IDF retrieval remains the primary mechanism and works.
- The degraded fallback retrieves relevant chunks by idf-weighted token
  overlap computed from the reference chunks themselves.
- Degradation is loudly logged (never silent again).
- Unsupported questions still surface the existing insufficient-information
  response and nothing is ever invented.

All tests are offline: no network, no live LLM calls.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

import app.mi.retrieval.retriever as retriever_module
from app.mi.retrieval.retriever import LantusRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]

_BASIC_QUESTIONS = (
    "What is Lantus SoloStar used for?",
    "How should Lantus be stored?",
    "What are the side effects of Lantus?",
)

# The semantically correct source chunk for each basic question, by page
# number in the processed package-insert chunks (committed reference data).
_EXPECTED_PAGE = {
    _BASIC_QUESTIONS[0]: [1],   # Indication (page 1)
    _BASIC_QUESTIONS[1]: [11],  # Storage Conditions and handling (page 11)
    _BASIC_QUESTIONS[2]: [6],   # Adverse Reactions (page 6)
}


def _degrade_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the retriever module into its no-sklearn degraded mode."""
    monkeypatch.setattr(retriever_module, "TfidfVectorizer", None)
    monkeypatch.setattr(retriever_module, "cosine_similarity", None)


# ---- dependency declaration -------------------------------------------------


def test_scikit_learn_is_declared_in_requirements() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(
        r"(?mi)^\s*scikit-learn\s*>=\s*\d", requirements
    ), "scikit-learn must be pinned in requirements.txt; without it the "
    "deployment silently degrades to keyword-only retrieval"


def test_tfidf_index_builds_in_this_environment() -> None:
    iris = LantusRetriever()
    assert retriever_module.TfidfVectorizer is not None, (
        "scikit-learn must be importable in the test environment"
    )
    assert iris.vectorizer is not None
    assert iris.tfidf_matrix is not None
    assert iris.get_chunk_count() > 0


# ---- primary TF-IDF retrieval ------------------------------------------------


@pytest.mark.parametrize("question", _BASIC_QUESTIONS)
def test_tfidf_retrieval_finds_relevant_chunks(question: str) -> None:
    retriever = LantusRetriever()
    results = retriever.retrieve(question, k=3)
    assert results, f"TF-IDF retrieval found nothing for: {question}"
    pages = [c.page_number for c in results]
    assert set(_EXPECTED_PAGE[question]) & set(pages)


# ---- degraded fallback retrieval (sklearn unavailable) -----------------------


@pytest.mark.parametrize("question", _BASIC_QUESTIONS)
def test_degraded_fallback_finds_relevant_chunks(
    question: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _degrade_module(monkeypatch)
    retriever = LantusRetriever()
    assert retriever.vectorizer is None
    assert retriever.tfidf_matrix is None
    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        results = retriever.retrieve(question, k=3)
    assert results, f"degraded fallback found nothing for: {question}"
    pages = [c.page_number for c in results]
    assert set(_EXPECTED_PAGE[question]) & set(pages), (
        f"degraded fallback missed the relevant source chunk for: {question}"
    )
    # The per-request log makes degraded retrieval identifiable in logs.
    assert any("degraded" in record.message.lower() for record in caplog.records)


def test_degraded_fallback_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _degrade_module(monkeypatch)
    first = LantusRetriever().retrieve(_BASIC_QUESTIONS[0], k=3)
    second = LantusRetriever().retrieve(_BASIC_QUESTIONS[0], k=3)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.relevance_score for c in first] == [
        c.relevance_score for c in second
    ]


# ---- degraded-mode diagnostics ------------------------------------------------


def test_warning_logged_when_sklearn_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _degrade_module(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        LantusRetriever()
    assert any(
        "scikit-learn unavailable" in record.message
        for record in caplog.records
    )


def test_warning_logged_when_chunks_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        LantusRetriever(chunks_path=tmp_path / "missing_chunks.json")
    assert any(
        "not found" in record.message.lower() for record in caplog.records
    )


def test_degraded_warning_never_carries_user_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _degrade_module(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=retriever_module.__name__):
        retriever = LantusRetriever()
        retriever.retrieve(_BASIC_QUESTIONS[0], k=3)
    assert all(
        _BASIC_QUESTIONS[0] not in record.getMessage() for record in caplog.records
    )


# ---- unsupported questions keep the insufficient response --------------------


def test_unsupported_question_gets_insufficient_response_in_degraded_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _degrade_module(monkeypatch)
    retriever = LantusRetriever()
    assert retriever.retrieve("nonexistentproductconceptxyz123", k=3) == []

    from app.providers import MockLLMProvider

    from app.mi.answer.service import MIAnswerService

    service = MIAnswerService(retriever=retriever, llm_provider=MockLLMProvider())
    result = service.answer("nonexistentproductconceptxyz123", top_k=2)
    assert result.unsupported is True
    assert result.grounded is False
    assert result.source_page_numbers == []
    assert "sufficient information is not available" in result.answer.lower()


def test_answer_service_degraded_still_answers_basic_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _degrade_module(monkeypatch)
    from app.providers import MockLLMProvider

    from app.mi.answer.service import MIAnswerService

    service = MIAnswerService(retriever=LantusRetriever(), llm_provider=MockLLMProvider())
    for question, expected_page in _EXPECTED_PAGE.items():
        result = service.answer(question, top_k=3)
        assert result.unsupported is False, question
        assert result.source_page_numbers, question
        assert expected_page[0] in result.source_page_numbers, question
        # Every fact stays quoted from the insert with a page citation.
        assert "(Source: Lantus package insert" in result.answer

    unsupported = service.answer("nonexistentproductconceptxyz123", top_k=3)
    assert unsupported.unsupported is True
    assert "sufficient information is not available" in unsupported.answer.lower()
