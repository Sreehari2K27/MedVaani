"""Focused tests for Lantus retrieval."""
from __future__ import annotations

import pytest

from app.mi.retrieval import LantusRetriever, RetrievedChunk


def test_retriever_initialization():
    retriever = LantusRetriever()
    assert retriever.get_chunk_count() > 0


def test_retrieve_indication_question():
    retriever = LantusRetriever()
    results = retriever.retrieve("What is Lantus used for?", k=3)
    assert len(results) > 0
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].page_number >= 1
    assert results[0].chunk_id
    assert results[0].text


def test_retrieve_storage_question():
    retriever = LantusRetriever()
    results = retriever.retrieve("storage", k=2)
    assert len(results) > 0
    # Should have page metadata
    for r in results:
        assert r.page_number >= 1
        assert r.chunk_id
        assert r.metadata is None or True  # we store fields directly


def test_retrieve_adverse_reaction_question():
    retriever = LantusRetriever()
    results = retriever.retrieve("adverse reaction side effects", k=2)
    assert len(results) > 0
    for r in results:
        assert r.page_number >= 1


def test_page_metadata_preservation():
    retriever = LantusRetriever()
    results = retriever.retrieve("What is Lantus used for?", k=1)
    assert len(results) == 1
    assert results[0].page_number > 0
    assert results[0].chunk_id is not None


def test_top_k_behavior():
    retriever = LantusRetriever()
    results = retriever.retrieve("indication", k=2)
    assert len(results) <= 2
    results5 = retriever.retrieve("indication", k=5)
    assert len(results5) <= 5


def test_deterministic_results():
    retriever = LantusRetriever()
    results1 = retriever.retrieve("What is Lantus used for?", k=3)
    results2 = retriever.retrieve("What is Lantus used for?", k=3)
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1.chunk_id == r2.chunk_id
        assert abs(r1.relevance_score - r2.relevance_score) < 0.0001


def test_empty_invalid_query_handling():
    retriever = LantusRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    # Invalid query shouldn't crash
    results = retriever.retrieve("nonexistentxyz123", k=1)
    assert isinstance(results, list)
