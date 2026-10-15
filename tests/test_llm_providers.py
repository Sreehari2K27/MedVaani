"""Focused unit tests for LLM provider abstraction."""
from __future__ import annotations

import os
from typing import Any, Dict

import pytest

from app.providers import (
    LLMProvider,
    LLMResult,
    MockLLMProvider,
    MistralLLMProvider,
    LLMProviderRegistry,
)


def test_provider_interface():
    """Test that providers implement the expected interface."""
    mock = MockLLMProvider()
    result = mock.interpret("test message")
    assert isinstance(result, LLMResult)
    assert hasattr(result, "intent")
    assert hasattr(result, "product")
    assert hasattr(result, "events")
    assert hasattr(result, "onset")
    assert hasattr(result, "seriousness")
    assert hasattr(result, "outcome")
    assert hasattr(result, "extracted_fields")
    assert hasattr(result, "missing_information")
    assert hasattr(result, "next_question")
    assert hasattr(result, "escalation")


def test_deterministic_mock_response():
    """Test mock provider returns deterministic responses."""
    mock = MockLLMProvider()
    result1 = mock.interpret("I developed a headache after taking Lantus.")
    result2 = mock.interpret("I developed a headache after taking Lantus.")
    assert result1.intent == result2.intent
    assert result1.product == result2.product
    assert result1.intent == "ae_report"


def test_structured_result_validation():
    """Test LLMResult has correct structure."""
    result = LLMResult(
        intent="ae_report",
        product="Lantus (fictional)",
        events=["headache"],
        extracted_fields={"product": "Lantus (fictional)"},
        missing_information=["onset_date"],
        next_question=["onset_date", "When did it start?"],
        escalation=False,
    )
    assert result.intent == "ae_report"
    assert result.product == "Lantus (fictional)"
    assert isinstance(result.events, list)
    assert isinstance(result.extracted_fields, dict)
    assert isinstance(result.missing_information, list)
    assert result.escalation is False


def test_missing_mistral_api_key():
    """Test Mistral provider handles missing API key cleanly."""
    # Ensure no API key in env for this test
    old_key = os.environ.get("MISTRAL_API_KEY")
    try:
        if "MISTRAL_API_KEY" in os.environ:
            del os.environ["MISTRAL_API_KEY"]
        provider = MistralLLMProvider()
        with pytest.raises(ValueError, match="API key is missing"):
            provider.interpret("test")
    finally:
        if old_key:
            os.environ["MISTRAL_API_KEY"] = old_key


def test_provider_selection():
    """Test provider selection mechanism."""
    registry = LLMProviderRegistry()
    mock_provider = registry.get("mock")
    assert isinstance(mock_provider, MockLLMProvider)

    # Test default
    default = registry.get()
    assert isinstance(default, MockLLMProvider)

    # Test switching
    registry.set_default("mock")
    assert registry.get_default() == "mock"


def test_malformed_provider_output_handling():
    """Test that malformed outputs are handled appropriately."""
    # Mock provider should always return valid structure
    mock = MockLLMProvider()
    result = mock.interpret("")
    assert isinstance(result, LLMResult)
    assert result.intent is not None
    assert isinstance(result.extracted_fields, dict)
    assert isinstance(result.missing_information, list)
    assert isinstance(result.events, list)


def test_registry_can_switch_between_providers():
    """Test registry supports switching providers."""
    LLMProviderRegistry.set_default("mock")
    provider = LLMProviderRegistry.get()
    assert isinstance(provider, MockLLMProvider)


def test_registry_mistral_provider_available_without_calls():
    """The mistral registry entry instantiates but never calls the network."""
    provider = LLMProviderRegistry.get("mistral")
    assert isinstance(provider, MistralLLMProvider)
    assert provider.api_key is None or isinstance(provider.api_key, str)
