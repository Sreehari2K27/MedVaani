"""Base interfaces for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class LLMResult:
    """Structured interpretation result from LLM provider.

    The provider only returns interpretation; state changes remain in
    the AE engine and safety layer.
    """

    intent: str = "unknown"
    product: Optional[str] = None
    events: List[str] = field(default_factory=list)
    onset: Optional[str] = None
    seriousness: Optional[str] = None
    outcome: Optional[str] = None
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    missing_information: List[str] = field(default_factory=list)
    next_question: Optional[List[str]] = None  # [field_key, question] or None
    escalation: bool = False
    raw: Optional[Dict[str, Any]] = None


class LLMProviderError(Exception):
    """Base error for LLM provider failures (timeout, network, parse, validation)."""


class LLMValidationError(LLMProviderError):
    """Raised when a provider payload cannot be converted into a valid LLMResult."""


class LLMProvider(Protocol):
    """Protocol/interface for LLM providers."""

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        """Interpret user message and return structured result.

        Args:
            message: User input message
            context: Optional context (case state, etc.)

        Returns:
            Structured LLMResult - no state mutations
        """
        ...
