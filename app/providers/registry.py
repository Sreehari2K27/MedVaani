"""Provider registry for selection (mock → mistral → evon)."""

from __future__ import annotations

import os
from typing import Dict, Type

from .base import LLMProvider
from .mock import MockLLMProvider
from .mistral import MistralLLMProvider


class LLMProviderRegistry:
    """Registry to manage provider selection."""

    _providers: Dict[str, Type[LLMProvider]] = {
        "mock": MockLLMProvider,
        "mistral": MistralLLMProvider,
    }
    _default = "mock"

    @classmethod
    def register(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def get(cls, name: str | None = None) -> LLMProvider:
        name = name or cls._default
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        provider_cls = cls._providers[name]
        return provider_cls()

    @classmethod
    def set_default(cls, name: str) -> None:
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        cls._default = name

    @classmethod
    def get_default(cls) -> str:
        return cls._default


def get_default_provider() -> LLMProvider:
    return LLMProviderRegistry.get()


def set_default_provider(name: str) -> None:
    LLMProviderRegistry.set_default(name)
