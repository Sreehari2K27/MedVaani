"""Mock LLM provider for deterministic testing."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .base import LLMProvider, LLMResult


class MockLLMProvider:
    """Deterministic mock provider with no network calls."""

    def __init__(self) -> None:
        pass

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        """Return deterministic interpretation based on message content."""
        if not message:
            return LLMResult(
                intent="unknown",
                missing_information=["product"],
                next_question=["product", "Which product was involved?"],
            )

        text = message.lower().strip()

        # Urgent/life-threatening patterns
        if re.search(r"severe difficulty breathing|unconscious|severe chest pain|severe bleeding", text):
            return LLMResult(
                intent="urgent_medical",
                escalation=True,
                extracted_fields={},
                missing_information=[],
            )

        # Personalized treatment/dosing
        if re.search(r"increase.*dose|stop taking|what dose should i take", text):
            return LLMResult(
                intent="treatment_query",
                escalation=False,
                extracted_fields={},
            )

        # Normal AE patterns
        ae_keywords = [
            "adverse event",
            "report adverse",
            "report side effect",
            "side effect",
            "reaction after taking",
            "had a reaction",
            "report this adverse",
            "severe headache",
            "vomiting",
            "headache",
        ]
        is_ae = any(k in text for k in ae_keywords) and ("lantus" in text or "taking" in text or "medication" in text or "report" in text)
        if is_ae:
            event = "adverse event"
            if "headache" in text:
                event = "headache"
            if "vomiting" in text:
                event = "vomiting"
            fields = {"adverse_event": event}
            if "lantus" in text:
                fields["product"] = "Lantus (fictional)"
            missing = []
            if "product" not in fields:
                missing.append("product")
            missing.extend(["onset_date", "seriousness", "outcome"])
            return LLMResult(
                intent="ae_report",
                product=fields.get("product"),
                events=[event],
                extracted_fields=fields,
                missing_information=missing,
                next_question=["product", "Which product was involved?"] if "product" not in fields else ["onset_date", "When did the adverse event start?"],
            )

        # Product info
        if "what is lantus" in text or "used for" in text or "grounding_instruction" in text:
            return LLMResult(
                intent="product_info",
                product="Lantus",
                extracted_fields={
                    "answer": "Lantus (insulin glargine) is a long-acting insulin analog indicated for improving glycemic control in adult and pediatric patients with type 1 diabetes mellitus and in adults with type 2 diabetes mellitus."
                },
            )

        # Default
        return LLMResult(
            intent="unknown",
            extracted_fields={},
            missing_information=[],
        )
