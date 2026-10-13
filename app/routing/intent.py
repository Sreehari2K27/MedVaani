"""Deterministic intent routing between AE and MI workflows.

Composes the existing deterministic signals (SafetyLayer + MIGuardrail) with
a precedence that keeps AE reporting/safety behavior intact:

1. Urgent utterances always go to the AE workflow (it escalates internally).
2. Safety-classified product information goes to MI.
3. Safety-classified personalized treatment/dosing goes to AE (bounded refusal,
   preserving the AE case state) - existing tested behavior.
4. Question-posed MI-guardrail decisions go to MI: allowed product info is
   answered from the package insert; blocked dosing/medication-change questions
   get the MI bounded refusal instead of the AE workflow's non-answer.
5. Explicit AE reports stay in the AE workflow.
6. Everything else defaults to the AE workflow (today's default).

No LLM is required; this is pure and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.safety import SafetyLayer, SafetySignal
from app.mi.guardrails import MIGuardrail, MIRule


@dataclass(frozen=True)
class RouteDecision:
    """Result of routing a transcript to a workflow."""

    intent: str
    workflow: str  # "ae" | "mi"
    reason: str


_QUESTION_START = re.compile(
    r"(?i)^\s*(what|which|when|where|who|why|how|does|do|did|is|are|"
    r"can|could|should|would|will|may|might)\b"
)


def looks_like_question(text: str) -> bool:
    """Interrogative utterances (by '?' or question-initial word)."""
    if not text:
        return False
    return "?" in text or bool(_QUESTION_START.match(text))


_ALLOWED_MI = "product_info"
_MI_BLOCKED_CATEGORIES = {
    MIRule.BLOCKED_PERSONALIZED_DOSING,
    MIRule.BLOCKED_MEDICATION_CHANGE,
}


def route_transcript(text: str) -> RouteDecision:
    """Route a user transcript to either the AE or MI workflow."""
    if not text or not text.strip():
        return RouteDecision(intent="unknown", workflow="ae", reason="empty_transcript")

    # 1. Safety first (must be active on both workflows).
    safety = SafetyLayer.classify_input(text)
    if safety.signal is SafetySignal.URGENT_LIFE_THREATENING:
        return RouteDecision(intent="urgent_medical", workflow="ae", reason="safety_urgent")

    # 2. Product information via safety layer.
    if safety.signal is SafetySignal.PRODUCT_INFORMATION_REQUEST:
        return RouteDecision(intent=_ALLOWED_MI, workflow="mi", reason="safety_product_info")

    # 3. Personalized treatment/dosing -> AE bounded refusal (existing behavior).
    if safety.signal is SafetySignal.PERSONALIZED_TREATMENT_REQUEST:
        return RouteDecision(intent="treatment_query", workflow="ae", reason="safety_treatment_bounded")

    # 4. MI guardrail decisions for question-posed utterances.
    if looks_like_question(text):
        guard = MIGuardrail.check(text)
        if guard.rule is MIRule.ALLOWED_PRODUCT_INFO:
            return RouteDecision(intent=_ALLOWED_MI, workflow="mi", reason="mi_guardrail_product_info")
        if guard.rule in _MI_BLOCKED_CATEGORIES:
            return RouteDecision(intent="treatment_query", workflow="mi", reason="mi_guardrail_blocked_question")

    # 5. Explicit AE reporting stays in the AE workflow.
    if safety.signal is SafetySignal.NORMAL_AE_REPORTING:
        return RouteDecision(intent="ae_report", workflow="ae", reason="safety_ae_reporting")

    # 6. Default to the AE workflow (today's default).
    return RouteDecision(intent="unknown", workflow="ae", reason="default_ae_workflow")