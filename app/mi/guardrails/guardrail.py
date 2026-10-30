"""Deterministic MI guardrails (no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import re


class MIRule(str, Enum):
    ALLOWED_PRODUCT_INFO = "allowed_product_info"
    BLOCKED_PERSONALIZED_DOSING = "blocked_personalized_dosing"
    BLOCKED_MEDICATION_CHANGE = "blocked_medication_change"
    BLOCKED_PERSONALIZED_TREATMENT = "blocked_personalized_treatment"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class MIGuardrailResult:
    allowed: bool
    category: str
    rule: MIRule
    response_text: str
    reason: Optional[str] = None
    requires_retrieval: bool = False


_PERSONALIZED_DOSE_PATTERNS = re.compile(
    r"(?i)(how\s+many\s+units|units\s+should|dose\s+should\s+i\s+take|what\s+dose\s+(tonight|today|now)"
    r"|increase\s+.*dose|decrease\s+.*dose|change\s+.*dose|my\s+blood\s+sugar.*dose|sugar.*250.*dose"
    r"|what\s+should\s+i\s+do.*sugar|how\s+much.*lantus|dose\s+calculation)"
)

_MED_CHANGE_PATTERNS = re.compile(
    r"(?i)(stop\s+taking|should\s+i\s+stop|reduce\s+.*dose|switch\s+.*insulin|change\s+my\s+(insulin|medication))"
)

_PERSONALIZED_TREATMENT_PATTERNS = re.compile(
    r"(?i)(what\s+should\s+i\s+do|is\s+this\s+dangerous\s+for\s+me|what\s+treatment\s+should\s+i\s+take|treat\s+me|for\s+me)"
)

_PRODUCT_INFO_PATTERNS = re.compile(
    r"(?i)(what\s+is\s+lantus|used\s+for|indication|storage|stor(?:e|ed|ing)"
    r"|refrigerat|fridge|adverse\s+reaction|side\s+effect|common.*reaction"
    r"|how\s+to\s+store|be\s+stored|starting\s+dose|recommended\s+dose|package\s+insert"
    # General mechanism-of-action questions. Mirrors SafetyLayer's clause so
    # both classifiers agree that "How does Lantus SoloStar work?" is product
    # information (routing already sent it here; the MI chat gate must not
    # reject it as out of scope).
    r"|how\s+does\s+.*work|mechanism\s+of\s+action"
    # General manufacturer-instruction questions about the device/leaflet
    # ("How do I use the pen?", "How do I attach a needle?", "How do I
    # perform the safety test?"). The personalized dosing, medication-change
    # and personalized-treatment checks above run FIRST, so unsafe intents
    # are still blocked before this rule is ever consulted.
    r"|how\s+(?:to\s+use|do\s+i\b[^.?!]{0,80}"
    r"(?:use|pen\b|needle|attach|injec\w*|safety\s+test|prim\w*|cap|dispose|discard|remove)))"
    # General composition questions ("What is the active ingredient in
    # Lantus SoloStar?"). Question-shaped span only: the question word must
    # precede the ingredient/composition wording within one sentence, so a
    # statement mentioning "ingredients" outside a question stays out of
    # scope and the bare keyword is never an automatic allow.
    r"|what\b[^.?!]{0,40}\b(?:active\s+)?ingredients?\b|composition\s+of\b"
    r"|which\b[^.?!]{0,40}\b(?:active\s+)?ingredients?\b"
    # General efficacy questions ("Can Lantus SoloStar cure diabetes
    # permanently?"). Same question-span rule; no bare "cure" allow.
    r"|can\b[^.?!]{0,40}\bcured?\b|does\b[^.?!]{0,40}\bcured?\b"
    # Device/leaflet reuse questions ("Can I reuse the needle on my pen?",
    # "How many times can I reuse the needle?"). Same question-span rule -
    # the leaflet's needle instructions (the grounding passages) cover it;
    # "reuse" outside a question span is not product info.
    r"|can\b[^.?!]{0,40}\bre-?us"
)


class MIGuardrail:
    @staticmethod
    def check(question: str) -> MIGuardrailResult:
        if not question or not question.strip():
            return MIGuardrailResult(
                allowed=False,
                category="out_of_scope",
                rule=MIRule.OUT_OF_SCOPE,
                response_text="I can only answer general product information questions about Lantus from the package insert.",
                reason="Empty question",
            )

        q = question.strip()

        if _PERSONALIZED_DOSE_PATTERNS.search(q):
            return MIGuardrailResult(
                allowed=False,
                category="personalized_dosing",
                rule=MIRule.BLOCKED_PERSONALIZED_DOSING,
                response_text="MedVaani cannot provide individualized dosing instructions. Please consult a qualified healthcare professional for dosing decisions.",
                reason="Personalized dosing request",
            )

        if _MED_CHANGE_PATTERNS.search(q):
            return MIGuardrailResult(
                allowed=False,
                category="medication_change",
                rule=MIRule.BLOCKED_MEDICATION_CHANGE,
                response_text="MedVaani cannot recommend starting, stopping, switching, or changing your medication. Please consult a qualified healthcare professional.",
                reason="Medication change request",
            )

        if _PERSONALIZED_TREATMENT_PATTERNS.search(q):
            return MIGuardrailResult(
                allowed=False,
                category="personalized_treatment",
                rule=MIRule.BLOCKED_PERSONALIZED_TREATMENT,
                response_text="MedVaani cannot provide personalized medical advice or treatment recommendations. Please consult a qualified healthcare professional.",
                reason="Personalized treatment request",
            )

        if _PRODUCT_INFO_PATTERNS.search(q):
            return MIGuardrailResult(
                allowed=True,
                category="product_info",
                rule=MIRule.ALLOWED_PRODUCT_INFO,
                response_text="",
                requires_retrieval=True,
            )

        # Out of scope
        return MIGuardrailResult(
            allowed=False,
            category="out_of_scope",
            rule=MIRule.OUT_OF_SCOPE,
            response_text="I can only answer general product information questions about Lantus from the package insert.",
            reason="Out of scope",
        )
