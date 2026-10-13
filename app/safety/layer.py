"""Deterministic safety classification layer (no LLM).

This layer screens user messages to detect:
- urgent/potentially life-threatening inputs (trigger HUMAN_REVIEW + high priority)
- personalized treatment/dosing requests (safe bounded refusal)
- product information requests (non-AE info)
- normal AE reporting (allow workflow to continue)

All logic is deterministic and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SafetySignal(str, Enum):
    """High-level safety classification signals."""

    URGENT_LIFE_THREATENING = "URGENT_LIFE_THREATENING"
    PERSONALIZED_TREATMENT_REQUEST = "PERSONALIZED_TREATMENT_REQUEST"
    PRODUCT_INFORMATION_REQUEST = "PRODUCT_INFORMATION_REQUEST"
    NORMAL_AE_REPORTING = "NORMAL_AE_REPORTING"
    UNKNOWN_AMBIGUOUS = "UNKNOWN_AMBIGUOUS"


@dataclass(frozen=True)
class SafetyResult:
    """Result of safety classification."""

    signal: SafetySignal
    message: str
    should_interrupt_ae: bool
    escalate_to_human_review: bool
    mark_high_priority: bool
    preserve_case_info: bool


# Patterns for urgent/life-threatening situations. Kept in one authoritative
# list so every workflow (voice and text) escalates on the same vocabulary.
_URGENT_PATTERNS = re.compile(
    r"(?i)(severe\s+(difficulty\s+breathing|chest\s+pain|bleeding"
    r"|(?:abdominal|stomach|belly)\s+pain)"
    r"|difficulty\s+breathing"
    r"|unconscious|unconsciousness"
    r"|severe\s+chest\s+pain"
    r"|severe\s+bleeding"
    r"|chest\s+pain\s+(that\s+)?severe"
    # Severe abdominal/stomach pain, mirroring the chest-pain forms above.
    # Severity is required: mild stomach discomfort never matches.
    r"|(?:abdominal|stomach|belly)\s+pain\s+(that\s+)?severe"
    r"|can'?t\s+breathe|hard\s+to\s+breathe"
    r"|not\s+breathing"
    r"|losing\s+consciousness"
    r"|loss\s+of\s+consciousness"
    r"|bleeding\s+(heavily|badly|severe)"
    r"|heart\s+attack|cardiac\s+arrest"
    r"|stroke"
    r"|seizure|convuls"
    r"|faint(?:s|ed|ing)|passed\s+out|unresponsive"
    r"|anaphyla|severe\s+allergic"
    r"|swell\w*[^.!?,;]{0,20}\b(?:face|lips?|tongue|throat)\b"
    r"|(?:facial|face|lips?|tongue|throat)(?:\s+(?:is|are|has|have))?\s+swell\w*"
    r"|throat\s+(?:is\s+)?closing"
    r"|overdose"
    # Explicit death reports. The existing clause-prefix negation window
    # cancels "I did not die" / "he didn't pass away" / "no one died".
    r"|\bdied\b|\bpassed\s+away\b"
    # Hindi urgent symptoms (Devanagari and unambiguous Romanized forms).
    # Bare Devanagari/Romanized tokens: no \b guards, because matras and the
    # chandrabindu are not word characters. Hindi negation follows the noun
    # ("दिल का दौरा नहीं हुआ", "dil ka daura nahi hua"), so negation of these
    # phrases is also checked immediately after the match.
    r"|दिल\s+का\s+दौरा|हृदय\s+आघात|हार्ट\s+अटैक"
    r"|स्ट्रोक|पक्षाघात"
    r"|मिर्गी"
    r"|बेहोश|अचेत"
    r"|एनाफिलैक्सि|गंभीर\s+एलर्जी"
    r"|ओवरडोज"
    # Hindi death reports (past-tense forms + the noun). Bare Devanagari: no
    # \b guards (matras); post-match negation covers "... नहीं".
    r"|मर\s+गया|मर\s+गई|मर\s+गए|मृत्यु"
    # Severe abdominal/stomach pain in Hindi. Severity is mandatory, exactly
    # like the English forms above: a bare "पेट दर्द" / "pet dard" mention is
    # never urgent. Post-noun negation ("पेट दर्द नहीं है", "pet dard nahi
    # hai") is cancelled by the existing Hindi after-match negation check.
    r"|गंभीर\s+पेट\s*दर्द|तेज़?\s+पेट\s*दर्द|बहुत\s+पेट\s*दर्द"
    r"|पेट\s+में\s+बहुत\s+दर्द|पेट\s*दर्द\s+(?:बहुत|तेज़?|गंभीर)"
    # Romanized Hindi (STT commonly emits these spellings).
    r"|(?:bahut|bohot|buhut|tez|zyada|gambhir|severe)\s+(?:pet|pait)\s+(?:mein\s+)?dard"
    r"|(?:pet|pait)\s+mein\s+(?:bahut|bohot|buhut|tez|zyada|gambhir)\s+dard"
    r"|(?:pet|pait)\s+(?:mein\s+)?dard\s+(?:bahut|bohot|buhut|tez|zyada|gambhir)"
    r"|dil\s+ka\s+daura|mirgi|behosh|gambhir\s+allerg"
    # Romanized Hindi death reports (English-script, so \b guards are safe).
    r"|\bmar\s+gaya\b|\bmar\s+gayi\b|\bmar\s+gaye\b"
    r"|life[-\s]threatening"
    r"|emergency"
    r")"
)

# Explicit denial of an urgent phrase ("I am not experiencing a
# life-threatening reaction") must not escalate. Only a negation cue sitting in
# the same clause and within a few words of the matched phrase cancels it, so
# genuine urgent statements are unaffected. Hindi negation (Devanagari and
# Romanized) is included for the same reason; it never appears in English text.
_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "without",
        "denies",
        "deny",
        "isn't",
        "isnt",
        "wasn't",
        "wasnt",
        "aren't",
        "arent",
        "don't",
        "dont",
        "didn't",
        "didnt",
        "can't",
        "cant",
        "won't",
        "wont",
        "hasn't",
        "hasnt",
        "haven't",
        "havent",
        "hadn't",
        "hadnt",
        "nor",
        "नहीं",
        "नही",
        "nahi",
        "nahin",
    }
)
# Hindi negation only: checked immediately AFTER a match as well, because
# Hindi places the negation after the noun. English negation is deliberately
# not checked after a match, so English behaviour stays exactly as before.
_HINDI_NEGATION_TOKENS = frozenset({"नहीं", "नही", "nahi", "nahin"})
# Punctuation stripped from tokens, including the Devanagari danda ("।") and
# a trailing full stop ("नहीं।", "nahi.").
_TOKEN_STRIP = "\"'()[]{},;:!?।."
_CLAUSE_BOUNDARY = re.compile(
    r"[.,;!?]|\b(?:but|however|except|though|although|yet)\b", re.IGNORECASE
)
_NEGATION_WINDOW_WORDS = 6
_HINDI_NEGATION_WINDOW_WORDS = 5


def _is_negated_at(text: str, match_start: int) -> bool:
    clause = _CLAUSE_BOUNDARY.split(text[:match_start])[-1]
    recent = clause.split()[-_NEGATION_WINDOW_WORDS:]
    return any(word.strip(_TOKEN_STRIP).lower() in _NEGATION_WORDS for word in recent)


def _is_hindi_negated_after(match: re.Match) -> bool:
    """Whether Hindi negation directly follows the matched urgent phrase.

    Hindi is head-final: "दिल का दौरा नहीं हुआ" / "dil ka daura nahi hua"
    negate *after* the phrase, which the clause-prefix check cannot see. Only
    the Hindi negation tokens are examined, so English text - where negation
    precedes the phrase - is unaffected.
    """
    following = match.string[match.end() :].split()[:_HINDI_NEGATION_WINDOW_WORDS]
    return any(
        token.strip(_TOKEN_STRIP).lower() in _HINDI_NEGATION_TOKENS
        for token in following
    )


def _has_unnegated_urgent_match(text: str) -> bool:
    return any(
        not _is_negated_at(text, match.start())
        and not _is_hindi_negated_after(match)
        for match in _URGENT_PATTERNS.finditer(text)
    )

# Patterns for personalized treatment/dosing requests
_DOSE_PATTERNS = re.compile(
    r"(?i)(increase\s+.*dose|decrease\s+.*dose|change\s+.*dose"
    r"|what\s+dose\s+(should|can|do)\s+I\s+(take|take\?|use|use\?)"
    r"|how\s+much\s+(should|can)\s+I\s+(take|use)"
    r"|dose\s+(increase|decrease|adjust|adjustment)"
    r"|stop\s+taking|should\s+I\s+stop\s+(taking)?"
    r"|should\s+I\s+(increase|decrease|change|adjust)\s+(my\s+)?(dose|medication|lantus|insulin)"
    r"|what\s+should\s+I\s+do\s+about\s+(my\s+)?(dose|dosing|medication)"
    r"|increase\s+my\s+(lantus\s+)?dose"
    r"|stop\s+(taking\s+)?lantus"
    r"|blood\s+sugar\s+high.*what\s+dose"
    r")"
)

# Patterns suggesting product information (not AE)
_PRODUCT_INFO_PATTERNS = re.compile(
    r"(?i)(how\s+does\s+.*work|side\s+effects\s+of|what\s+is\s+.*used\s+for"
    r"|indication|mechanism\s+of\s+action|storage|administration"
    r"|how\s+to\s+use|when\s+to\s+take|drug\s+interactions"
    r")"
)

# Patterns suggesting AE reporting
_AE_PATTERNS = re.compile(
    r"(?i)(headache|nausea|vomiting|rash|itch|swelling|dizziness|fatigue|fever|pain"
    r"|developed|experienced|had\s+.*after\s+taking|reaction|adverse\s+event"
    r"|side\s+effect|felt\s+.*after"
    r")"
)


class SafetyLayer:
    """Deterministic safety classifier."""

    @staticmethod
    def classify_input(text: str, case: Any | None = None) -> SafetyResult:
        """Classify input text and return safety result.

        Args:
            text: User input text to classify
            case: Optional AE case (preserved if urgent)
        """
        if not text or not text.strip():
            return SafetyResult(
                signal=SafetySignal.UNKNOWN_AMBIGUOUS,
                message="I'm not sure how to interpret that. Could you please clarify?",
                should_interrupt_ae=False,
                escalate_to_human_review=False,
                mark_high_priority=False,
                preserve_case_info=True,
            )

        normalized = text.strip()

        # Check for urgent/life-threatening first (highest priority)
        if _has_unnegated_urgent_match(normalized):
            return SafetyResult(
                signal=SafetySignal.URGENT_LIFE_THREATENING,
                message=(
                    "If this is a medical emergency, please seek immediate medical "
                    "attention or call your local emergency services. This appears "
                    "potentially urgent. A healthcare professional should be contacted "
                    "immediately."
                ),
                should_interrupt_ae=True,
                escalate_to_human_review=True,
                mark_high_priority=True,
                preserve_case_info=True,
            )

        # Check for personalized treatment/dosing requests
        if _DOSE_PATTERNS.search(normalized):
            return SafetyResult(
                signal=SafetySignal.PERSONALIZED_TREATMENT_REQUEST,
                message=(
                    "I cannot provide personalized dosing, treatment recommendations, "
                    "or medication changes. Please consult with a qualified healthcare "
                    "professional for any changes to your medication regimen."
                ),
                should_interrupt_ae=False,
                escalate_to_human_review=False,
                mark_high_priority=False,
                preserve_case_info=True,
            )

        # Check for product information requests
        if _PRODUCT_INFO_PATTERNS.search(normalized) and not _AE_PATTERNS.search(normalized):
            return SafetyResult(
                signal=SafetySignal.PRODUCT_INFORMATION_REQUEST,
                message="This appears to be a general product information request.",
                should_interrupt_ae=False,
                escalate_to_human_review=False,
                mark_high_priority=False,
                preserve_case_info=True,
            )

        # Check for normal AE reporting
        if _AE_PATTERNS.search(normalized):
            return SafetyResult(
                signal=SafetySignal.NORMAL_AE_REPORTING,
                message="Normal AE reporting - continuing workflow.",
                should_interrupt_ae=False,
                escalate_to_human_review=False,
                mark_high_priority=False,
                preserve_case_info=True,
            )

        # Default to unknown/ambiguous
        return SafetyResult(
            signal=SafetySignal.UNKNOWN_AMBIGUOUS,
            message="I'm not sure how to interpret that. Could you please clarify?",
            should_interrupt_ae=False,
            escalate_to_human_review=False,
            mark_high_priority=False,
            preserve_case_info=True,
        )

    @staticmethod
    def apply_safety_to_case(result: SafetyResult, engine: Any) -> None:
        """Apply safety result to an AE engine case.

        Args:
            result: Safety classification result
            engine: AEStateEngine instance
        """
        if result.mark_high_priority and hasattr(engine, "mark_urgent"):
            reason = result.message
            engine.mark_urgent(reason=reason)
