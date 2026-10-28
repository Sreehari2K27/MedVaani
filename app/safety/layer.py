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
# The severity-bearing abdominal/stomach-pain vocabulary lives in
# _ABDOMINAL_SEVERITY_PATTERNS below (see _is_fresh_intake for why it is
# special-cased on a fresh case).
_URGENT_PATTERNS = re.compile(
    r"(?i)(severe\s+(difficulty\s+breathing|chest\s+pain|bleeding)"
    r"|difficulty\s+breathing"
    r"|unconscious|unconsciousness"
    r"|severe\s+chest\s+pain"
    r"|severe\s+bleeding"
    r"|chest\s+pain\s+(that\s+)?severe"
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
    r"|dil\s+ka\s+daura|mirgi|behosh|gambhir\s+allerg"
    # Romanized Hindi death reports (English-script, so \b guards are safe).
    r"|\bmar\s+gaya\b|\bmar\s+gayi\b|\bmar\s+gaye\b"
    # Bengali urgent symptoms. Bare Bengali-script tokens: no \b guards
    # (vowel signs are not word characters, mirroring the Devanagari rule);
    # post-match negation ("... না / নেই") handles head-final denial.
    r"|হার্ট\s+অ্যাট|স্ট্রোক|পক্ষাঘাত"
    r"|খিঁচুনি|মৃগী"
    r"|অজ্ঞান|অচেতন|জ্ঞান\s+হারা"
    r"|শ্বাস\s+নিতে\s+কষ্ট|শ্বাসকষ্ট|শ্বাস\s+বন্ধ"
    r"|অ্যানাফিল্যাক্সি|গুরুতর\s+অ্যালার্জি"
    r"|ওভারডোজ"
    r"|মারা\s+গে|মৃত্যু"
    # Gujarati urgent symptoms (same bare-token rule).
    r"|હાર્ટ\s+એટેક|હૃદયરોગનો\s+હુમલો"
    r"|પક્ષાઘાત|લકવો"
    r"|મરડી"
    r"|બેભાન|મૂર્છા"
    r"|શ્વાસ\s+લેવામાં\s+તકલીફ|શ્વાસ\s+લેવામાં\s+દિક્કત"
    r"|એનાફિલેક્સિ|ગંભીર\s+એલર્જી"
    r"|ઓવરડોઝ"
    r"|મૃત્યુ\s+પામ|મરી\s+ગય|મૃત્યુ\s+થય"
    # Kannada urgent symptoms (same bare-token rule).
    r"|ಹೃದಯಾಘಾತ|ಪಕ್ಷವಾತ|ಪಾರ್ಶ್ವವಾಯು"
    r"|ಅಪಸ್ಮಾರ|ಏಪು"
    r"|ಮೂರ್ಛೆ|ಪ್ರಜ್ಞೆತಪ್ಪ|ಪ್ರಜ್ಞೆ\s+ಇಲ್ಲದ"
    r"|ಉಸಿರು\s+ಕಟ್ಟ|ಉಸಿರಾಡಲು\s+ಆಗುತ್ತಿಲ್ಲ|ಉಸಿರಾಡಲು\s+ಆಗಲಿಲ್ಲ|ಉಸಿರಾಡಲು\s+ಕಷ್ಟ"
    r"|ಅನಾಫಿಲ್ಯಾಕ್ಸಿ|ತೀವ್ರ\s+ಅಲರ್ಜಿ"
    r"|ಓವರ್ಡೋಸ್"
    r"|ಮರಣ\s+ಹೊಂದಿದ|ಸತ್ತ|ಪ್ರಾಣ\s+ಕಳೆದುಕೊಂಡ"
    # Malayalam urgent symptoms (same bare-token rule).
    r"|ഹൃദയാഘാതം|പക്ഷാഘാതം"
    r"|അപസ്മാരം|വലിപ്പം"
    r"|ബോധരഹിത|ബോധം\s+പോയി"
    r"|ശ്വാസം\s+കിട്ടാതെ|ശ്വാസം\s+കുരുങ്ങി|ശ്വാസം\s+നിന്നുപോയി"
    r"|അനാഫിലാക്സി|കടുത്ത\s+അലർജി"
    r"|ഓവർഡോസ്"
    r"|മരിച്ച|മരണമടഞ്ഞ"
    # Marathi urgent symptoms (Devanagari, same bare-token rule as Hindi).
    r"|हृदयविकाराचा\s+झटका|हार्ट\s+अट"
    r"|लकवा|मिरगी"
    r"|बेशुद्ध|मूर्च्छा|भान\s+हरप"
    r"|श्वास\s+घेण्यास\s+त्रास|दम\s+घेण्यास\s+त्रास|श्वास\s+कठीण"
    r"|एनाफिलॅक्सि|गंभीर\s+अ[ॅऍ]?लर्जी"
    r"|ओव्हरडोझ"
    r"|मरण\s+पावल|मृत्यू\s+झाल"
    # Punjabi urgent symptoms (Gurmukhi, same bare-token rule).
    r"|ਦਿਲ\s+ਦਾ\s+ਦੌਰਾ|ਹਾਰਟ\s+ਅਟੈਕ"
    r"|ਸਟਰੋਕ|ਲਕਵਾ"
    r"|ਮਿਰਗੀ"
    r"|ਬੇਹੋਸ(?:਼|ਸ਼)?"
    r"|ਸਾਹ\s+ਲੈਣ\s+ਵਿੱਚ\s+ਤਕਲੀਫ਼|ਸਾਹ\s+ਲੈਣ\s+ਵਿੱਚ\s+ਦਿੱਕਤ|ਸਾਹ\s+ਰੁਕ"
    r"|ਐਨਾਫਿਲੈਕਸਿ|ਗੰਭੀਰ\s+ਐਲਰਜੀ"
    r"|ਓਵਰਡੋਜ਼?"
    r"|ਮਰ\s+ਗਿ|ਮੌਤ"
    # Tamil urgent symptoms (same bare-token rule).
    r"|மாரடைப்பு|பக்கவாதம்"
    r"|வலிப்பு"
    r"|மயக்கம(?:்|ா)|உணர்வற்ற"
    r"|மூச்சு\s+விட\s+முடியவில்லை|மூச்சுத்\s*திணறல்|மூச்சு\s+விட\s+சிரமம்"
    r"|அனாபிலாக்சி|கடும்\s+ஒவ்வாமை"
    r"|ஓவர்டோஸ்"
    r"|இறந்து|உயிரிழந்த"
    # Telugu urgent symptoms (same bare-token rule).
    r"|గుండెపోటు|పక్షవాతం"
    r"|మూర్ఛ"
    r"|స్పృహతప్ప|స్పృహ\s*లేకుండా"
    r"|ఊపిరి\s+ఆడక|శ్వాస\s+ఆడక|శ్వాస\s+తీసుకోలేక"
    r"|అనాఫిలాక్సి|తీవ్ర\s+అలర్జీ"
    r"|ఓవర్డోస్"
    r"|మరణించ|చనిపోయ|ప్రాణాలు\s+కోల్పో"
    r"|life[-\s]threatening"
    r"|emergency"
    r")"
)

# Severity-bearing abdominal/stomach-pain vocabulary, split out from
# _URGENT_PATTERNS because these phrases double as ordinary event-description
# wording in Hindi/Hinglish/English ("bahut pet dard karne laga hai"). They
# stay urgent in every situation EXCEPT as the opening event description of a
# fresh case (see _is_fresh_intake): there the person is reporting what
# happened, and severity is assessed at the designated severity step instead
# of intercepting the intake. Once any wording is recorded on the case, these
# phrases are urgent updates again, exactly as before. Standalone
# classification (no case) is unchanged. Severity is mandatory in every
# language: a bare "पेट दर्द" / "pet dard" or mild mention never matches.
_ABDOMINAL_SEVERITY_PATTERNS = re.compile(
    r"(?i)severe\s+(?:abdominal|stomach|belly)\s+pain"
    r"|(?:abdominal|stomach|belly)\s+pain\s+(?:that\s+)?severe"
    r"|गंभीर\s+पेट\s*दर्द|तेज़?\s+पेट\s*दर्द|बहुत\s+पेट\s*दर्द"
    r"|पेट\s+में\s+बहुत\s+दर्द|पेट\s*दर्द\s+(?:बहुत|तेज़?|गंभीर)"
    # Romanized Hindi (STT commonly emits these spellings).
    r"|(?:bahut|bohot|buhut|tez|zyada|gambhir|severe)\s+(?:pet|pait)\s+(?:mein\s+)?dard"
    r"|(?:pet|pait)\s+mein\s+(?:bahut|bohot|buhut|tez|zyada|gambhir)\s+dard"
    r"|(?:pet|pait)\s+(?:mein\s+)?dard\s+(?:bahut|bohot|buhut|tez|zyada|gambhir)"
    # Severe abdominal pain in the other supported languages. Severity is
    # mandatory in every language: a bare stomach-pain mention never matches.
    r"|கடுமையான\s+வயிற்று\s+வலி|வயிற்றில்\s+கடும்\s+வலி"
    r"|తీవ్రమైన\s+కడుపు\s+నొప్పి|కడుపులో\s+తీవ్ర\s+నొప్పి"
    r"|ತೀವ್ರ\s+ಹೊಟ್ಟೆ\s+ನೋವು|ಹೊಟ್ಟೆಯಲ್ಲಿ\s+ತೀವ್ರ\s+ನೋವು"
    r"|കടുത്ത\s+വയറുവേദന|വയറിൽ\s+കടുത്ത\s+വേദന"
    r"|गंभीर\s+पोटदुखी|पोटात\s+जोरदार\s+दुखण"
    r"|ગંભીર\s+પેટનો\s+દુખાવો|પેટમાં\s+ભારે\s+દુખાવો"
    r"|তীব্র\s+পেট\s+ব্যথা|পেটে\s+প্রচণ্ড\s+ব্যথা"
    r"|ਗੰਭੀਰ\s+ਪੇਟ\s+ਦਰਦ|ਪੇਟ\s+ਵਿੱਚ\s+ਬਹੁਤ\s+ਦਰਦ"
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
        # Standalone negators of the other supported languages. These are
        # script-distinct, so English matching is unaffected.
        "இல்லை",
        "இல்ல",
        "లేదు",
        "లేద",
        "కాదు",
        "ಇಲ್ಲ",
        "ಅಲ್ಲ",
        "ഇല്ല",
        "അല്ല",
        "नाही",
        "नाहीत",
        "નહીં",
        "નહી",
        "નથી",
        "ના",
        "না",
        "নেই",
        "নি",
        "ਨਹੀਂ",
        "ਨਹੀ",
        "ਨਾ",
    }
)
# Head-final negation: checked immediately AFTER a match as well, because
# Hindi and the other supported Indic languages typically place the negation
# after the noun ("दिल का दौरा नहीं हुआ", "গুরুতর অ্যালার্জি নেই",
# "గుండెపోటు లేదు"), which the clause-prefix check cannot see. English
# negation is deliberately not checked after a match, so English behaviour
# stays exactly as before.
_AFTER_NEGATION_TOKENS = frozenset(
    {
        "नहीं",
        "नही",
        "nahi",
        "nahin",
        "இல்லை",
        "లేదు",
        "లేద",
        "కాదు",
        "ಇಲ್ಲ",
        "ಅಲ್ಲ",
        "ഇല്ല",
        "അല്ല",
        "नाही",
        "नाहीत",
        "નહીં",
        "નહી",
        "નથી",
        "না",
        "নেই",
        "নি",
        "ਨਹੀਂ",
        "ਨਹੀ",
    }
)
# Punctuation stripped from tokens, including the Devanagari danda ("।") and
# a trailing full stop ("नहीं।", "nahi.").
_TOKEN_STRIP = "\"'()[]{},;:!?।."
_CLAUSE_BOUNDARY = re.compile(
    r"[.,;!?]|\b(?:but|however|except|though|although|yet)\b", re.IGNORECASE
)
_NEGATION_WINDOW_WORDS = 6
_AFTER_NEGATION_WINDOW_WORDS = 5


def _is_negated_at(text: str, match_start: int) -> bool:
    clause = _CLAUSE_BOUNDARY.split(text[:match_start])[-1]
    recent = clause.split()[-_NEGATION_WINDOW_WORDS:]
    return any(word.strip(_TOKEN_STRIP).lower() in _NEGATION_WORDS for word in recent)


def _is_negated_after_match(match: re.Match) -> bool:
    """Whether head-final negation directly follows the matched urgent phrase.

    Hindi and the other supported Indic languages are head-final:
    "दिल का दौरा नहीं हुआ", "হার্ট অ্যাটাক নেই", "గుండెపోటు లేదు" negate
    *after* the phrase, which the clause-prefix check cannot see. Only the
    Indic negation tokens are examined, so English text - where negation
    precedes the phrase - is unaffected.
    """
    following = match.string[match.end() :].split()[:_AFTER_NEGATION_WINDOW_WORDS]
    return any(
        token.strip(_TOKEN_STRIP).lower() in _AFTER_NEGATION_TOKENS
        for token in following
    )


def _has_unnegated_match(text: str, pattern: re.Pattern) -> bool:
    return any(
        not _is_negated_at(text, match.start())
        and not _is_negated_after_match(match)
        for match in pattern.finditer(text)
    )


def _is_fresh_intake(case: Any | None) -> bool:
    """Whether none of the person's wording has been recorded on ``case``.

    On a fresh case the FIRST message is the initial event description. When
    that opening statement carries severity-bearing abdominal wording (the
    report itself: "… bahut pet dard karne laga hai"), it must be captured
    verbatim and taken through the approved interview — severity is assessed
    at the designated severity step — instead of being intercepted as an
    in-progress urgent update. With no case context (standalone
    classification, routing, MI) or once any wording is recorded, behaviour
    is unchanged: these phrases classify as urgent exactly as before.
    """
    if case is None:
        return False
    return not getattr(case, "adverse_event", None) and not getattr(
        case, "unclassified_initial_input", None
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

        # Check for urgent/life-threatening first (highest priority). The
        # abdominal-severity class is urgent except as the fresh case's
        # initial event description (see _is_fresh_intake).
        if _has_unnegated_match(normalized, _URGENT_PATTERNS) or (
            _has_unnegated_match(normalized, _ABDOMINAL_SEVERITY_PATTERNS)
            and not _is_fresh_intake(case)
        ):
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
