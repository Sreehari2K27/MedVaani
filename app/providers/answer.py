"""Deterministic answer-turn LLM provider for the AE workflow.

The base ``MockLLMProvider`` recognises only first-turn utterances (an adverse
event report, an urgent statement, a dosing question, or a product explanation).
This provider layers the missing multi-turn behaviour on top: when the base
provider cannot classify a message, it extracts the structured AE fields a plain
answer conveys (reporter role, patient age including written-out numbers,
patient identifier, phone/email, consent, onset, seriousness, outcome).

Design boundaries:
    - Intent stays unchanged. When fields are extracted from an answer, the
      base provider's intent (e.g. ``"unknown"``) is preserved verbatim, so the
      API response envelope and intent contract are not altered.
    - No state mutations here. The returned ``LLMResult`` is merged by the
      conversational workflow, which owns the state machine and guardrails.
    - Fully offline and deterministic; no network or model calls.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Optional

from .base import LLMResult
from .mock import MockLLMProvider
from app.workflow.onset import format_report_date

# Intents the base provider has already fully interpreted; the answer parser
# must not re-parse those messages.
_BASE_INTERPRETED = frozenset(
    {"ae_report", "product_info", "urgent_medical", "treatment_query"}
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")

# Final-connection-domain labels accepted for dictated/spelled addresses, so
# ordinary speech that merely contains "at" and "dot" can never become an
# address ("look at this dot on my arm" ends in "onmyarm", which is not a
# recognised top-level/domain label).
_EMAIL_TLDS = frozenset(
    {
        "com", "net", "org", "co", "in", "edu", "gov", "mil", "int", "io", "ai",
        "me", "info", "biz", "uk", "us", "ca", "au", "de", "fr", "nl", "jp",
        "cn", "br", "za", "sg", "ae", "tv", "xyz", "online", "site", "tech",
    }
)

# Provider/domain words that legitimately appear inside a dictated address.
_EMAIL_DICTATION_WORDS = frozenset(
    {
        "gmail", "googlemail", "yahoo", "ymail", "hotmail", "outlook",
        "protonmail", "proton", "icloud", "mail", "email", "live",
        "rocketmail", "zoho", "aol", "msn",
    }
)

# Malayalam letter-name/dictation tokens the browser STT emits for a spoken
# email dictation ("ആർ എ ജെ യു പി കെ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
# -> "rajupk@gmail.com"). Each token maps to its intended Latin spelling;
# the assembled address must still fully validate (_EMAIL + known TLD), so a
# wrong letter can never be accepted. Only tokens evidenced by the transcript
# are listed; other letter names stay unmapped rather than guessed.
_MALAYALAM_EMAIL_TOKENS: Dict[str, str] = {
    "ആർ": "r",
    "എ": "a",
    "എസ്": "s",
    "ഇ": "e",
    "എച്ച്": "h",
    "ഐ": "i",
    "ജെ": "j",
    "ജി": "g",
    "യു": "u",
    "പി": "p",
    "കെ": "k",
    "മെയിൽ": "mail",
    "ജിമെയിൽ": "gmail",
    # The observed Devanagari-transcribed letter names and email
    # terms (the browser STT wrote the spoken English letter names
    # into Devanagari script): attested variants only.
    "एस": "s",
    "आर": "r",
    "ई": "e",
    "एच": "h",
    "जीमेल": "gmail",
    "कॉम": "com",
    "കോം": "com",
}

# The Malayalam pronunciation of the "@" cue ("അറ്റ് ദ റേറ്റ്" = "at the
# rate") and the spoken negation leads that may precede a dictated address
# ("ഇല്ല ആർ …" means "no — it is  …", a correction, never a decline).
_MALAYALAM_AT_RATE = re.compile(
    r"അറ്റ്\s+ദ\s+റേറ്റ്"
    r"|एट\s+द\s+रेट"
    r"|अॅट\s+द\s+रेट"
)
_MALAYALAM_EMAIL_NEGATION = re.compile(r"^\s*(?:ഇല്ല|അല്ല|അതല്ല)[\s,]*")
# Deterministic dictation forms of an email address: a short known lead-in is
# dropped first, "at the rate" / a standalone "at" dictate "@", a standalone
# "dot" dictates ".", and spacing directly around those components is removed.
_EMAIL_LEAD = re.compile(
    r"(?i)^\s*(?:"
    r"(?:my\s+)?(?:e-?mail(?:\s+address)?|mail(?:\s+address)?|address)"
    r"\s*(?:is|:)\s*"
    r"|(?:e-?mail|reach|contact)\s+me\s+at\s+"
    r"|it'?s\s+"
    r"|that'?s\s+"
    r")"
)
_SPOKEN_AT_RATE = re.compile(r"\bat\s+the\s+rate\b", re.IGNORECASE)
_SPOKEN_AT = re.compile(r"\bat\b", re.IGNORECASE)
_SPOKEN_DOT = re.compile(r"\bdot\b", re.IGNORECASE)
_EMAIL_COMPONENT_SPACES = re.compile(r"\s*([@.])\s*")
_PHONE = re.compile(r"^\+?[\d][\d\s\-().]{6,}\d$")
_DIGITS = re.compile(r"\d+")
# Affirmations and negations in English and Hindi (Devanagari and Romanized),
# shared by pending-field extraction (consent, seriousness), the conversational
# confirmation replies, and every other caller of these two patterns, so Hindi
# answers follow exactly the same deterministic paths as English ones. Latin
# tokens keep their \b guards; Devanagari tokens are matched bare because matras
# and the chandrabindu are not word characters, so a trailing \b would never
# match ("हाँ ", "नहीं,"). The two token sets do not overlap: the Hindi negation
# forms never contain an affirmative form and vice versa.
_NEGATION = re.compile(
    r"\b(no|not|never|none|neither|without|denies|deny|refuse|refuses|"
    r"do not|don't|did not|didn't|was not|wasn't|is not|isn't|cannot|can't"
    r"|nahi|nahin)\b"
    r"|नहीं|नही"
    # Standalone negators of the other supported languages. Bare Indic
    # tokens (vowel signs break \b, same rule as the Devanagari forms above);
    # script-distinct, so they never overlap with _AFFIRM.
    r"|ഇല്ല|അല്ല|അതല്ല|இல்லை|కాదు|లేదు|ಇಲ್ಲ|नाही|ના|નહીં|না|নয়|ਨਹੀਂ|ਨਾ"
    r"|രണ്ടും\s+സെറ്റാക്കും",
    re.IGNORECASE,
)
_AFFIRM = re.compile(
    r"\b(yes|yep|yeah|yup|sure|ok|okay|fine|correct|true|affirmative|definitely|"
    r"absolutely|of course|that's right|thats right|please do"
    r"|ji haan|haan|han"
    # Hindi affirmations, parallel to the English ones: "okay" -> "theek
    # hai", "agree" -> "sahmat", "absolutely" -> "bilkul", "correct" ->
    # "sahi hai", "certainly" -> "jarur". A following negation ("sahi
    # nahi", "jarur nahi") still wins via the negation rules.
    r"|theek\s+hai|thik\s+hai|sahmat|bilkul"
    r"|sahi\s+hai|sahi|jarur|zarur)\b"
    r"|जी हाँ|जी हां|हाँ|हां|ठीक\s+है|सहमत|बिल्कुल"
    r"|सही\s+है|सही|जरूर|ज़रूर"
    # Standalone affirmatives of the other supported languages (bare Indic
    # tokens / \s+ compounds, same rule as the Devanagari forms above; the
    # two token sets never overlap with _NEGATION). "യെസ്"/"യസ്" are the
    # Malayalam-script transliterations of spoken "yes" that the browser
    # STT emits. "hoy"/"ho"/"barobar"/"aahe" are the Latin-script
    # transliterations of the Marathi affirmatives (word-boundary guarded,
    # so English words containing them never match).
    r"|അതെ|ശരി|ശരിയാണ്|യെസ്|യസ്|ஆம்|சரி|అవును|సరే|ಹೌದು|ಸರಿ|होय|बरोबर|હા|સાચું"
    r"|\bhoy\b|\bho\b|\bbarobar\b|\baahe\b"
    r"|হ্যাঁ|ঠিক\s+আছে|ਹਾਂ|ਠੀਕ\s+ਹੈ",
    re.IGNORECASE,
)
# Only explicit "serious" wording maps to the seriousness claim. Whether the
# person sought medical care is tracked separately in ``medical_care`` so that
# "I did not go to the hospital" is never read as "not serious".
_SERIOUS_TOPIC = re.compile(
    r"\b(serious|seriously|seriousness|life[-\s]?threatening)\b",
    re.IGNORECASE,
)
# A negation that directly denies the seriousness claim ("it is not serious",
# "no, it's not serious", "not ... life-threatening") is a valid "no".
# Anything else is not treated as a denial.
_SERIOUS_NEGATED = re.compile(
    r"\b(?:not|never|isn'?t|wasn'?t|ain'?t)\b[^.!?]{0,25}"
    r"\b(?:serious(?:ly|ness)?|life[-\s]?threatening)\b",
    re.IGNORECASE,
)
_MEDICAL_CARE_YES = re.compile(
    r"\b(?:went|been|admitted|taken|rushed)\s+to\s+(?:the\s+)?"
    r"(?:hospital|er|a&e|emergency(?:\s+room)?|clinic)\b"
    r"|\bsaw\s+(?:a|the|my)\s+(?:doctor|physician|nurse|gp)\b"
    r"|\bvisited\s+(?:the\s+)?(?:doctor|hospital|clinic|er)\b"
    r"|\breceived\s+medical\s+(?:care|attention|treatment|help)\b"
    r"|\bcalled\s+(?:an?\s+)?ambulance\b"
    r"|\bhospitali[sz]ed\b|\bwent\s+to\s+see\s+a\s+doctor\b",
    re.IGNORECASE,
)
_MEDICAL_CARE_NO = re.compile(
    r"\b(?:didn'?t|did\s+not|haven'?t|have\s+not|never|no|not)\b[^.!?]{0,30}"
    r"\b(?:go\s+to|gone\s+to|been\s+to|visit|see|saw|receive|seek|contact|call)\b"
    r"[^.!?]{0,20}\b(?:hospital|er|a&e|emergency|doctor|physician|clinic|medical|ambulance)\b"
    r"|\bno\s+medical\s+(?:care|attention|help)\b"
    r"|\bnot\s+hospitali[sz]ed\b",
    re.IGNORECASE,
)
# An existing medical-care "yes" (hospitalization/medical attention reported)
# may only be replaced by a weaker answer when the reporter explicitly signals
# that they are correcting that earlier statement. Plain later mentions never
# silently downgrade it.
_EXPLICIT_CORRECTION = re.compile(
    r"\b(?:actually|i\s+mean(?:\s+to\s+say)?|correction|to\s+correct|"
    r"on\s+second\s+thought|let\s+me\s+correct|previous\s+(?:answer|statement|reply)|"
    r"i\s+(?:was|must\s+be)\s+wrong)\b",
    re.IGNORECASE,
)
# Known product variants that deterministic STT commonly mangles. Canonicalised
# so "lantern" and "lantis" both resolve to the real product name. "Solar star"
# compounds only count together with the Lantus/Lanta stem, so an unrelated
# "solar star" product is never reclassified; the Devanagari sub-brand
# ("सोलो स्टार") is recognised the same way.
_PRODUCT_VARIANTS = re.compile(
    r"\b(lantern|lantus|lantis|lantos|lanthus|lan\s+tus|landus|"
    r"solo\s*star|solostar|zolo\s*star|zolostar"
    r"|lanta\s+sol(?:o|ar)\s+star|lantus\s+sol(?:o|ar)\s+star)\b"
    r"|सोलो\s*स्टार"
    # Narrow Devanagari transliterations of "Lantus" as STT emits them;
    # exact aliases only, no fuzzy matching (matras break \b, so bare
    # Devanagari like the सोलो स्टार token above).
    r"|लेंटस|लैंटस|लैन्टस|लेन्टस"
    # Observed Hindi STT variants of the brand stem ("लैंटर" never matched on
    # its own; "लैंडर सोलो स्टार" reached Step 4 only via the sub-brand).
    r"|लैंटर|लैंडर"
    # Conservative exact native-script brand aliases for the other supported
    # locales: bare Indic tokens (vowel signs break \b), \s* spacing inside
    # the sub-brand compounds, no fuzzy matching — a garbled word (e.g. the
    # Malayalam "കൊതുക") is never aliased.
    r"|ল্যান্টাস|সোলো\s*স্টার"
    r"|લાન્ટસ|લેન્ટસ|સોલો\s*સ્ટાર"
    r"|ಲ್ಯಾಂಟಸ್|ಸೋಲೋ\s*ಸ್ಟಾರ್"
    r"|ലാൻഡസ്|ലാൻറസ്|ലാൻടസ്|ലാന്ടസ്|ലാന്ദസ്|ലാറ്റസ്|സോ[ലള]ോ\s*സ്റ്റാർ"
    r"|സോളോസ്റ്റാ|സോലോസ്റ്റാ"
    r"|लँटस"
    r"|ਲੈਂਟਸ|ਸੋਲੋ\s*ਸਟਾਰ"
    r"|லான்டஸ்|லானடஸ்|சோலோ\s*ஸ்டார்"
    r"|లాంటస్|సోలో\s*స్టార్",
    re.IGNORECASE,
)
_PRODUCT_CANONICAL = "Lantus SoloStar"

# Known non-Lantus medicines the prototype recognises conservatively for the
# suspect-product answer and for first-report causation statements. Variant
# patterns map to a canonical name; the original event description is never
# altered and the medicine is never re-attributed to Lantus.
_OTHER_MEDICINE_VARIANTS = {
    "paracetamol": re.compile(
        r"\b(?:paracetamol|paraacetamol|paracetomol)\b"
        r"|पैरासिटामोल|पैरासीटामोल|पेरासिटामोल",
        re.IGNORECASE,
    ),
}
_OTHER_MEDICINE_CANONICAL = {"paracetamol": "Paracetamol"}

# Replies containing letters of the supported Indic scripts (Devanagari,
# Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam). Such a
# reply that names no recognised medicine is unclear speech — never a
# confirmed different medicine — and must be clarified instead of being
# captured as a product value.
_INDIC_SCRIPT_REPLY = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    r"\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]"
)

# Cues that name a medicine as the suspected cause of the reaction
# ("due to paracetamol", "पैरासिटामोल के कारण").
_SUSPECT_MEDICINE_CUE = re.compile(
    r"\b(?:due\s+to|because\s+of|caused\s+by)\b"
    r"|के\s+कारण|की\s+वजह\s+से|\bwajah\s+se\b|\bse\s+hua\b",
    re.IGNORECASE,
)


def _known_other_medicine(text: str) -> Optional[str]:
    """Canonical name of a recognised non-Lantus medicine in ``text``."""
    for canonical_key, pattern in _OTHER_MEDICINE_VARIANTS.items():
        if pattern.search(text or ""):
            return _OTHER_MEDICINE_CANONICAL[canonical_key]
    return None
# First-report language: taking/using a product or noticing a reaction.
_REPORT_LANGUAGE = re.compile(
    r"\b(i\s+took|i'?ve\s+taken|i\s+have\s+taken|i\s+used|i\s+injected|"
    r"after\s+taking|after\s+i\s+took|since\s+taking|started\s+taking|"
    r"side\s+effect|adverse\s+event|bad\s+reaction|had\s+a\s+reaction)\b"
    # Narrow English/Hindi/Hinglish report cues (audit step 2). Latin cues
    # keep \b guards; Devanagari stays bare because matras break \b.
    r"|\b(?:injection|davai|dawa|dawai|ke\s+baad|le\s+li(?:ya)?|lagaya)\b"
    r"|दवा|दवाई|इंजेक्शन|ले\s+लिया|लगाया|के\s+बाद",
    re.IGNORECASE,
)
_SYMPTOM_LANGUAGE = re.compile(
    r"\b(headache|nausea|nauseous|vomit\w*|rash|hives|swell\w*|swollen|dizz\w*|"
    r"fever|pain\w*|bleed\w*|cramp\w*|diarrh\w*|itch\w*|sore\w*|tired|fatigue|"
    r"breath\w*|blurr\w*|symptom\w*|reaction|ache|aches)\b"
    # Romanised Hindi symptoms (unambiguous; \b-guarded so English words like
    # "ultimately" can never match "ulti").
    r"|(?:dard|ulti|khujli|bukhar|chakkar|sujan|thakan)\b"
    # Devanagari symptoms, matched bare: words may end in a matra, which is
    # not a word character, so they cannot sit inside a \b-guarded group.
    r"|दर्द|उल्टी|खुजली|बुखार|चक्कर|सूजन|थकान",
    re.IGNORECASE,
)
# Plain non-answers that must never be stored as free-text content.
_NON_ANSWER = re.compile(
    r"^\s*(?:nothing\s+else|nothing|that'?s\s+all|that\s+is\s+all|n/?a|"
    r"not\s+applicable|no\s+thanks|none|got\s+it|i'?m?\s+providing\s+you\s+the\s+information)\s*[.!]?$",
    re.IGNORECASE,
)
_SIMPLE_YESNO = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|no|nope|ok|okay|sure|fine|correct|right|"
    r"that'?s\s+right)\s*[.!]?$",
    re.IGNORECASE,
)
_UNKNOWN_ANSWER = re.compile(
    r"\b(?:i\s+)?(?:don'?t|do\s+not|cannot|can'?t|can\s+not)\s+"
    r"(?:know|remember|recall)\b|\bnot\s+sure\b|\bunsure\b|\bno\s+idea\b"
    # Hindi / Romanized-Hindi "don't know" and "don't remember".
    r"|pata\s+nahi|nahi\s+pata|yaad\s+nahi|nahi\s+yaad"
    r"|पता\s+नहीं|नहीं\s+पता|याद\s+नहीं|नहीं\s+याद"
    # "Don't know" in the other supported languages (bare Indic tokens /
    # \s+ phrases, no \b for Indic scripts).
    r"|അറിയില്ല|தெரியல|తెలియదు|ಗೊತ್ತಿಲ್ಲ"
    r"|माहीत\s+नाही|ખબર\s+નથી|জানি\s+না|ਪਤਾ\s+ਨਹੀਂ",
    re.IGNORECASE,
)
# Expressed uncertainty ("maybe", "not sure", "hard to say") must never be
# silently stored as a definitive answer such as seriousness "no". Hindi:
# "shayad" (maybe) and "pata nahi" (don't know).
_UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|possibly|uncertain|unsure|not\s+sure|"
    r"don'?t\s+know|do\s+not\s+know|hard\s+to\s+say|can'?t\s+say|cannot\s+say|"
    r"i\s+guess|kind\s+of|sort\s+of)\b"
    r"|shayad\b|pata\s+nahi|nahi\s+pata|शायद|पता\s+नहीं|नहीं\s+पता",
    re.IGNORECASE,
)
_CONTACT_WORDS = re.compile(
    r"\b(?:phone|email|e-mail|number|call|contact|reach|mobile|telephone|cell)\b",
    re.IGNORECASE,
)
# Hindi seriousness wording for the *pending* seriousness question only:
# "गंभीर है" / "gambhir hai" (it is serious). Deliberately not added to
# _SERIOUS_TOPIC, so opportunistic extraction from unrelated Hindi sentences
# (e.g. "gambhir allergy") stays unchanged.
_HINDI_SERIOUS = re.compile(r"\bgambhir\b|गंभीर", re.IGNORECASE)
# A reply asking to repeat or expressing confusion must never be stored as a
# field value ("dobara kahiye" is a request to repeat, not a name or event).
_CLARIFY_REPLY = re.compile(
    r"\b(?:dobara|samajh\s+nahi\s+aaya|kya|kyaa)\b"
    r"|दोबारा|समझ\s+नहीं\s+आया|क्या",
    re.IGNORECASE,
)
# An obvious question is never an adverse-event description: an MI-style
# question that reaches the AE interview (or a request to repeat) must be
# re-asked, never stored as the reporter's own wording. "?" catches typed
# questions; the anchored word list catches dictated ones whose transcript
# carries no question mark - the same question-initial vocabulary the intent
# router uses, plus explicit repeat/clarify requests.
_QUESTION_INITIAL = re.compile(
    r"^\s*(?:what|which|when|where|who|why|how|does|do|did|is|are|can|could|"
    r"should|would|will|may|might|"
    r"please\s+(?:repeat|clarify|say\s+(?:that\s+)?again))\b",
    re.IGNORECASE,
)
# Negated auxiliaries and date-led openings are statements, not questions
# ("do not report this", "is not working", "May 12 I felt a rash").
_QUESTION_NOT_STATEMENT = re.compile(r"^\s*not\b", re.IGNORECASE)
_DATE_LED = re.compile(r"^\s*\d", re.IGNORECASE)


def _is_obvious_question(text: str) -> bool:
    """Whether ``text`` is plainly a question rather than a statement."""
    stripped = text or ""
    if "?" in stripped:
        return True
    match = _QUESTION_INITIAL.match(stripped)
    if match is None:
        return False
    remainder = stripped[match.end() :]
    if _QUESTION_NOT_STATEMENT.match(remainder) or _DATE_LED.match(remainder):
        return False
    return True

_ROLE_SELF = re.compile(
    r"\b(?:i am|i'm|im|myself)\b[^.]{0,25}\bpatient\b|\bpatient\s+is\s+me\b|\bmyself\b"
    # Colloquial self-answers to the reporter-role question: "yes, it's me",
    # "that's me", "my own experience", "this happened to me". Checked after
    # _ROLE_OTHER, so "no, it's my wife" is still someone_else.
    r"|\b(?:yes[,\s]+)?(?:it'?s|its|that'?s|thats)\s+me\b"
    r"|\bmy\s+own\s+(?:experience|side\s+effect|reaction|problem|issue)\b"
    r"|\bhappened\s+to\s+me\b"
    # Hindi / Romanized-Hindi self-answers: "happened to me", "myself", "my
    # own experience". The bare "it's me" forms ("main hoon", "मैं हूँ") are
    # anchored to the start of the reply so a casual mention inside a longer
    # sentence is never over-read.
    r"|\b(?:mere\s+saath\s+(?:hi\s+)?hua|mujhe\s+(?:hi\s+)?hua|main\s+khud|mai\s+khud)\b"
    r"|\bmera\s+apna\s+(?:anubhav|experience)\b"
    # The reported Devanagari self-answers: "यह मेरे साथ ही हुई घटना है"
    # ("this happened to me myself" — the dative structure, mirroring the
    # Malayalam structural pattern) and "ये मेरा अब खुद का अनुभव है" ("this
    # is my own experience" — the possessive "मेरा/मेरी/मेरे" with one
    # optional intervening word before "खुद का अनुभव"). Bare Indic tokens
    # (no \b for Indic scripts).
    r"|(?:\u092f\u0939|\u092f\u0947|\u092f\u0939\s+\u0918\u091f\u0928\u093e)?\s*"
    r"\u092e\u0947\u0930\u0947\s+\u0938\u093e\u0925(?:\s+\u0939\u0940)?\s+"
    r"\u0939\u0941(?:\u0908|\u0906)"
    r"|(?:\u092e\u0947\u0930\u093e|\u092e\u0947\u0930\u0940|\u092e\u0947\u0930\u0947)"
    r"\s+(?:\S+\s+)?\u0916\u0941\u0926\s+\u0915\u093e\s+\u0905\u0928\u0941\u092d\u0935"
    # "I'm the one experiencing/having it" (the verb list is narrow on
    # purpose: "I'm the one reporting for my wife" stays OTHER).
    r"|\b(?:i\s+am|i'?m|im)\s+the\s+one(?:\s+who)?(?:'s|\s+is|\s+was)?\s+"
    r"(?:experienc\w*|having|facing|going\s+through|suffering)"
    # "the patient who experienced this event" (any leading filler, e.g.
    # "and the patient who experienced this event").
    r"|\bpatient\s+who\s+experienced\b"
    # "I experienced it / I experienced this".
    r"|\bi\s+experienced\s+(?:it|this|firsthand|anything)\b"
    # "it is I myself": "main hi patient hoon", "नहीं मैं ही पेशेंट हूँ।".
    r"|\b(?:main|mai)\s+hi\s+(?:patient|h(?:oon|u|n))\b"
    # "it's my own experience": "mera hi anubhav", "apna hi anubhav".
    r"|\b(?:mera|apna)\s+hi\s+(?:anubhav|experience)\b"
    r"|मेरे\s+साथ\s+(?:ही\s+)?हुआ|मुझे\s+(?:ही\s+)?हुआ|मैं\s+खुद|मेरा\s+अपना\s+अनुभव"
    r"|मैं\s+ही\s+(?:पेशेंट|मरीज|हूँ|हूं)"
    r"|(?:मेरा|अपना)\s+ही\s+अनुभव|यही\s+अनुभव"
    r"|\byahi\s+anubhav\b"
    r"|^\s*(?:ji\s+)?(?:haan\s*,?\s+)?(?:main|mai)\s+(?:hoon|hu)\b"
    r"|^\s*(?:मैं|मै)\s+(?:हूँ|हूं)\s*[.!?]?\s*$"
    # First-person / own-experience cues for the other supported languages
    # (bare Indic tokens, conservative exact phrases, the emphatic
    # "me myself" and "my own experience" idioms, and the structural
    # "happened to me" pattern).
    r"|എനിക്ക്\s+തന്നെ"
    # Chillu-spelling alternates: browser STT also emits എൻ്റെ (chillu ൻ,
    # as in the audited combined name/DOB transcript), and the name-lead
    # stripper already accepts both spellings - the role classifier must
    # not know only one of them.
    r"|(?:എന്റെ|എൻ്റെ)\s+സ്വന്തം\s+അനുഭവ"
    r"|(?:എന്റെ|എൻ്റെ)\s+അനുഭവ"
    r"|(?:എന്റെ|എൻ്റെ)\s+തന്നെ\s+അനുഭവ"
    r"|എനിക്കാണ്\s+സംഭവിച്ചത്|എനിക്ക്\s+ആണ്\s+സംഭവിച്ചത്"
    # "It happened to me": the dative/accusative pronoun directly followed
    # (optionally with the emphatic തന്നെ/തന്നെയാണ് or ആണ് between) by the
    # happened-stem സംഭവിച്ച- covers every verb ending without enumerating
    # words. Adjacency keeps a later mention of another person unclassified
    # (the pronoun is no longer next to the verb), and _ROLE_OTHER is still
    # checked first for mixed answers.
    r"|എനിക്ക്(?:\s+തന്നെ(?:യാണ്)?|\s+ആണ്)?\s+സംഭവിച്ച"
    r"|എന്നെ(?:\s+തന്നെ(?:യാണ്)?|\s+ആണ്)?\s+സംഭവിച്ച"
    r"|എനിക്കാണ്\s+സംഭവിച്ച"
    # "The experience I had" (എനിക്കുണ്ടായ … അനുഭവം) and the emphatic
    # first person "ഞാൻ തന്നെ"; the bare "ഞാനാണ്" is anchored to the whole
    # reply (mirroring the anchored Hindi "मैं हूँ") so a mention inside a
    # longer sentence is never over-read.
    r"|എനിക്കുണ്ടായ"
    r"|ഞാൻ\s+തന്നെ(?:യാണ്)?"
    r"|^\s*ഞാനാണ്\s*[.!?]?\s*$"
    r"|எனக்கு\s+தான்|என்னுடைய\s+சொந்த\s+அனுபவம்"
    r"|నాకే|నా\s+సొంత\s+అనుభవం"
    r"|ನನಗೆ\s+ಸಂಬಂಧಿಸಿದ|ನನ್ನ\s+ಸ್ವಂತ\s+ಅನುಭವ"
    r"|আমার\s+নিজের\s+অভিজ্ঞতা|আমার\s+সাথেই"
    r"|મને\s+જ|મારો\s+પોતાનો\s+અનુભવ"
    r"|मलाच|माझा\s+स्वतःचा\s+अनुभव|मला\s+स्वत"
    r"|ਮੈਨੂੰ\s+ਹੀ|ਮੇਰਾ\s+ਆਪਣਾ\s+ਤਜਰਬਾ",
    re.IGNORECASE,
)
_ROLE_OTHER = re.compile(
    r"\b(someone else|not the patient|my\s+(?:wife|husband|son|daughter|mother|father|"
    r"mom|dad|parent|caregiver|pharmacist|doctor|nurse|relative|friend|partner|colleague))\b"
    # "on behalf of someone else", "reporting this for another person".
    r"|\b(?:on\s+behalf\s+of|for)\s+another\s+(?:person|individual|patient)\b"
    r"|\breporting\s+(?:this\s+)?for\s+(?:another|someone\s+else)\b"
    # Hindi / Romanized-Hindi someone-else answers: possessive + kinship
    # (including mixed "meri wife"), the "reporting for someone else"
    # phrasings, and the bare "someone else" form anchored to the whole reply.
    r"|(?:mera|meri|mere)\s+(?:pati|patni|beta|beti|ladka|ladki|maa|mummy|papa|pita|"
    r"dost|rishtedaa?r|bhai|behen|wife|husband|son|daughter|friend)\b"
    r"|मेरी\s+पत्नी|मेरे\s+पति|मेरा\s+बेटा|मेरी\s+बेटी|मेरा\s+लड़का|मेरी\s+लड़की"
    r"|मेरी\s+माँ|मेरे\s+पिता|मेरे\s+दोस्त|मेरे\s+रिश्तेदार|मेरे\s+भाई|मेरी\s+बहन"
    r"|\bdoosre\s+ke\s+liye\b|दूसरे\s+के\s+लिए"
    # "reporting on someone else's behalf": "किसी और की तरफ से बता रहा हूँ" /
    # "kisi aur ki taraf se ..." (with or without the "ki"; फ may carry its
    # nukta: तरफ़).
    r"|किसी\s+और\s+(?:की\s+)?तरफ\u093c?\s+से|\bkisi\s+aur\s+(?:ki\s+)?taraf\s+se\b"
    # Bare "someone else" reply, anchored to the whole reply: "koi aur" /
    # "koi aur ka" / "कोई और" / "किसी और का".
    r"|^\s*(?:koi\s+aur|kisi\s+aur|कोई\s+और|किसी\s+और)"
    r"\s*(?:ka|ki|ke|का|की|के)?\s*[.!?]?\s*$"
    # Someone-else cues for the other supported languages: "on behalf of
    # another person" and possessive + relative forms (checked before
    # _ROLE_SELF, so "no, it happened to my wife" stays someone else).
    # Chillu എൻ്റെ alternate (the same STT spelling variation _ROLE_SELF
    # accepts): a chillu possessive before a relative stays someone_else.
    r"|മറ്റൊരാൾക്ക്\s+വേണ്ടി|(?:എന്റെ|എൻ്റെ)\s+(?:ഭാര്യ|ഭർത്താവ്|മകൻ|മകൾ|അമ്മ|അച്ഛൻ)"
    # Dative-spelling variants: the l-sound before ക്ക് is written with the
    # chillu ൾ (either encoding, U+0D7E / U+0D3D) or as the explicit ള്
    # spelling — the reported phrase is "മറ്റൊരാള്ക്ക് വേണ്ടി റിപ്പോര്ട്ട്
    # ചെയ്യണം" (reporting on behalf of another person).
    r"|മറ്റൊരാ(?:[\u0D7E\u0D3D]|\u0D33\u0D4D)ക്ക്\s+വേണ്ടി"
    # "someone else's?" — the possessive-pronoun follow-up at the identity
    # question, with or without the doubt particle ോ.
    r"|മറ്റാരുടേതോ?"
    # "reporting about/about another person's report" — the observed
    # dative ("മറ്റൊരാ[ള്ക്ക|ൾക്ക]ാണ്") and possessive ("മറ്റാരുടെ")
    # cues before the report noun. The റ-cluster accepts both the ര്
    # spelling and either chillu encoding of ർ (sign U+0D3C / letter
    # U+0D7C), and the final ് may contract away before ആണ്.
    r"|മറ്റൊരാ(?:ൾക്കാണ്|ള്ക്കാണ്|ൾക്ക്\s*ആണ്|ള്ക്ക്\s*ആണ്)"
    r"\s*റിപ്പോ(?:ര്|[\u0D3C\u0D7C])ട്ട്?"
    r"|മറ്റാരുടെ\s*റിപ്പോ(?:ര്|[\u0D3C\u0D7C])ട്ട്?"
    r"|অন্য\s+কারো\s+জন্য|আমার\s+(?:স্ত্রী|স্বামী|ছেলে|মেয়ে|মা|বাবা)"
    r"|બીજા\s+કોઈ\s+માટે|મારી\s+(?:પત્ની|પતિ|દીકરો|દીકરી)"
    r"|ಬೇರೊಬ್ಬರಿಗಾಗಿ|ನನ್ನ\s+(?:ಹೆಂಡತಿ|ಗಂಡ|ಮಗ|ಮಗಳು)"
    r"|दुसऱ्यासाठी|माझ्या\s+(?:पत्नी|पती|मुलाचा|लेकीची)"
    r"|ਕਿਸੇ\s+ਹੋਰ\s+ਲਈ|ਮੇਰੀ\s+(?:ਪਤਨੀ|ਪਤੀ|ਬੇਟਾ|ਬੇਟੀ)"
    r"|வேறு\s+ஒருவருக்காக|என்னுடைய\s+(?:மனைவி|கணவர்|மகன்|மகள்)"
    r"|వేరొకరి\s+కోసం|నా\s+(?:భార్య|భర్త|కొడుకు|కూతురు)",
    re.IGNORECASE,
)
# Symptom still present / unchanged. Maps to ``not_recovered``: a patient
# saying "still ongoing" or "hasn't changed" is reporting no recovery.
_OUTCOME_SAME = re.compile(
    r"ongoing|still\s+(?:have|having|experiencing|waiting|there|happening|going|"
    r"the\s+same|present)|stayed\s+(?:the\s+)?same|staying\s+(?:the\s+)?same|"
    r"about\s+the\s+same|remains?\s+the\s+same|no\s+change|unchanged|"
    r"same\s+as\s+before|persist\w*|"
    r"\b(?:hasn'?t|has\s+not|have\s+not|haven'?t|didn'?t|did\s+not|not)\s+chang\w*"
    # Hindi: "still there" - "abhi tak hai" / "abhi bhi hai".
    r"|abhi\s+tak\s+hai|abhi\s+bhi\s+hai|अभी\s+तक\s+है|अभी\s+भी\s+है",
    re.IGNORECASE,
)
# An explicit "not recovered" claim is preserved as its own value.
_OUTCOME_NOT_RECOVERED = re.compile(
    r"not\s+recovered|not\s+better|not\s+improved|not\s+gone|getting\s+better\s+but\b"
    # Hindi: "not fine (yet)" - "theek nahi".
    r"|theek\s+nahi|thik\s+nahi|ठीक\s+नहीं",
    re.IGNORECASE,
)
# Worsening is preserved as its own current-status value rather than collapsed
# into the generic "not recovered" bucket. Paraphrases such as "getting way too
# severe" or "the pain is much worse now" are recognised too.
_OUTCOME_WORSE = re.compile(
    r"\bworsen\w*|\bgetting\s+worse\b|\bgo(?:ing|t|ne)\s+worse\b|\bworse\b|"
    r"\bdeteriorat\w*|\bmore\s+severe\b|"
    r"\b(?:getting|got|becoming|becomes|going)\s+"
    r"(?:way\s+|much\s+|so\s+|really\s+|far\s+)?(?:too\s+)?(?:more\s+)?"
    r"(?:severe|intense|unbearable|painful|bad)\b"
    # Hindi: "got worse" - "bigad gaya" (with or without the nukta).
    r"|bigad\s+(?:gaya|gayi|raha|rahi)"
    r"|बिगड़?\s+गया|बिगड़?\s+गई|बिगड़?\s+रहा|बिगड़?\s+रही",
    re.IGNORECASE,
)
# A negation immediately preceding the worsening phrase ("not getting worse",
# "no worse", "haven't gotten worse") means the symptoms did not worsen.
# Hindi negation is included for the same pre-posed pattern ("nahi bigad gaya").
_WORSE_NEGATION = re.compile(
    r"\b(?:not|never|no|neither)\b|\b\w+n'?t\b|\bnahi\b|\bnahin\b|नहीं|नही",
    re.IGNORECASE,
)
_OUTCOME_FATAL = re.compile(
    r"\bfatal\b|\bdied\b|\bdeath\b|\bpassed\s+away\b"
    # Hindi: "died" - "mar gaya / mar gayi / mar gaye".
    r"|mar\s+(?:gaya|gayi|gaye)|मर\s+गया|मर\s+गई|मर\s+गए",
    re.IGNORECASE,
)
_OUTCOME_RECOVERED = re.compile(
    r"\bfully\s+recovered\b|\brecovered\b|\bgot\s+better\b|\bresolved\b|\bgone\b|"
    r"\bstopped\b|\bno\s+longer\b|\bcleared\s+up\b|\bfeeling\s+(?:fine|well)\b"
    # Hindi: "became fine / better" - "theek ho gaya", "behtar ho gaya".
    r"|theek\s+ho\s+(?:gaya|gayi)|behtar\s+ho\s+(?:gaya|gayi)"
    r"|ठीक\s+हो\s+गया|ठीक\s+हो\s+गई|बेहतर\s+हो\s+गया|बेहतर\s+हो\s+गई",
    re.IGNORECASE,
)
_OUTCOME_RECOVERING = re.compile(
    r"recovering|improving|getting\s+better|partly\s+better|feeling\s+better|"
    r"\bimproved\b|\bbetter\b"
    # Hindi: "(is) becoming fine/better" - "theek ho raha hai".
    r"|theek\s+ho\s+(?:raha|rahi)\s+hai|behtar\s+ho\s+(?:raha|rahi)\s+hai"
    r"|ठीक\s+हो\s+रहा\s+है|बेहतर\s+हो\s+रहा\s+है",
    re.IGNORECASE,
)
_ONSET = re.compile(r"\b(started|began|begin|onset|since|first noticed)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(
    r"\b(?:demo\s+)?(?:patient|pt|subject)[\s-]*(?:id[\s:.-]*)?\d{2,}\b|\bpt-?\d{2,}\b",
    re.IGNORECASE,
)

# ---- word-number ages ------------------------------------------------------

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_MAX_AGE = 150

# ---- spoken digit phones ----------------------------------------------------

# Single-digit words only (zero..nine), derived from the age word table.
_SPOKEN_DIGIT_WORDS = {word: str(value) for word, value in _ONES.items() if value <= 9}
_MIN_PHONE_DIGITS = 7

# The longest supported local contact number: 10 digits, or 11 when the
# number carries a leading zero (the STD-prefixed form). A longer run is a
# dictation/transcription artifact (a doubled digit, an extra word) and is
# never silently accepted as a contact number.
_MAX_PHONE_DIGITS = 10


def _is_plausible_phone(run: str | None) -> bool:
    """Whether an extracted digit run has a plausible contact-number length.

    The established minimum is ``_MIN_PHONE_DIGITS``; the maximum is
    ``_MAX_PHONE_DIGITS``, relaxed for the attested prefixed forms: an
    11-digit run with a leading zero (the STD-prefixed national form) and
    a 12-digit run with the ``91`` country prefix (the documented
    ``+91 …`` input). Leading zeros/prefixes are preserved; the run is
    never trimmed or padded to force a length."""
    if not isinstance(run, str):
        return False
    digits = re.sub(r"\D", "", run)
    if 7 <= len(digits) <= _MAX_PHONE_DIGITS:
        return True
    if len(digits) == _MAX_PHONE_DIGITS + 1:
        return digits.startswith("0")
    if len(digits) == _MAX_PHONE_DIGITS + 2:
        return digits.startswith("91")
    return False

# Hindi zero words; Hindi one-nine reuse _DEVANAGARI_NUMBERS (single-digit
# entries only, looked up lazily inside the function below).
_HINDI_ZERO_DIGIT_WORDS = {"शून्य": "0", "जीरो": "0", "ज़ीरो": "0"}

# Devanagari spellings of the English digit words, as STT emits them for
# Hinglish speech ("नाइन टू टू फोर फोर वन वन वन"). Only single-digit tokens
# are listed, so multi-digit age words such as "पच्चीस" (= 25) can never
# become phone digits.
_TRANSLITERATED_DIGIT_WORDS = {
    "वन": "1",
    "टू": "2",
    "थ्री": "3",
    "फोर": "4",
    "फ़ोर": "4",
    "फाइव": "5",
    "सिक्स": "6",
    "सेवन": "7",
    "एट": "8",
    "ऐट": "8",
    # The observed ASR spellings of "nine" and "zero" (the ि/ी matra and
    # the झ variant), mirroring the known variants already mapped above.
    "नाइन": "9",
    "नाईन": "9",
    "जीरो": "0",
    "ज़ीरो": "0",
    "झीरो": "0",
}

# Keep Latin letters, ASCII digits, and the letters of every supported Indic
# script (Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada,
# Malayalam — full blocks, mirroring the existing Devanagari handling, which
# keeps each script's own digit glyphs inside tokens); everything else
# (punctuation) separates tokens. ASCII digits are kept so a transcript that
# mixes spoken digit words with literal digits ("nine 2 2 four four …")
# preserves every digit in order.
_SPOKEN_PHONE_TOKENS = re.compile(
    r"[^a-z0-9"
    r"\u0900-\u097F"  # Devanagari (Hindi, Marathi)
    r"\u0980-\u09FF"  # Bengali
    r"\u0A00-\u0A7F"  # Gurmukhi
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0C80-\u0CFF"  # Kannada
    r"\u0D00-\u0D7F"  # Malayalam
    r"]+"
)


def _spoken_digit_lookup(token: str) -> str | None:
    """The single digit a spoken token dictates, or ``None``.

    A token made entirely of digit glyphs (ASCII ``0-9`` or any supported
    script's digits, e.g. Devanagari ``०-९``) contributes its digits in
    order, so a transcript mixing spoken digit words with literal digits
    ("nine 2 2 four four") preserves every digit including repeats and
    leading zeros."""
    if token and token.isdigit():
        try:
            return "".join(str(int(ch)) for ch in token)
        except ValueError:
            return None
    if token in _SPOKEN_DIGIT_WORDS:
        return _SPOKEN_DIGIT_WORDS[token]
    if token in _HINDI_ZERO_DIGIT_WORDS:
        return _HINDI_ZERO_DIGIT_WORDS[token]
    # Hindi number words (एक, दो, ... नौ) from the shared table; only
    # single-digit values qualify, never multi-digit ages.
    value = _DEVANAGARI_NUMBERS.get(token)
    if value is not None and value <= 9:
        return str(value)
    digit = _TRANSLITERATED_DIGIT_WORDS.get(token)
    if digit is not None:
        return digit
    # The other supported locales' digit words (zero..nine, from the
    # reviewed locale tables).
    return _ALL_LOCALE_DIGIT_WORDS.get(token)


def _extract_spoken_phone(text: str) -> str | None:
    """Map a run of spelled-out digits (>= ``_MIN_PHONE_DIGITS``) to digits.

    Handles Prisma-style transcripts such as
    ``"nine eight eight six zero nine four three two one"`` -> ``"9886094321"``,
    Hindi digit words (``"दो दो चार चार एक एक एक"`` -> ``"2244111"``) and the
    Devanagari transliterations STT emits for Hinglish speech
    (``"नाइन टू टू फोर फोर वन वन वन"`` -> ``"92244111"``). Runs shorter than
    the minimum are ignored and no digit is ever invented, so ages and
    unrelated sentences can never become a phone number.
    """
    best: str | None = None
    run: list[str] = []

    def flush() -> None:
        nonlocal best
        if len(run) >= _MIN_PHONE_DIGITS and (best is None or len(run) > len(best)):
            best = "".join(run)
        run.clear()

    for token in _SPOKEN_PHONE_TOKENS.split(
        text.lower().replace("।", " ").replace("॥", " ")
    ):
        digit = _spoken_digit_lookup(token)
        if digit is not None:
            # A digit-word token contributes one digit; a digit-glyph token
            # ("९२२४४…") contributes each of its digits in order, so the
            # minimum-length check below counts actual digits.
            run.extend(digit)
        else:
            flush()
    flush()
    return best


def _spelled_email(text: str) -> str | None:
    """A spelled/dictated email address, or ``None`` when clearly not one.

    Handles individually spoken letters and digits separated by transcript
    punctuation ("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com" ->
    ``sreehari@gmail.com``), whole-word local parts ("srihari at gmail dot
    com"), spoken ``underscore``/``hyphen`` and the ``at the rate`` / ``at``
    / ``dot`` cues. Character order is preserved exactly; nothing is guessed.
    Both an ``@`` cue and a ``dot`` cue must be present, and the result must
    be a single valid address ending in a recognised domain label, so
    ordinary speech never becomes an address.
    """
    work = (text or "").strip().lower()
    if not work:
        return None
    # A leading Malayalam negation before a dictated address ("ഇല്ല ആർ …
    # ഡോട്ട് കോം") is a correction lead, never a decline: strip it the same
    # way the Latin "no, …" lead below is removed.
    work = _MALAYALAM_EMAIL_NEGATION.sub("", work)
    # Correction leads ("no, the email should be ...", "it should be ...").
    work = re.sub(r"^\s*(?:no|nope)[,\s!]+", "", work)
    work = re.sub(
        r"^\s*(?:(?:my|the)?\s*(?:e-?mail(?:\s+address)?|mail\s+address|address)"
        r"|it'?s|that'?s|it|that)?\s*(?:should\s+be|is|:)?\s*",
        "",
        work,
    )
    # The phrase cue becomes the "@" token before tokenising.
    work = re.sub(r"\bat\s+the\s+rate\b", "@", work)
    work = _MALAYALAM_AT_RATE.sub("@", work)
    has_at = bool(re.search(r"\bat\b|@", work))
    has_dot = bool(re.search(r"\bdot\b|\.", work) or "ഡോട്ട്" in work or "डॉट" in work)
    if not (has_at and has_dot):
        return None

    out: list[str] = []
    seen_at = False
    local_letters = False
    local_words = False
    for raw in re.split(r"[\s,;]+", work):
        token = raw.strip(" .!?;:")
        if not token:
            continue
        if token == "@" or token == "at":
            if seen_at or not out:
                return None  # ambiguous or leading "@"
            out.append("@")
            seen_at = True
            continue
        if token in ("dot", "ഡോട്ട്", "डॉट"):
            out.append(".")
            continue
        if token == "underscore":
            out.append("_")
            continue
        if token in ("hyphen", "dash"):
            out.append("-")
            continue
        # Malayalam dictation tokens translate to their Latin spelling before
        # the letter/word logic: letter names become single letters ("പി" ->
        # "p"), provider/domain words become words ("ജിമെയിൽ" -> "gmail").
        token = _MALAYALAM_EMAIL_TOKENS.get(token, token)
        digit = _spoken_digit_lookup(token)
        if digit is not None:
            out.append(digit)
            continue
        if seen_at and "." in token and re.fullmatch(
            r"[a-z0-9](?:[a-z0-9.+-]*[a-z0-9])?", token
        ):
            # A compact domain token directly after the "@" cue
            # ("@ GMAIL.COM", "at gmail.com"): appended verbatim. The full
            # address is still validated below (single "@", _EMAIL, and a
            # recognised final label), so nothing is ever guessed.
            out.append(token)
            continue
        if not seen_at:
            # The local part must be either fully spelled ("s, r, e, e") or
            # whole words ("sreehari", "john.doe_work"): mixing the two is
            # ambiguous speech and is never guessed. Symbol cues (dot,
            # underscore, hyphen) and spoken digits count as neither.
            if re.fullmatch(r"[a-z]", token):
                local_letters = True
            elif re.fullmatch(r"[a-z]{2,}", token):
                local_words = True
        if re.fullmatch(r"[a-z]", token) or token.isdigit():
            out.append(token)
            continue
        if token in _EMAIL_DICTATION_WORDS or re.fullmatch(r"[a-z]{2,}", token):
            out.append(token)
            continue
        # Not part of an address: stop rather than guess.
        return None
    if local_letters and local_words:
        return None

    candidate = "".join(out)
    if candidate.count("@") != 1:
        return None
    if not _EMAIL.fullmatch(candidate):
        return None
    # The final label must be a recognised domain/TLD, which is the strongest
    # guard against ordinary speech ever becoming an address.
    if candidate.rsplit(".", 1)[-1] not in _EMAIL_TLDS:
        return None
    return candidate


def _normalize_spoken_email(text: str) -> str | None:
    """Normalize a dictated email address, or ``None`` when there is none.

    A literal address already present in the text is returned exactly as
    matched (existing behaviour). Spelled-out addresses ("S, R, E, E, H, A,
    R, I, at the rate, Gmail, dot, com") are folded in deterministically, and
    otherwise the existing spoken forms are applied: ``"at the rate"`` / a
    standalone ``"at"`` become ``"@"``, a standalone ``"dot"`` becomes
    ``"."``, one known lead-in ("my email is", "it's", ...) is dropped, and
    spacing directly around those components is removed.

    The result is returned only when it fully validates against ``_EMAIL``,
    so nothing is stored otherwise: missing characters are never invented and
    partial or ambiguous input (no ``"@"``, several ``"@"`` cues, spaces or
    extra words inside a component) is rejected.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None

    # 1. Literal address, exactly as written.
    literal = _EMAIL.search(stripped)
    if literal:
        return literal.group(0)

    # 2. Spelled-out letters/digits ("S, R, E, E, H, A, R, I, at the rate, ...").
    spelled = _spelled_email(stripped)
    if spelled is not None:
        return spelled

    # 3. Spoken form: one known lead-in, then trailing punctuation.
    work = _EMAIL_LEAD.sub("", stripped, count=1).rstrip(" \t.!?,;")
    work = _SPOKEN_AT_RATE.sub("@", work)
    work = _SPOKEN_AT.sub("@", work)
    work = _SPOKEN_DOT.sub(".", work)
    work = _EMAIL_COMPONENT_SPACES.sub(r"\1", work)

    # Exactly one "@": none means no address, several are ambiguous.
    if work.count("@") != 1:
        return None

    # The whole utterance must be one complete address.
    if not _EMAIL.fullmatch(work):
        return None
    return work


def parse_word_number(phrase: str) -> int | None:
    """Convert a small English number phrase (e.g. ``"thirty four"``) to an int.

    Supports whole numbers up to ``<digits> thousand`` and compound forms
    (``"two thousand five"`` -> 2005, ``"one hundred and five"`` -> 105).
    Digits must appear in descending magnitude: decade-style phrases such as
    ``"nineteen ninety eight"`` (19 then 90) are rejected instead of being
    summed into nonsense (117); they are handled by the spoken-year parser.
    Returns ``None`` when the phrase is not a recognised number.
    """
    tokens = [
        t
        for t in re.sub(r"\band\b", " ", phrase.lower()).replace("-", " ").split()
        if t
    ]
    if not tokens:
        return None
    total = 0
    for token in tokens:
        if token in _ONES:
            value = _ONES[token]
            if total > 0 and value >= total:
                return None
            total += value
        elif token in _TENS:
            value = _TENS[token]
            if total > 0 and value >= total:
                return None
            total += value
        elif token == "hundred":
            if total == 0:
                return None
            total *= 100
        elif token == "thousand":
            if total == 0:
                return None
            total *= 1000
        else:
            return None
    return total if total >= 1 else None


_AGE_SUFFIX = re.compile(
    r"\byears?\s*old\b|\byears\s+of\s+age\b|\byrs?\s*old\b|\by/o\b", re.IGNORECASE
)
_AGE_PREFIX = re.compile(
    r"\b(?:my\s+)?age\s+(?:is\s+)?(\d{1,3}|[a-z]+(?:\s+[a-z]+){0,2})\b"
    r"|\baged\s+(\d{1,3}|[a-z]+(?:\s+[a-z]+){0,2})\b",
    re.IGNORECASE,
)


def _normalise_age(value: str) -> str | None:
    """Validate a parsed age candidate and return it as a digit string."""
    if re.fullmatch(r"\d+", value):
        number = int(value)
    else:
        number = parse_word_number(value)
    if number is None or not 1 <= number <= _MAX_AGE:
        return None
    return str(number)


def _age_rejected_for_dob_question(text: str) -> bool:
    """A bare 3+ digit answer while a date of birth is pending is not an age.

    Mis-heard date fragments (e.g. ``117``) must never be stored as an age
    and read back as part of the patient's details; the date-of-birth
    question is simply asked again.
    """
    return bool(re.fullmatch(r"\d{3,}", (text or "").strip()))


def _age_from_suffix_clause(clause: str) -> str | None:
    suffix = _AGE_SUFFIX.search(clause)
    if suffix is None:
        return None
    before = clause[: suffix.start()].rstrip()
    digit = re.search(r"(\d{1,4})\s*$", before)
    if digit is not None:
        candidate = _normalise_age(digit.group(1))
        if candidate is not None:
            return candidate
    words = before.split()
    for size in range(min(4, len(words)), 0, -1):
        phrase = " ".join(words[-size:])
        candidate = _normalise_age(phrase)
        if candidate is not None:
            return candidate
    return None


def _extract_age(text: str) -> str | None:
    """Find a plausible age in free text, preferring '<n> years old' forms."""
    for clause in re.split(r"[.;!?\n]", text):
        candidate = _age_from_suffix_clause(clause)
        if candidate is not None:
            return candidate
        prefix = _AGE_PREFIX.search(clause)
        if prefix is not None:
            candidate = _normalise_age(prefix.group(1) or prefix.group(2))
            if candidate is not None:
                return candidate
    return None


# ---- Devanagari number-word ages ---------------------------------------------

# Common Hindi number words for ages, with the spelling variants STT commonly
# emits. Unknown words are simply not recognised; nothing is inferred.
_DEVANAGARI_NUMBERS: Dict[str, int] = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5,
    "छह": 6, "छे": 6,
    "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18,
    "उन्नीस": 19, "उनीस": 19,
    "बीस": 20,
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "पचीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "पैंतीस": 35,
    "चालीस": 40, "पैंतालीस": 45,
    "पचास": 50, "पचपन": 55,
    "साठ": 60, "पैंसठ": 65,
    "सत्तर": 70, "पचहत्तर": 75,
    "अस्सी": 80, "पचासी": 85,
    "नब्बे": 90, "पंचानवे": 95,
}

# Age cues. "उम्र" (age) is strong enough for opportunistic capture; "साल"
# (years) is only trusted while the age question is pending, because "दो साल
# पहले" ("two years ago") is an onset statement, not an age.
_DEVANAGARI_AGE_CUE = re.compile(r"उम्र")
_DEVANAGARI_AGE_CUE_PENDING = re.compile(r"उम्र|साल")
# Unit/filler words dropped before the bare pending-path lookup.
_DEVANAGARI_AGE_FILLER = frozenset(
    {"साल", "है", "हैं", "का", "की", "के", "मेरी", "मेरे", "उम्र", "लगभग", "हूँ", "हूं"}
)


def _extract_devanagari_age(text: str, *, pending: bool = False) -> str | None:
    """A plausible age from Hindi number words, or ``None``.

    Opportunistic capture requires the strong "उम्र" cue. The pending
    (age-question) path also trusts "साल" and bare number words such as
    "पच्चीस", mirroring how bare digits and English number words are already
    accepted while that question is pending. Values are re-checked against
    the existing 1.._MAX_AGE range.
    """
    if pending:
        stripped = text.strip().replace("।", " ").rstrip(" .,!?")
        cleaned = [
            token for token in stripped.split() if token not in _DEVANAGARI_AGE_FILLER
        ]
        if len(cleaned) == 1:
            value = _DEVANAGARI_NUMBERS.get(cleaned[0])
            if value is not None and 1 <= value <= _MAX_AGE:
                return str(value)
        if _DEVANAGARI_AGE_CUE_PENDING.search(text):
            for token in stripped.split():
                value = _DEVANAGARI_NUMBERS.get(token.strip(".,!?()\"'"))
                if value is not None and 1 <= value <= _MAX_AGE:
                    return str(value)
        return None
    if not _DEVANAGARI_AGE_CUE.search(text):
        return None
    for token in text.replace("।", " ").split():
        value = _DEVANAGARI_NUMBERS.get(token.strip(".,!?()\"'"))
        if value is not None and 1 <= value <= _MAX_AGE:
            return str(value)
    return None


# ---- date of birth -----------------------------------------------------------

# Explicit DOB wording: such a date is always a DOB answer, whatever question
# is pending. A bare complete date counts as a DOB only while the age/DOB
# question is pending (``allow_bare_dob``), so onset and treatment dates such
# as "12/03/2026" are never misread as a date of birth.
_DOB_CUE = re.compile(r"\b(?:born|birth|dob|date\s+of\s+birth)\b", re.IGNORECASE)

# A trailing date-of-birth lead-in ("and my date of birth is", "dob:") that
# must never be stored as part of a name.
_DOB_NAME_LEAD = re.compile(
    r"(?i)(?:^|[\s,;&]+)(?:and\s+)?(?:my\s+)?(?:date\s+of\s+birth|dob)\s*(?:is|:)?\s*$"
)

# The Hindi equivalent for combined answers: "अल्फा बीटा और जन्मदिन पच्चीस
# में दो हजार पाँच" — everything from the DOB lead to the end (the lead,
# an optional possessive such as "मेरा", and the date attempt, parseable or
# not) is never part of the name. The date itself is handled by the DOB
# parsers when it is well formed.
_HINDI_DOB_NAME_TAIL = re.compile(
    r"\s+(?:और\s+)?(?:मेरा\s+|मेरी\s+)?(?:जन्मदिन|जन्म\s+तिथि)(?:\s+.*)?$"
    # Malayalam combined answers ("… ജന്മദിനം ഇരുപത് മെയ് …", with the
    # connector "പിന്നെ", the ജന്മതിഥി label spelling and the observed ASR
    # variant ജന്മതിയി of it): everything from the DOB lead to the end is
    # never part of the name.
    r"|\s+ജന്മദിനം(?:\s+.*)?$"
    r"|\s+(?:പിന്നെ\s+)?(?:ജന്മതിഥി|ജന്മതിയി)(?:\s+.*)?$"
    # The remaining supported locales' DOB labels (established project
    # phrases, mirrored from the workflow's _DOB_CHOICE): everything from
    # the label to the end is never part of the name.
    r"|\s+(?:জন্ম\s+তারিখ|জন্মদিন)(?:\s+.*)?$"
    r"|\s+(?:જન્મ\s+તારીખ|જન્મદિન)(?:\s+.*)?$"
    r"|\s+(?:ಜನ್ಮ\s+ದಿನಾಂಕ|ಹುಟ್ಟಿದ\s+ದಿನ)(?:\s+.*)?$"
    # Marathi combined answers ("… राजू के आणि माझा जन्मतारीख वीस जुलै
    # …"): the conjunction "आणि" and the possessive leads "माझा/माझं/माझी"
    # before the DOB label are never part of the name (mirroring the Hindi
    # branch's "और मेरा" prefix above).
    r"|\s+(?:आणि\s+)?(?:माझा\s+|माझं\s+|माझी\s+)?(?:जन्मतारीख|जन्मदिन)(?:\s+.*)?$"
    r"|\s+(?:ਜਨਮ\s+ਤਾਰੀਖ|ਜਨਮਦਿਨ)(?:\s+.*)?$"
    r"|\s+(?:பிறந்த\s+தேதி|பிறந்தநாள்)(?:\s+.*)?$"
    r"|\s+(?:పుట్టిన\s+తేదీ|జన్మ\s+దినం)(?:\s+.*)?$"
)

_DOB_MONTH_DATE = re.compile(
    # "12 March 1980", "21-Mar-2008" (day and month name may be joined by a
    # space or a hyphen, the form a typed DD-MMM-YYYY answer produces).
    r"\b(\d{1,2})(?:st|nd|rd|th)?[\s-]([A-Za-z]{3,9})\.?[\s-]+(\d{4})\b",
    re.IGNORECASE,
)
_DOB_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
# DD-MM-YYYY, the format the flow asks the person to speak/type. Disjoint from
# _DOB_DASH (YYYY-MM-DD): the four-digit year sits at the end here.
_DOB_DMY_DASH = re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b")
_DOB_DASH = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOB_BORN = re.compile(r"\bborn(?:\s+in)?\s+(\d{4})\b", re.IGNORECASE)

_MONTH_NAMES: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Spoken-English date of birth from STT transcripts: "<day> <month name>
# <year>" with word or digit components, e.g. "twenty seven may two thousand"
# or "27 may two thousand". The month alternation is built from _MONTH_NAMES
# (longest first) so only real month names can anchor a match; every
# component is validated in code before anything is stored.
_SPOKEN_DOB = re.compile(
    r"\b(\d{1,2}(?:st|nd|rd|th)?|[a-z]+(?:\s+[a-z]+)?)\s+("
    + "|".join(
        re.escape(name) for name in sorted(_MONTH_NAMES, key=len, reverse=True)
    )
    + r")\.?\s+(\d{4}|[a-z]+(?:\s+[a-z]+){0,3})\b",
    re.IGNORECASE,
)

# Ordinal day forms spoken by users ("twenty seventh", "27th") are stripped
# to their cardinal value before parsing; nothing is guessed.
_ORDINAL_SUFFIX = re.compile(r"(?i)(?:st|nd|rd|th)$")

# Irregular spoken ordinal days ("eighth", "twelfth", "twentieth", "first"):
# the suffix-strip trick cannot recover these, so they are mapped directly.
_DAY_ORDINALS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}


def _spoken_day(value: str) -> Optional[int]:
    """Parse a spoken or numeric day, tolerating ordinal forms."""
    text = (value or "").strip().lower()
    direct = _DAY_ORDINALS.get(text)
    if direct is not None:
        return direct
    parts = text.split()
    if (
        len(parts) == 2
        and parts[0] in ("twenty", "thirty")
        and parts[1] in _DAY_ORDINALS
    ):
        # "twenty eighth" -> 28, "thirty first" -> 31.
        return (20 if parts[0] == "twenty" else 30) + _DAY_ORDINALS[parts[1]]
    text = _ORDINAL_SUFFIX.sub("", text).strip()
    if text.isdigit():
        return int(text)
    return parse_word_number(text)


# Decade-style spoken years ("nineteen ninety eight" -> 1998, "twenty twenty
# five" -> 2025). Only these exact shapes are accepted; a partial year such
# as "nineteen ninety" stays incomplete at the caller and is never guessed.
_DECADE_YEAR = re.compile(
    r"^(nineteen|twenty)\s+(eighty|seventy|sixty|fifty|forty|thirty|twenty|ninety)"
    r"(?:\s+([a-z]+|\d{1,2}))?$",
    re.IGNORECASE,
)


def _spoken_year(text: str) -> tuple[Optional[int], str]:
    """Parse a spoken/digit year, trusting only the longest parsed prefix.

    "two thousand" -> (2000, "two thousand"), "two thousand and five" ->
    (2005, "two thousand and five"), "2000" -> (2000, "2000"), "nineteen
    ninety eight" -> (1998, "nineteen ninety eight"); trailing sentence words
    ("... and I have a headache") are ignored and the returned text is only
    the parsed prefix so nothing extra is ever stored.
    """
    # Decade form first ("nineteen ninety eight"): the generic prefix loop
    # would otherwise accept the leading "nineteen" on its own. A trailing
    # conjunction ("... and I have a headache") is dropped first.
    decade_text = re.sub(r"\s+and\s*$", "", text.strip())
    decade = _DECADE_YEAR.match(decade_text)
    if decade is not None:
        prefix = 1900 if decade.group(1).lower() == "nineteen" else 2000
        tens = _TENS.get(decade.group(2).lower())
        if tens is not None:
            year = prefix + tens
            clean = f"{decade.group(1)} {decade.group(2)}"
            tail = (decade.group(3) or "").strip()
            if tail:
                units = parse_word_number(tail)
                if units is None:
                    if tail.isdigit():
                        units = int(tail)
                    else:
                        return None, ""
                if not 0 <= units <= 9:
                    return None, ""
                year += units
                clean = f"{clean} {tail}"
            if _dob_year_ok(year):
                return year, clean
    parts = text.split()
    for size in range(len(parts), 0, -1):
        candidate = " ".join(parts[:size])
        if candidate.isdigit():
            return int(candidate), candidate
        parsed = parse_word_number(candidate)
        if parsed is not None:
            # Drop only a trailing conjunction from the junk case ("two
            # thousand and" before a sentence); keep year variants verbatim
            # ("two thousand and five").
            clean = candidate
            if clean.lower().endswith(" and"):
                clean = clean[:-4].rstrip()
            return parsed, clean
    return None, ""


def _dob_year_ok(year: int) -> bool:
    return 1900 <= year <= date.today().year


# Hindi month names (Devanagari) with the common STT spellings, for spoken
# dates of birth like "पच्चीस मई दो हजार चार" (25-May-2004). "मे" and "में"
# are frequent speech-recognition variants of मई; they are accepted only in
# the complete date shape (see _extract_hindi_dob) because both double as
# everyday words/particles.
_HINDI_MONTHS: Dict[str, int] = {
    "जनवरी": 1,
    "फ़रवरी": 2,
    "फरवरी": 2,
    "मार्च": 3,
    "अप्रैल": 4,
    "मई": 5,
    "मे": 5,
    "में": 5,
    "जून": 6,
    "जुलाई": 7,
    "अगस्त": 8,
    "सितम्बर": 9,
    "सितंबर": 9,
    "अक्टूबर": 10,
    "नवम्बर": 11,
    "नवंबर": 11,
    "दिसम्बर": 12,
    "दिसंबर": 12,
}

# The date-marker-prone variants: accepted only when every component of the
# date is present, so an incomplete/ambiguous "… में दो हजार" is never read
# as a month.
_HINDI_MONTH_VARIANTS = frozenset({"मे", "में"})


def _extract_hindi_dob(text: str) -> tuple[Optional[date], Optional[str]]:
    """A Devanagari spoken date of birth and its verbatim form, or none.

    Matches "<day word> <Hindi month> दो हजार [<year unit>]", e.g.
    "पच्चीस मई दो हजार चार" -> 25-May-2004 (also with the STT variants
    "मे"/"में" for मई, and with nukta spellings such as "हज़ार"). The day
    and the year may each be a plain 4-digit-or-fewer digit token, because
    STT commonly transcribes spoken numbers as digits ("पच्चीस मई 2004").
    Day, month and the year (the "दो हजार" lead or a digit year) must all be
    present; the variant months additionally require the unit word so an
    ambiguous date marker is never guessed as a month. Nothing else is
    inferred, so a month-less "पच्चीस दो हजार चार" yields no date.
    """
    raw_tokens = [
        token.strip(".,!?()\"'\u0964")
        for token in (text or "").replace("\u0964", " ").split()
    ]
    # Lookup tokens ignore the nukta (हज़ार/फ़रवरी); the raw tokens are kept
    # for the verbatim form so later name-splitting replaces the original text.
    tokens = [token.replace("\u093c", "") for token in raw_tokens]
    for index, token in enumerate(tokens):
        month = _HINDI_MONTHS.get(token)
        if month is None or index == 0 or index + 1 >= len(tokens):
            continue
        day = _DEVANAGARI_NUMBERS.get(tokens[index - 1])
        if day is None and tokens[index - 1].isdigit():
            day = int(tokens[index - 1])
        if day is None or not 1 <= day <= 31:
            continue
        # A digit year ("2004") is an exact spoken year; the "दो हजार" word
        # lead keeps the existing unit/filler rules. A variant month (मे/में)
        # stays conservative: a bare "दो हजार"/digit year after it is never
        # enough to read the marker as the month.
        year_token = tokens[index + 1]
        digit_year = year_token.isdigit() and len(year_token) == 4
        if digit_year and token not in _HINDI_MONTH_VARIANTS:
            if not _dob_year_ok(int(year_token)):
                continue
            try:
                parsed = date(int(year_token), month, day)
            except ValueError:
                continue
            return parsed, " ".join(raw_tokens[index - 1 : index + 2])
        if index + 2 >= len(tokens):
            continue
        if tokens[index + 1] != "\u0926\u094b" or tokens[index + 2] != "\u0939\u091c\u093e\u0930":
            continue
        if index + 3 < len(tokens):
            unit = _DEVANAGARI_NUMBERS.get(tokens[index + 3])
            unit_end = index + 3
            if unit is None and tokens[index + 3] in _HINDI_DATE_FILLERS:
                # "… दो हजार है": the particle is speech filler, never part
                # of the date; an optional year unit may follow it
                # ("… दो हजार चार है"). A variant month (मे/में) still
                # requires that unit word — filler alone never disambiguates.
                if index + 4 < len(tokens):
                    unit = _DEVANAGARI_NUMBERS.get(tokens[index + 4])
                    unit_end = index + 4
                elif token not in _HINDI_MONTH_VARIANTS:
                    # "दो हजार" alone: the year is exactly 2000.
                    unit = 0
                    unit_end = index + 2
            if unit is None or unit > 99:
                # Unrecognisable text follows the year lead: ambiguous, so
                # the date is never guessed at.
                continue
            year = 2000 + unit
            end = unit_end
        elif token in _HINDI_MONTH_VARIANTS:
            # "… में दो हजार" without the unit word: the marker may not be a
            # month at all, so an ambiguous date is never confirmed.
            continue
        else:
            year = 2000
            end = index + 2
        if not _dob_year_ok(year):
            continue
        try:
            parsed = date(year, month, day)
        except ValueError:
            continue
        return parsed, " ".join(raw_tokens[index - 1 : end + 1])
    return None, None


# ---- spoken DOB tables for the other supported languages ---------------------
# Conservative, data-only tables mirroring the Hindi parser's shape: a
# complete "<day> <month> <year>" phrase is required, nothing is inferred,
# and only unambiguous year-unit forms (2001-2009) are supported. Tables
# should be native-speaker-reviewed before production trust.

_SPOKEN_DATE_TABLES: Dict[str, Dict[str, Any]] = {
    "ml-IN": {
        "days": {
            "ഒന്ന്": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5,
            "ആറ്": 6, "ഏഴ്": 7, "എട്ട്": 8, "ഒമ്പത്": 9,
            # The chillu spelling of nine ("ഒൻപത്") that the browser STT
            # emitted for phone digits; "ഒമ്പത്" and "ഒൻപത്" are both
            # standard Malayalam spellings.
            "ഒൻപത്": 9, "പത്ത്": 10,
            "പതിനൊന്ന്": 11, "പന്ത്രണ്ട്": 12, "പതിമൂന്ന്": 13,
            "പതിനാല്": 14, "പതിനഞ്ച്": 15, "പതിനാറ്": 16,
            "പതിനേഴ്": 17, "പതിനെട്ട്": 18, "പത്തൊൻപത്": 19,
            "പത്തൊമ്പത്": 19, "ഇരുപത്": 20, "ഇരുപത്തൊന്ന്": 21,
            "ഇരുപത്തിയൊന്ന്": 21, "ഇരുപത്തിരണ്ട്": 22,
            "ഇരുപത്തിമൂന്ന്": 23, "ഇരുപത്തിനാല്": 24,
            "ഇരുപത്തിയഞ്ച്": 25, "ഇരുപത്തിയാറ്": 26,
            "ഇരുപത്തിയേഴ്": 27, "ഇരുപത്തിയെട്ട്": 28,
            "ഇരുപത്തൊൻപത്": 29, "ഇരുപത്തൊമ്പത്": 29, "മുപ്പത്": 30,
            "മുപ്പത്തൊന്ന്": 31, "മുപ്പത്തിയൊന്ന്": 31,
            # Spaced compound day forms (ASR often emits the compound with a
            # space, mirroring the Telugu table's "ఇరవై ఒకటി" pattern);
            # without these, "ഇരുപത്തി രണ്ട്" silently matched the unit word
            # alone as day 2.
            "ഇരുപത്തി ഒന്ന്": 21, "ഇരുപത്തി രണ്ട്": 22,
            "ഇരുപത്തി മൂന്ന്": 23, "ഇരുപത്തി നാല്": 24,
            "ഇരുപത്തി അഞ്ച്": 25, "ഇരുപത്തി ആറ്": 26,
            "ഇരുപത്തി ഏഴ്": 27, "ഇരുപത്തി എട്ട്": 28,
            "ഇരുപത്തി ഒമ്പത്": 29, "മുപ്പത്തി ഒന്ന്": 31,
        },
        "months": {
            "ജനുവരി": 1, "ഫെബ്രുവരി": 2, "മാർച്ച്": 3, "മാർച്ച": 3,
            "ഏപ്രിൽ": 4, "മെയ്": 5, "മേയ്": 5, "മേ": 5, "ജൂൺ": 6,
            "ജൂലൈ": 7, "ഓഗസ്റ്റ്": 8, "ഓഗസ്റ്റ": 8, "സെപ്റ്റംബർ": 9,
            "ഒക്ടോബർ": 10, "നവംബർ": 11, "ഡിസംബർ": 12,
        },
        "year_leads": {"രണ്ടായിരം": 2000, "രണ്ടായിരത്തി": 2000},
        "year_units": {
            "ഒന്ന്": 1, "രണ്ട്": 2, "മൂന്ന്": 3, "നാല്": 4, "അഞ്ച്": 5,
            "ആറ്": 6, "ഏഴ്": 7, "എട്ട്": 8, "ഒമ്പത്": 9,
        },
    },
    "bn-IN": {
        "days": {
            "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6,
            "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11, "বারো": 12,
            "তেরো": 13, "চৌদ্দ": 14, "পনেরো": 15, "ষোলো": 16,
            "সতেরো": 17, "আঠারো": 18, "উনিশ": 19, "বিশ": 20, "একুশ": 21,
            "বাইশ": 22, "তেইশ": 23, "চব্বিশ": 24, "পঁচিশ": 25,
            "ছাব্বিশ": 26, "সাতাশ": 27, "আটাশ": 28, "উনত্রিশ": 29,
            "ত্রিশ": 30, "একত্রিশ": 31,
        },
        "months": {
            "জানুয়ারি": 1, "জানুয়ারী": 1, "জানুযারি": 1, "জানুযারী": 1,
            "ফেব্রুয়ারি": 2, "ফেব্রুয়ারী": 2, "ফেব্রুযারি": 2, "ফেব্রুযারী": 2,
            "মার্চ": 3, "এপ্রিল": 4, "মে": 5, "জুন": 6, "জুলাই": 7, "আগস্ট": 8,
            "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12,
        },
        "year_leads": {"দুই হাজার": 2000},
        "year_units": {
            "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6,
            "সাত": 7, "আট": 8, "নয়": 9,
        },
    },
    "gu-IN": {
        "days": {
            "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6,
            "સાત": 7, "આઠ": 8, "નવ": 9, "દસ": 10, "અગિયાર": 11,
            "બાર": 12, "તેર": 13, "ચૌદ": 14, "પંદર": 15, "સોળ": 16,
            "સત્તાર": 17, "અઢાર": 18, "ઓગણીસ": 19, "વીસ": 20,
            "એકવીસ": 21, "બાવીસ": 22, "તેવીસ": 23, "ચોવીસ": 24,
            "પચીસ": 25, "છવીસ": 26, "સત્તાવીસ": 27, "અઠ્ઠાવીસ": 28,
            "ઓગણત્રીસ": 29, "ત્રીસ": 30, "એકત્રીસ": 31,
        },
        "months": {
            "જાન્યુઆરી": 1, "ફેબ્રુઆરી": 2, "માર્ચ": 3, "એપ્રિલ": 4,
            "મે": 5, "જૂન": 6, "જુલાઈ": 7, "ઓગસ્ટ": 8,
            "સપ્ટેમ્બર": 9, "ઓક્ટોબર": 10, "નવેમ્બર": 11, "ડિસેમ્બર": 12,
        },
        "year_leads": {"બે હજાર": 2000},
        "year_units": {
            "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5, "છ": 6,
            "સાત": 7, "આઠ": 8, "નવ": 9,
        },
    },
    "kn-IN": {
        "days": {
            "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲು": 4, "ಐದು": 5,
            "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8, "ಒಂಬತ್ತು": 9, "ಹತ್ತು": 10,
            "ಹನ್ನೊಂದು": 11, "ಹನ್ನೆರಡು": 12, "ಹದಿಮೂರು": 13,
            "ಹದಿನಾಲ್ಕು": 14, "ಹದಿನೈದು": 15, "ಹದಿನಾರು": 16,
            "ಹದಿನೇಳು": 17, "ಹದಿನೆಂಟು": 18, "ಹತ್ತೊಂಬತ್ತು": 19,
            "ಇಪ್ಪತ್ತು": 20, "ಇಪ್ಪತ್ತೊಂದು": 21, "ಇಪ್ಪತ್ತೆರಡು": 22,
            "ಇಪ್ಪತ್ತಮೂರು": 23, "ಇಪ್ಪತ್ತನಾಲ್ಕು": 24, "ಇಪ್ಪತ್ತೈದು": 25,
            "ಇಪ್ಪತ್ತಾರು": 26, "ಇಪ್ಪತ್ತೇಳು": 27, "ಇಪ್ಪತ್ತೆಂಟು": 28,
            "ಇಪ್ಪತ್ತೊಂಬತ್ತು": 29, "ಮೂವತ್ತು": 30, "ಮೂವತ್ತೊಂದು": 31,
        },
        "months": {
            "ಜನವರಿ": 1, "ಫೆಬ್ರವರಿ": 2, "ಫೆಬ್ರುವರಿ": 2, "ಮಾರ್ಚ್": 3,
            "ಮಾರ್ಚಿ": 3, "ಏಪ್ರಿಲ್": 4, "ಮೇ": 5, "ಜೂನ್": 6, "ಜುಲೈ": 7,
            "ಆಗಸ್ಟ್": 8, "ಸೆಪ್ಟೆಂಬರ್": 9, "ಅಕ್ಟೋಬರ್": 10, "ನವೆಂಬರ್": 11,
            "ಡಿಸೆಂಬರ್": 12,
        },
        "year_leads": {"ಎರಡು ಸಾವಿರ": 2000, "ಎರಡು ಸಾವಿರದ": 2000},
        "year_units": {
            "ಒಂದು": 1, "ಎರಡು": 2, "ಮೂರು": 3, "ನಾಲು": 4, "ಐದು": 5,
            "ಆರು": 6, "ಏಳು": 7, "ಎಂಟು": 8, "ಒಂಬತ್ತು": 9,
        },
    },
    "mr-IN": {
        "days": {
            "एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6,
            "सात": 7, "आठ": 8, "नऊ": 9, "दहा": 10, "अकरा": 11, "बारा": 12,
            "तेरा": 13, "चौदा": 14, "पंधरा": 15, "सोळा": 16, "सतरा": 17,
            "अठरा": 18, "एकोणीस": 19, "वीस": 20, "एकवीस": 21, "बावीस": 22,
            "तेवीस": 23, "चोवीस": 24, "पंचवीस": 25, "सव्वीस": 26,
            "सत्तावीस": 27, "अठ्ठावीस": 28, "एकोणतीस": 29, "तीस": 30,
            "एकत्तीस": 31,
        },
        "months": {
            "जानेवारी": 1, "फेब्रुवारी": 2, "मार्च": 3, "एप्रिल": 4,
            "मे": 5, "जून": 6, "जुलै": 7, "ऑगस्ट": 8, "सप्टेंबर": 9,
            "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12,
        },
        "year_leads": {"दोन हजार": 2000},
        "year_units": {
            "एक": 1, "दोन": 2, "तीन": 3, "चार": 4, "पाच": 5, "सहा": 6,
            "सात": 7, "आठ": 8, "नऊ": 9,
        },
    },
    "pa-IN": {
        "days": {
            "ਇੱਕ": 1, "ਦੋ": 2, "ਤਿੰਨ": 3, "ਚਾਰ": 4, "ਪੰਜ": 5, "ਛੇ": 6,
            "ਸੱਤ": 7, "ਅੱਠ": 8, "ਨੌਂ": 9, "ਦਸ": 10, "ਗਿਆਰਾਂ": 11,
            "ਬਾਰਾਂ": 12, "ਤੇਰਾਂ": 13, "ਚੌਦਾਂ": 14, "ਪੰਦਰਾਂ": 15,
            "ਸੋਲਾਂ": 16, "ਸਤਾਰਾਂ": 17, "ਅਠਾਰਾਂ": 18, "ਉੱਨੀ": 19,
            "ਵੀਹ": 20, "ਇੱਕੀ": 21, "ਬਾਈ": 22, "ਤੇਈ": 23, "ਚੌਵੀ": 24,
            "ਪੱਚੀ": 25, "ਛੱਬੀ": 26, "ਸਤਾਈ": 27, "ਅਠਾਈ": 28,
            "ਉਨੱਤੀ": 29, "ਤੀਹ": 30, "ਇੱਕਤੀ": 31,
        },
        "months": {
            "ਜਨਵਰੀ": 1, "ਫ਼ਰਵਰੀ": 2, "ਫਰਵਰੀ": 2, "ਮਾਰਚ": 3,
            "ਅਪ੍ਰੈਲ": 4, "ਮਈ": 5, "ਜੂਨ": 6, "ਜੁਲਾਈ": 7, "ਅਗਸਤ": 8,
            "ਅਗਸਟ": 8, "ਸਤੰਬਰ": 9, "ਸਿਤੰਬਰ": 9, "ਅਕਤੂਬਰ": 10,
            "ਨਵੰਬਰ": 11, "ਦਸੰਬਰ": 12,
        },
        "year_leads": {"ਦੋ ਹਜ਼ਾਰ": 2000, "ਦੋ ਹਜਾਰ": 2000},
        "year_units": {
            "ਇੱਕ": 1, "ਦੋ": 2, "ਤਿੰਨ": 3, "ਚਾਰ": 4, "ਪੰਜ": 5, "ਛੇ": 6,
            "ਸੱਤ": 7, "ਅੱਠ": 8, "ਨੌਂ": 9,
        },
    },
    "ta-IN": {
        "days": {
            "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4,
            "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8,
            "ஒன்பது": 9, "பத்து": 10, "பதினொன்று": 11,
            "பன்னிரண்டு": 12, "பதின்மூன்று": 13, "பதினான்கு": 14,
            "பதினைந்து": 15, "பதினாறு": 16, "பதினேழு": 17,
            "பதினெட்டு": 18, "பத்தொன்பது": 19, "இருபது": 20,
            "இருபத்தொன்று": 21, "இருபத்திரண்டு": 22,
            "இருபத்திமூன்று": 23, "இருபத்திநான்கு": 24,
            "இருபத்தியைந்து": 25, "இருபத்தாறு": 26,
            "இருபத்திஏழு": 27, "இருபத்தெட்டு": 28,
            "இருபத்தொன்பது": 29, "முப்பது": 30, "முப்பத்தொன்று": 31,
        },
        "months": {
            "ஜனவரி": 1, "பிப்ரவரி": 2, "மார்ச்": 3, "ஏப்ரல்": 4,
            "மே": 5, "ஜூன்": 6, "ஜூலை": 7, "ஆகஸ்ட்": 8, "ஆகஸ்டு": 8,
            "செப்டம்பர்": 9, "அக்டோபர்": 10, "நவம்பர்": 11, "டிசம்பர்": 12,
        },
        "year_leads": {"இரண்டு ஆயிரம்": 2000, "இரண்டு ஆயிரத்து": 2000},
        "year_units": {
            "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4,
            "ஐந்து": 5, "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9,
        },
    },
    "te-IN": {
        "days": {
            "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
            "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
            "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13,
            "పద్నాలుగు": 14, "పదిహేను": 15, "పదహారు": 16,
            "పదిహేడు": 17, "పదియెనిమిది": 18, "పందొమ్మిది": 19,
            "ఇరవై": 20, "ఇరవై ఒకటి": 21, "ఇరవై రెండు": 22,
            "ఇరవై మూడు": 23, "ఇరవై నాలుగు": 24, "ఇరవై ఐదు": 25,
            "ఇరవై ఆరు": 26, "ఇరవై ఏడు": 27, "ఇరవై ఎనిమిది": 28,
            "ఇరవై తొమ్మిది": 29, "ముప్పై": 30, "ముప్పై ఒకటి": 31,
        },
        "months": {
            "జనవరి": 1, "ఫిబ్రవరి": 2, "మార్చి": 3, "మార్చ్": 3,
            "ఏప్రిల్": 4, "మే": 5, "జూన్": 6, "జులై": 7, "ఆగస్టు": 8,
            "ఆగష్టు": 8, "సెప్టెంబర్": 9, "అక్టోబర్": 10, "నవంబర్": 11,
            "డిసెంబర్": 12,
        },
        "year_leads": {"రెండు వెయ్యి": 2000},
        "year_units": {
            "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
            "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
        },
    },
}


# ---- spoken digit words for the phone collector ------------------------------
#
# Zero words per locale: the single-digit 1-9 words are reused from the
# reviewed ``_SPOKEN_DATE_TABLES`` day entries (the same words count digits
# in these languages), so only zero needs its own explicit vocabulary. Both
# attested spellings are listed for Punjabi, Tamil and Telugu.
_LOCALE_ZERO_DIGIT_WORDS = {
    "bn-IN": ["শূন্য"],
    "gu-IN": ["શૂન્ય"],
    "kn-IN": ["ಶೂನ್ಯ"],
    "ml-IN": ["പൂജ്യം"],
    "mr-IN": ["शून्य"],
    "pa-IN": ["ਸਿਫ਼ਰ", "ਸਿਫਰ"],
    "ta-IN": ["பூஜ்ஜியம்", "பூஜ்யம்"],
    "te-IN": ["సున్నా", "శూన్యం"],
}

# Malayalam transliterations of the ENGLISH digit words, as STT emits them
# when a Malayalam speaker recites digits in English ("നൈൻ ടു ടു ഫോർ ഫോർ
# വൺ നൈൻ വൺ സീറോ ത്രീ" -> "9224419103"), mirroring the Devanagari
# transliterated-English table. "വൺ"=1 and "ത്രീ"=3 are confirmed by that
# browser transcript's positional decoding; the unverified forms
# ("ഫൈവ്", "സിക്സ്", "സെവൻ") stay unmapped rather than guessed.
# ("എട്ട്" is already the reviewed native 8.)
_EXTRA_DIGIT_WORDS: Dict[str, Dict[str, str]] = {
    "ml-IN": {"നൈൻ": "9", "ടു": "2", "ഫോർ": "4", "സീറോ": "0", "വൺ": "1", "ത്രീ": "3"},
}

# locale -> {digit word: digit}, explicit and independently testable.
_LOCALE_DIGIT_WORDS = {
    locale: {
        **{word: str(value) for word, value in table["days"].items() if value <= 9},
        **{word: "0" for word in _LOCALE_ZERO_DIGIT_WORDS.get(locale, [])},
        **_EXTRA_DIGIT_WORDS.get(locale, {}),
    }
    for locale, table in _SPOKEN_DATE_TABLES.items()
}

# Flat lookup: every supported script's words are mutually distinct, and the
# one shared script (Devanagari: hi-IN/mr-IN) shares words with equal values
# ("एक" 1, "तीन" 3, "चार" 4, "सात" 7, "आठ" 8), so one merged map is
# unambiguous.
_ALL_LOCALE_DIGIT_WORDS = {
    word: digit
    for words in _LOCALE_DIGIT_WORDS.values()
    for word, digit in words.items()
}

# A FAILED date attempt at the end of a combined name+date-of-birth answer
# (the observed "राजू शर्मा बीस में दो हजार": the month word was
# misrecognised, so the date cannot parse) must never contaminate the name.
# ट्वेंटी ("twenty") is the attested transliterated multi-digit day word
# (the live retry used it); the mapping is used ONLY by the failed-attempt
# detector, never by date parsing itself.
# The tail is identifiable without knowing the month: a day word (or digit)
# followed by unparseable-month tokens and closed by a year-lead phrase
# (with an optional unit word). The DOB itself stays unset - one bounded
# clarification asks for the date again - and no letter or number is ever
# invented.
_FAILED_DATE_YEAR_LEADS = frozenset(
    {"दो हजार", "टू थाउजंट"}  # the Hindi parser's hardcoded 2000 lead and the
    # observed Devanagari transliteration of "two thousand"
    | {lead for table in _SPOKEN_DATE_TABLES.values() for lead in table["year_leads"] if " " in lead}
)
_FAILED_DATE_UNIT_WORDS = frozenset(
    {unit for table in _SPOKEN_DATE_TABLES.values() for unit in table["year_units"]}
) | {"हजार"}


_TRANSLITERATED_MULTI_DIGIT_DAY_WORDS = {"ट्वेंटी": 20}


def _failed_date_attempt_span(text: str) -> Optional[tuple[int, int]]:
    """Character span ``(start, end)`` of a failed date attempt at the end
    of ``text``, or ``None``. Only a day-word-anchored tail closed by a
    year-lead phrase qualifies; any other shape is never stripped."""
    raw_tokens = [
        token.strip(".,!?()\"'।")
        for token in (text or "").replace("।", " ").split()
    ]
    tokens = [token.replace("़", "") for token in raw_tokens]
    count = len(tokens)
    if count < 3:
        return None
    # Find a year-lead phrase ending the utterance (allowing one trailing
    # unit word), then walk backwards over the allowed date-attempt tokens
    # to the day word. At least one token must precede the day (the name).
    for lead in _FAILED_DATE_YEAR_LEADS:
        lead_tokens = lead.split()
        for lead_start in range(count - len(lead_tokens), -1, -1):
            if " ".join(tokens[lead_start : lead_start + len(lead_tokens)]) != lead:
                continue
            walk_end = lead_start + len(lead_tokens)
            if walk_end < count:
                tail_unit = tokens[walk_end]
                if tail_unit in _FAILED_DATE_UNIT_WORDS or tail_unit.isdigit():
                    walk_end += 1
            if walk_end != count:
                continue  # the lead phrase must close the utterance
            index = lead_start - 1
            saw_day = False
            while index >= 0:
                token = tokens[index]
                is_day = (
                    token in _ALL_LOCALE_DIGIT_WORDS
                    or token.isdigit()
                    or token in _DEVANAGARI_NUMBERS
                    or token in _TRANSLITERATED_DIGIT_WORDS
                    or token in _TRANSLITERATED_MULTI_DIGIT_DAY_WORDS
                )
                if is_day:
                    day_value = int(token) if token.isdigit() else int(
_ALL_LOCALE_DIGIT_WORDS.get(
                            token,
                            _DEVANAGARI_NUMBERS.get(
                                token,
                                _TRANSLITERATED_MULTI_DIGIT_DAY_WORDS.get(
                                    token,
                                    int(_TRANSLITERATED_DIGIT_WORDS.get(token, 0))
                                ),
                            ),
                            )
                    )
                    if not 1 <= day_value <= 31:
                        break
                    saw_day = True
                    break
                if token in _HINDI_MONTH_VARIANTS:
                    index -= 1
                    continue
                break
            if not saw_day or index <= 0:
                continue  # no day word, or nothing would remain as the name
            start_char = len(" ".join(raw_tokens[:index]))
            if start_char:
                start_char += 1
            return start_char, len(text)
    return None


# Speech particles that STT commonly appends after the year ("पच्चीस मई दो
# हजार है"). A particle right after "दो हजार" is filler, never part of the
# date; it is skipped and the year unit (if any) is read after it.
_HINDI_DATE_FILLERS = frozenset({"है"})



def _ends_with_failed_date_attempt(text: str) -> bool:
    """Whether a pending-DOB reply is a recognizable but unparseable date
    attempt: a day word (or digit), optionally followed by the known
    unparseable-month variants ("मे/में"), closed by a year-lead phrase
    (with an optional unit word). The month itself is never guessed — this
    only tells the workflow to ask a specific clarification (say the date
    as day, month, and year) instead of the generic re-ask. Unlike
    ``_failed_date_attempt_span`` there is no name before the day, so a
    bare retry ("बीस में दो हजार") qualifies too."""
    raw_tokens = [
        token.strip(".,!?()\"'।")
        for token in (text or "").replace("।", " ").split()
    ]
    tokens = [token.replace("़", "") for token in raw_tokens]
    count = len(tokens)
    if count < 2:
        return False
    for lead in _FAILED_DATE_YEAR_LEADS:
        lead_tokens = lead.split()
        for lead_start in range(count - len(lead_tokens), -1, -1):
            if " ".join(tokens[lead_start : lead_start + len(lead_tokens)]) != lead:
                continue
            walk_end = lead_start + len(lead_tokens)
            if walk_end < count:
                tail_unit = tokens[walk_end]
                if tail_unit in _FAILED_DATE_UNIT_WORDS or tail_unit.isdigit():
                    walk_end += 1
            if walk_end != count:
                continue  # the lead phrase must close the utterance
            index = lead_start - 1
            while index >= 0:
                token = tokens[index]
                if (
                    token in _ALL_LOCALE_DIGIT_WORDS
                    or token in _DEVANAGARI_NUMBERS
                    or token in _TRANSLITERATED_DIGIT_WORDS
                    or token in _TRANSLITERATED_MULTI_DIGIT_DAY_WORDS
                    or token.isdigit()
                ):
                    day_value = (
                        int(token)
                        if token.isdigit()
                        else int(
_ALL_LOCALE_DIGIT_WORDS.get(
                                token,
                                _DEVANAGARI_NUMBERS.get(
                                    token,
                                    _TRANSLITERATED_MULTI_DIGIT_DAY_WORDS.get(
                                        token,
                                        int(_TRANSLITERATED_DIGIT_WORDS.get(token, 0))
                                    ),
                                ),
                                )
                        )
                    )
                    if 1 <= day_value <= 31:
                        return True
                    break
                if token in _HINDI_MONTH_VARIANTS:
                    index -= 1
                    continue
                break
    return False



def _extract_locale_dob(
    text: str, table: Dict[str, Dict[str, Any]]
) -> tuple[Optional[date], Optional[str]]:
    """A spoken date of birth for one locale table, mirroring
    ``_extract_hindi_dob``'s conservative contract: a complete
    "<day> <month> <year>" phrase is required; incomplete or ambiguous forms
    yield ``(None, None)`` and nothing is ever inferred. The day and the year
    may each be a plain digit token, because STT commonly transcribes spoken
    numbers as digits even in Indic-script transcripts."""
    days: Dict[str, int] = table["days"]
    months: Dict[str, int] = table["months"]
    year_leads: Dict[str, int] = table["year_leads"]
    year_units: Dict[str, int] = table["year_units"]

    raw_tokens = [
        token.strip(".,!?()\"'\u0964")
        for token in (text or "").replace("\u0964", " ").split()
    ]
    tokens = [
        token.replace("\u093c", "").replace("\u09bc", "") for token in raw_tokens
    ]

    def _day_before(index: int) -> tuple[Optional[int], Optional[int]]:
        # Longest (two-token) day form first, then the single-token form
        # ("ఇరవై ఒకటి" vs "ఇరవై"). A digit token ("20") is the spoken day
        # the STT wrote as a number; the caller's 1..31 check bounds it.
        if index >= 2:
            day = days.get(tokens[index - 2] + " " + tokens[index - 1])
            if day is not None:
                return day, index - 2
        if index >= 1:
            token = tokens[index - 1]
            day = days.get(token)
            if day is None and token.isdigit():
                day = int(token)
            if day is not None:
                return day, index - 1
        return None, None

    def _year_from(year_start: int, month: int, day: int, day_start: int):
        """The year beginning at ``year_start`` (a 4-digit token, or a
        spoken lead plus optional unit), or ``None`` when it does not
        form a complete valid date with the day and month."""
        if year_start >= len(tokens):
            return None
        year_token = tokens[year_start]
        digit_year = year_token.isdigit() and len(year_token) == 4
        base = None
        lead_end = None
        if digit_year:
            base = int(year_token)
            lead_end = year_start
        else:
            two = " ".join(tokens[year_start : year_start + 2])
            if two in year_leads:
                base = year_leads[two]
                lead_end = year_start + 1
            elif year_token in year_leads:
                base = year_leads[year_token]
                lead_end = year_start
        if base is None:
            return None
        # An optional unambiguous unit word follows the year lead
        # ("… two thousand five"): 2001-2009 only. An exact digit year is
        # already complete, so no unit is ever added to it.
        unit = 0
        if not digit_year and lead_end + 1 < len(tokens):
            unit_two = " ".join(tokens[lead_end + 1 : lead_end + 3])
            if unit_two in year_units:
                unit = year_units[unit_two]
                lead_end = lead_end + 2
            elif tokens[lead_end + 1] in year_units:
                unit = year_units[tokens[lead_end + 1]]
                lead_end = lead_end + 1
        year = base + unit
        if not _dob_year_ok(year):
            return None
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        return parsed, " ".join(raw_tokens[day_start : lead_end + 1])

    for index, token in enumerate(tokens):
        month = months.get(token)
        if month is None:
            continue
        # Order A (the documented "<day> <month> <year>"): the day
        # precedes the month.
        day, day_start = _day_before(index)
        if day is not None and 1 <= day <= 31 and index + 1 < len(tokens):
            resolved = _year_from(index + 1, month, day, day_start)
            if resolved is not None:
                return resolved
        # Order B (month-first, the observed Marathi "<month> <day> <year>"
        # answer): the day follows the month and the year follows the day,
        # with the same conservative year rules; nothing is ever guessed.
        # The verbatim span starts at the month token, mirroring Order A's
        # month-inclusive span.
        if index + 1 < len(tokens):
            token_after = tokens[index + 1]
            day_after = days.get(token_after)
            if day_after is None and token_after.isdigit():
                day_after = int(token_after)
            if (
                day_after is not None
                and 1 <= day_after <= 31
                and index + 2 < len(tokens)
            ):
                resolved = _year_from(index + 2, month, day_after, index)
                if resolved is not None:
                    return resolved
    return None, None


def _extract_dob_date(text: str) -> tuple[Optional[date], Optional[str]]:
    """Return the calendar date of birth found in ``text`` and its verbatim form.

    Only a fully specified, valid calendar date is returned; "born 1990"
    (year only) yields ``(None, ...)`` because a day and month are never
    invented. Implausible values (unknown month, day/month out of range, year
    outside 1900..current year) are rejected rather than guessed at.
    """
    hindi_dob, hindi_verbatim = _extract_hindi_dob(text)
    if hindi_dob is not None:
        return hindi_dob, hindi_verbatim
    for table in _SPOKEN_DATE_TABLES.values():
        locale_dob, locale_verbatim = _extract_locale_dob(text, table)
        if locale_dob is not None:
            return locale_dob, locale_verbatim
    match = _DOB_MONTH_DATE.search(text)
    if match is not None:
        month = _MONTH_NAMES.get(match.group(2).lower().rstrip("."))
        year = int(match.group(3))
        if month is not None and _dob_year_ok(year):
            try:
                parsed = date(year, month, int(match.group(1)))
            except ValueError:
                pass  # e.g. "31 February": fall through to the other formats
            else:
                return parsed, match.group(0)
    match = _DOB_SLASH.search(text)
    if match is not None:
        day, month, year = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if _dob_year_ok(year):
            try:
                parsed = date(year, month, day)
            except ValueError:
                pass  # e.g. "31/02/1990": fall through to the other formats
            else:
                return parsed, match.group(0)
    match = _DOB_DMY_DASH.search(text)
    if match is not None:
        day, month, year = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if _dob_year_ok(year):
            try:
                parsed = date(year, month, day)
            except ValueError:
                pass
            else:
                return parsed, match.group(0)
    match = _DOB_DASH.search(text)
    if match is not None:
        year, month, day = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if _dob_year_ok(year):
            try:
                parsed = date(year, month, day)
            except ValueError:
                pass
            else:
                return parsed, match.group(0)
    # Spoken-English form ("twenty seven may two thousand"). Every component
    # must parse; nothing is guessed and only the parsed components are kept.
    # A rejected candidate must not consume an overlapping one: the greedy
    # two-word day group can swallow a lead-in word ("…is twenty May two
    # thousand" first matches with day "is twenty"), so the search restarts
    # one character later and the true date is examined next. Validation is
    # unchanged - only fully valid candidates are ever returned.
    pos = 0
    while True:
        match = _SPOKEN_DOB.search(text, pos)
        if match is None:
            break
        day = _spoken_day(match.group(1))
        month = _MONTH_NAMES.get(match.group(2).lower())
        year, year_text = _spoken_year(match.group(3))
        if day is None or month is None or year is None or not _dob_year_ok(year):
            pos = match.start() + 1
            continue
        try:
            parsed = date(year, month, day)
        except ValueError:
            pos = match.start() + 1
            continue
        return parsed, f"{match.group(1)} {match.group(2)} {year_text}"
    return None, None


def normalized_dob_in(text: str) -> Optional[str]:
    """The date of birth in ``text`` normalized to ``DD-MMM-YYYY``, or None."""
    parsed, _verbatim = _extract_dob_date(text or "")
    if parsed is None:
        return None
    return format_report_date(parsed)


def _extract_dob_with_year(text: str) -> tuple[Optional[str], Optional[int]]:
    """Return the verbatim DOB found in ``text`` and its birth year.

    Supports DD Month YYYY (with named months only), DD/MM/YYYY, DD-MM-YYYY,
    YYYY-MM-DD, "born YYYY", and the spoken-English form "twenty seven may
    two thousand". Callers that STORE a DOB must use ``normalized_dob_in``
    instead so structured case data is always ``DD-MMM-YYYY``.
    """
    parsed, verbatim = _extract_dob_date(text)
    if parsed is not None:
        return verbatim, parsed.year
    match = _DOB_BORN.search(text)
    if match is not None and _dob_year_ok(int(match.group(1))):
        return match.group(0), int(match.group(1))
    return None, None


def _extract_dob(text: str) -> str | None:
    """The verbatim date of birth in ``text``, or ``None``."""
    return _extract_dob_with_year(text)[0]


def _dob_conflicts_with_age(dob_year: Optional[int], age: str) -> bool:
    """Whether a DOB and a stated age disagree by more than a year."""
    if dob_year is None:
        return False
    try:
        age_value = int(age)
    except (TypeError, ValueError):
        return False
    if not 1 <= age_value <= _MAX_AGE:
        return False
    return abs(dob_year - (date.today().year - age_value)) > 1


def _normalize_product(text: str) -> str | None:
    """Return the canonical product name when the text names a known variant."""
    if _PRODUCT_VARIANTS.search(text):
        return _PRODUCT_CANONICAL
    return None


def _parse_initial_report(text: str) -> Dict[str, Any]:
    """Extract product and the person's own description from a first report.

    An obvious question is never a first report: nothing is extracted, so the
    workflow simply asks for the event description again instead of opening a
    report from the question itself.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    fields: Dict[str, Any] = {}

    if _is_obvious_question(stripped):
        return fields

    product = _normalize_product(stripped)
    if product:
        fields["product"] = product

    if _NON_ANSWER.match(stripped) or _SIMPLE_YESNO.match(stripped):
        return fields

    looks_like_report = bool(
        _REPORT_LANGUAGE.search(lowered) or _SYMPTOM_LANGUAGE.search(lowered)
    )
    if looks_like_report:
        # Preserve the reporter's own wording faithfully; never overwrite later.
        fields["adverse_event"] = stripped
    elif product:
        # Product-anchored capture: keep the full wording verbatim only when
        # the message says something substantive beyond the medicine name
        # itself, so a bare product mention or product question ("सोलो स्टार
        # के बारे में") never becomes a fabricated event description. Split
        # on whitespace (not \w: Devanagari matras break \w tokenisation).
        remainder = _PRODUCT_VARIANTS.sub("", stripped).strip()
        if len(remainder.split()) >= 4 or _REPORT_LANGUAGE.search(remainder.lower()):
            fields["adverse_event"] = stripped
    return fields


def _outcome_value(lowered: str) -> str | None:
    """Map outcome wording to its structured value, or ``None`` if unmentioned."""
    if _OUTCOME_FATAL.search(lowered):
        return "fatal"
    worse = _OUTCOME_WORSE.search(lowered)
    if worse is not None:
        preceding = lowered[max(0, worse.start() - 25) : worse.start()]
        if _WORSE_NEGATION.search(preceding):
            return "not_recovered"
        return "worsening"
    if _OUTCOME_NOT_RECOVERED.search(lowered):
        return "not_recovered"
    if _OUTCOME_SAME.search(lowered):
        # "still ongoing" / "hasn't changed" is reported as no recovery.
        return "not_recovered"
    if _OUTCOME_RECOVERED.search(lowered):
        return "recovered"
    if _OUTCOME_RECOVERING.search(lowered):
        return "recovering"
    return None


def _seriousness_value(lowered: str) -> str | None:
    """Map one utterance to ``yes``/``no``/``unknown``, or ``None``.

    ``None`` means the utterance makes no clear seriousness claim. A mixed,
    ambiguous claim (a serious statement bundled with an unrelated negation,
    e.g. "it is serious for me but I don't ...") returns ``"ambiguous"`` so
    callers can ask instead of guessing; it is never mapped to ``"no"``.
    """
    if not _SERIOUS_TOPIC.search(lowered):
        return None
    if _UNCERTAIN.search(lowered):
        return "unknown"
    if _SERIOUS_NEGATED.search(lowered):
        return "no"
    if _NEGATION.search(lowered):
        # e.g. "it is serious for me but I don't ..." - not a denial of the
        # seriousness claim itself, so never stored as "no".
        return "ambiguous"
    return "yes"


_VOLUNTEER_CUE = re.compile(
    r"\b(?:also|as\s+well|too|plus|additionally|another|and\s+i|bhi)\b|भी",
    re.IGNORECASE,
)
_DOSE_VALUE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:units?|iu|mg|ml|mcg|micrograms?)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|thirty|forty|fifty)\s+units?\b",
    re.IGNORECASE,
)
_PRESCRIBER = re.compile(
    r"\b(?:prescribed|recommended|given)\s+by\s+"
    r"(?:dr\.?|doctor|physician|my\s+doctor|my\s+physician)\s*"
    r"([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})",
    re.IGNORECASE,
)
# A volunteered concomitant medication ("I also take metformin", "along with
# Lantus I take metformin"): framed as ADDITIONAL treatment, never as the
# suspect cause - the suspect product is answered through the product field.
_CONCOMITANT_MED = re.compile(
    r"\b(?:also|i'?m\s+also|plus)\s+(?:i\s+|we\s+)?(?:currently\s+)?"
    r"(?:taking|take|takes|using|use|uses)\s+(?P<med>[a-z][a-z.-]{2,})\b"
    r"|\b(?:along\s+with|together\s+with)[^.]{0,40}?"
    r"(?:i\s+)?(?:take|taking|takes|use|using|am\s+on)\s+(?P<med2>[a-z][a-z.-]{2,})\b"
    r"|\bi'?m\s+also\s+on\s+(?P<med3>[a-z][a-z.-]{2,})\b",
    re.IGNORECASE,
)
# Vague words that are never a medicine name.
_CONCOMITANT_STOPWORDS = frozenset(
    {
        "care", "food", "medicine", "medicines", "meds", "pill", "pills",
        "tablet", "tablets", "drug", "drugs", "water", "the", "a", "an",
    }
)


def _extract_concomitant_med(text: str) -> Optional[str]:
    """The volunteered concomitant medicine in ``text``, or ``None``.

    Conservative: one medicine per reply, never a vague word, and never the
    suspect product itself ("I also take Lantus" is not a concomitant).
    """
    match = _CONCOMITANT_MED.search((text or "").strip().lower())
    if match is None:
        return None
    name = next((value for value in match.groupdict().values() if value), None)
    if not name or name in _CONCOMITANT_STOPWORDS:
        return None
    if _normalize_product(name) is not None:
        return None
    return name


def _volunteered_symptom(text: str) -> str | None:
    """Return a short volunteered symptom phrase, or None.

    The original event description is never replaced; a later, clearly
    additive symptom statement is what this recognises so it can be appended
    instead of lost. Onset wording, outcome wording, medical-care statements,
    negations, uncertainty and obvious questions are excluded so those
    answers are not mistaken for new symptoms.
    """
    lowered = text.lower()
    if _is_obvious_question(text):
        return None
    if not _SYMPTOM_LANGUAGE.search(lowered):
        return None
    if (
        _NEGATION.search(lowered)
        or _UNCERTAIN.search(lowered)
        or _UNKNOWN_ANSWER.search(lowered)
        or _ONSET.search(lowered)
        or _MEDICAL_CARE_NO.search(lowered)
        or _MEDICAL_CARE_YES.search(lowered)
        or _outcome_value(lowered) is not None
    ):
        return None
    stripped = text.strip().rstrip(" .!?")
    if _VOLUNTEER_CUE.search(lowered):
        return stripped
    if len(stripped.split()) <= 6:
        return stripped
    return None


def _clean_name_value(text: str) -> str:
    """Strip conversational filler ("sure, it's") from a captured name."""
    value = re.sub(
        r"(?i)^(?:sure|yes|yeah|yep|ok|okay|of\s+course|well|um|uh|so)[,\s]+",
        "",
        text.strip(),
        count=1,
    )
    value = re.sub(
        r"(?i)^(?:my\s+name\s+is|name\s+is|this\s+is|it'?s|i'?m|i\s+am)\s+",
        "",
        value,
        count=1,
    )
    # Hindi wrappers, mirroring the correction-path patterns in the workflow
    # (_HINDI_NAME_LEAD / _HINDI_NAME_TRAIL): a leading affirmative, the
    # "मेरा नाम ..." lead and a trailing "है" are never part of the name
    # ("ज़रूर मेरा नाम बिट्टू जोरदार है" -> "बिट्टू जोरदार"). The lead also
    # covers the "है" form ("मेरा नाम है राजू…"), its glued speech-recognition
    # variant ("नामहै"), and common insertions ("मेरा सही नाम राजू…"); the
    # same pattern lives in the workflow's correction cleaner.
    value = re.sub(
        r"^\s*(?:जी[,\s]+)?(?:ज़रूर|जरूर)\s*[,\s]*",
        "",
        value,
        count=1,
    )
    value = re.sub(
        r"^(?:जी[,\s]+)?(?:मेरा\s+)?(?:सही\s+|शुद्ध\s+|सही\s+वाला\s+)?नाम\s*(?:है|हैं)?(?:[\s,]+|$)",
        "",
        value,
        count=1,
    )
    value = re.sub(r"\s+है[.!?।]*$", "", value)
    # Malayalam name lead: "എന്റെ പേരാണ് രാജു …" -> "രാജു …" (the observed
    # glued form "പേരാണ്", the bare "പേര്" lead, and the standard
    # "പേര് ആണ്" are all stripped, including the chillu spellings).
    value = re.sub(r"^\s*എൻ്റെ\s+പേരാണ്\s*", "", value, count=1)
    value = re.sub(r"^\s*എൻ്റെ\s+പേര്\s*", "", value, count=1)
    value = re.sub(r"^\s*എന്റെ\s+പേരാണ്\s*", "", value, count=1)
    value = re.sub(r"^\s*എന്റെ\s+പേര്\s*", "", value, count=1)
    value = re.sub(r"^\s*എന്റെ\s+പേര്\s*ആണ്\s*", "", value, count=1)
    # Marathi name lead: "माझं नाव Raju Patil आहे" -> "Raju Patil" (the
    # possessive leads "माझं/माझे/माझा" and the trailing copula "आहे" are
    # never part of the name; mixed-script names keep every Latin letter
    # and native-script letter of the name itself).
    value = re.sub(r"^\s*माझं\s+नाव\s*", "", value, count=1)
    value = re.sub(r"^\s*माझे\s+नाव\s*", "", value, count=1)
    value = re.sub(r"^\s*माझा\s+नाव\s*", "", value, count=1)
    value = re.sub(r"\s+आहे[.!?।]*$", "", value)
    return value.strip().rstrip(" .!?")


def _parse_answer(text: str, allow_bare_dob: bool = False) -> Dict[str, Any]:
    """Map one utterance to whatever structured AE fields it plainly conveys.

    ``allow_bare_dob`` should be set when the age/date-of-birth question is
    the workflow's pending question, so a bare complete date answers it
    verbatim.
    """
    stripped = text.strip()
    lowered = stripped.lower()
    fields: Dict[str, Any] = {}

    def record(key: str, value: Any) -> None:
        fields.setdefault(key, value)

    email = _normalize_spoken_email(stripped)
    if email:
        record("contact_email", email)

    if _PHONE.fullmatch(stripped):
        candidate = stripped if stripped.isdigit() else "".join(_DIGITS.findall(stripped))
        if _is_plausible_phone(candidate):
            record("contact_phone", candidate)
    else:
        spoken_phone = _extract_spoken_phone(stripped)
        if spoken_phone and _is_plausible_phone(spoken_phone):
            record("contact_phone", spoken_phone)

    if "consent" in lowered:
        record("consent_to_contact", not bool(_NEGATION.search(lowered)))

    product = _normalize_product(stripped)
    if product:
        record("product", product)
    else:
        other_medicine = _known_other_medicine(stripped)
        if other_medicine is not None and _SUSPECT_MEDICINE_CUE.search(stripped):
            # Another medicine named as the suspected cause ("...paracetamol
            # ke karan"): captured so the workflow's product-scope response
            # applies; never re-attributed to Lantus.
            record("product", other_medicine)

    if _ROLE_OTHER.search(lowered):
        record("reporter_role", "someone_else")
    elif _ROLE_SELF.search(lowered):
        record("reporter_role", "self")

    if _MEDICAL_CARE_NO.search(lowered):
        record("medical_care", "no")
    elif _MEDICAL_CARE_YES.search(lowered):
        record("medical_care", "yes")

    seriousness = _seriousness_value(lowered)
    if seriousness in ("yes", "no", "unknown"):
        record("seriousness", seriousness)
    # An ambiguous claim records nothing here; when seriousness is the pending
    # question the pending-field path asks one concise clarification instead of
    # guessing.

    outcome = _outcome_value(lowered)
    if outcome:
        record("outcome", outcome)

    # Preserve volunteered dose and prescribing-physician details verbatim.
    if _DOSE_VALUE.search(stripped):
        record("dose_and_use", stripped)
    prescriber = _PRESCRIBER.search(stripped)
    if prescriber is not None:
        record("prescribing_physician", stripped)

    # A volunteered concomitant medicine is preserved separately from the
    # suspect product; the original event description is never touched.
    concomitant = _extract_concomitant_med(stripped)
    if concomitant:
        record("concomitant_medications", [concomitant])

    # A clearly additive symptom statement is kept (merged, never replacing the
    # original description) so every reported symptom survives.
    volunteered = _volunteered_symptom(stripped)
    if volunteered:
        record("adverse_event", volunteered)

    age = _extract_age(stripped)
    if age is None:
        # Hindi number words with the strong "उम्र" cue ("मेरी उम्र पच्चीस
        # साल है"); the DOB/conflict machinery below sees the same age value.
        age = _extract_devanagari_age(stripped)
    # A complete date of birth answers the age question verbatim: it is
    # stored as given and never turned into a derived age. A bare date only
    # counts while the age/DOB question is pending; explicit "born" /
    # "date of birth" wording always counts. When one answer carries both a
    # DOB and an age that disagree by more than a year, neither is stored:
    # both are held for one clarification instead of silently choosing one.
    dob_allowed = allow_bare_dob or _DOB_CUE.search(stripped) is not None
    dob: Optional[str] = None
    dob_year: Optional[int] = None
    if dob_allowed:
        parsed_dob, _verbatim = _extract_dob_date(stripped)
        if parsed_dob is not None:
            # Stored DOBs are always normalized to DD-MMM-YYYY (27-May-2000).
            dob = format_report_date(parsed_dob)
            dob_year = parsed_dob.year
        else:
            # Year-only forms ("born 1990") cannot be normalized without a day
            # and month; the verbatim value is kept so nothing is invented.
            dob, dob_year = _extract_dob_with_year(stripped)
    if dob is not None and age is not None and _dob_conflicts_with_age(dob_year, age):
        record("dob_age_conflict", {"dob": dob, "age": age})
    elif dob is not None:
        record("patient_age", dob)
    elif age is not None:
        if not (
            allow_bare_dob
            and _age_rejected_for_dob_question(stripped)
        ):
            record("patient_age", age)

    if _ONSET.search(lowered) and not _DIGITS.search(stripped):
        record("onset_date", stripped)

    identifier = _IDENTIFIER.search(stripped)
    if identifier:
        record("patient_identifier", stripped[identifier.start() : identifier.end()])

    return fields


def _parse_for_pending_field(field_key: str, text: str) -> Dict[str, Any]:
    """Answer the workflow's current question when no field is detectable."""
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return {}
    # A repeat/confusion request answers no field: return nothing so the
    # workflow re-asks instead of storing the request as a name, an event or
    # any other value.
    if _CLARIFY_REPLY.search(stripped):
        return {}

    if field_key in (
        "contact_phone or contact_email",
        "contact_phone",
        "reporter_contact_phone",
        "contact_email",
    ):
        email = _normalize_spoken_email(stripped)
        spoken_phone = _extract_spoken_phone(stripped)
        digits = "".join(_DIGITS.findall(stripped))
        if field_key == "contact_email":
            return {"contact_email": email} if email else {}
        if field_key in ("contact_phone", "reporter_contact_phone"):
            if len(digits) >= _MIN_PHONE_DIGITS and _is_plausible_phone(digits):
                return {field_key: digits}
            if spoken_phone and _is_plausible_phone(spoken_phone):
                return {field_key: spoken_phone}
            return {}
        if email:
            return {"contact_email": email}
        if len(digits) >= _MIN_PHONE_DIGITS and _is_plausible_phone(digits):
            return {"contact_phone": digits}
        if spoken_phone and _is_plausible_phone(spoken_phone):
            return {"contact_phone": spoken_phone}
        return {}
    if field_key == "contact_method":
        if (
            "both" in lowered
            or "either" in lowered
            or "दोनों" in lowered
            or "dono" in lowered
            or ("phone" in lowered and ("email" in lowered or "e-mail" in lowered or "mail" in lowered))
        ):
            return {"preferred_contact_method": "both"}
        if (
            _normalize_spoken_email(stripped)
            or "email" in lowered
            or "e-mail" in lowered
            or "mail" in lowered
            # Hindi: "ईमेल", "ईमेल चलेगा", "ईमेल से", "ईमेल पर".
            or "ईमेल" in lowered
            or "ई-मेल" in lowered
        ):
            return {"preferred_contact_method": "email"}
        if (
            any(
                word in lowered
                for word in ("phone", "call", "number", "mobile", "telephone", "cell")
            )
            # Hindi: "फोन" / "फ़ोन".
            or "फोन" in lowered
            or "फ़ोन" in lowered
        ):
            return {"preferred_contact_method": "phone"}
        return {}
    if field_key == "reporter_role":
        # OTHER is checked first (same order as _parse_answer) so phrases
        # mixing both cues ("मेरी माँ का यही अनुभव है") stay someone_else.
        if _ROLE_OTHER.search(lowered):
            return {"reporter_role": "someone_else"}
        if _ROLE_SELF.search(lowered):
            return {"reporter_role": "self"}
        return {}
    if field_key == "patient_age":
        # A bare 3+ digit answer while the date of birth is pending is a
        # mis-heard date fragment (e.g. "117"), never an age.
        if _age_rejected_for_dob_question(stripped):
            return {}
        age = _extract_age(stripped)
        if age is None:
            # Hindi number words: cued ("पच्चीस साल है") or bare ("पच्चीस"),
            # mirroring the existing bare digit/English-word support.
            age = _extract_devanagari_age(stripped, pending=True)
        if age is None:
            bare = re.sub(r"[\s.,?!]+$", "", stripped)
            age = _normalise_age(bare)
        return {"patient_age": age} if age is not None else {}
    if field_key == "consent_to_contact":
        negation = _NEGATION.search(lowered)
        affirmation = _AFFIRM.search(lowered)
        # A clear negation - one that leads the reply, is the only answer, or
        # ends it emphatically ("sahmat nahi", "bilkul nahi", "absolutely
        # not") - never reads as consent. An affirmation-led reply with a
        # negated idiom in the middle stays a yes ("yes, no problem").
        if negation is not None and (
            affirmation is None
            or negation.start() < affirmation.start()
            or not lowered[negation.end() :].strip(" \t.!?,;।")
        ):
            return {"consent_to_contact": False}
        if affirmation is not None:
            return {"consent_to_contact": True}
        return {}
    if field_key == "seriousness":
        if _UNCERTAIN.search(lowered):
            return {"seriousness": "unknown"}
        # Ambiguous mixed claims ("it is serious for me but I don't ...")
        # return {} so the workflow asks one concise clarification; they are
        # never stored as "no".
        value = _seriousness_value(lowered)
        if value in ("yes", "no", "unknown"):
            return {"seriousness": value}
        if value is None:
            # No serious claim made: a plain yes/no reply answers the question.
            if _NEGATION.search(lowered):
                return {"seriousness": "no"}
            # Hindi: "गंभीर है" / "gambhir hai" states the problem is serious.
            # Negation is checked first, so "gambhir nahi hai" stays "no".
            if _HINDI_SERIOUS.search(lowered):
                return {"seriousness": "yes"}
            if _AFFIRM.search(lowered):
                return {"seriousness": "yes"}
        return {}
    if field_key == "outcome":
        outcome = _outcome_value(lowered)
        if outcome:
            return {"outcome": outcome}
        if _UNKNOWN_ANSWER.search(lowered):
            return {"outcome": "unknown"}
        return {}
    if field_key == "medical_care":
        if _MEDICAL_CARE_NO.search(lowered):
            return {"medical_care": "no"}
        if _MEDICAL_CARE_YES.search(lowered):
            return {"medical_care": "yes"}
        if _UNKNOWN_ANSWER.search(lowered):
            return {"medical_care": "unknown"}
        return {}
    if field_key == "product":
        product = _normalize_product(stripped)
        if product:
            return {"product": product}
        # A known non-Lantus medicine answering the suspect-product question
        # is stored under its canonical name ("पैरासिटामोल" -> "Paracetamol");
        # the workflow's product-scope response then applies.
        other_medicine = _known_other_medicine(stripped)
        if other_medicine is not None:
            return {"product": other_medicine}
        if _UNKNOWN_ANSWER.search(lowered):
            return {"product": "Unknown (to be confirmed)"}
        # A reply in a supported Indic script that names no recognised
        # medicine is unclear speech, never a confirmed different medicine:
        # return nothing so the bounded product clarification/re-ask runs and
        # the case parks for human review instead of terminating out-of-scope.
        if _INDIC_SCRIPT_REPLY.search(stripped):
            return {}
        if _NON_ANSWER.match(stripped) or _SIMPLE_YESNO.match(stripped):
            return {}
        # An uncertain or question-shaped reply is not a product name: ask
        # again rather than guessing (and rather than ending the session on
        # an out-of-scope guess).
        if "?" in stripped or _UNCERTAIN.search(lowered):
            return {}
        if _CONTACT_WORDS.search(lowered):
            return {}
        if _REPORT_LANGUAGE.search(lowered) or len(stripped.split()) > 8:
            return {}
        return {"product": stripped}
    if field_key == "adverse_event":
        if (
            _NON_ANSWER.match(stripped)
            or _SIMPLE_YESNO.match(stripped)
            # An obvious question ("what are the side effects?", a request to
            # repeat) answers no field: the workflow re-asks instead of
            # storing the question as the reporter's own event description.
            or _is_obvious_question(stripped)
        ):
            return {}
        if _UNKNOWN_ANSWER.search(lowered):
            return {"adverse_event": "Unknown (to be confirmed)"}
        if len(stripped) < 3:
            return {}
        return {"adverse_event": stripped}
    if field_key in ("patient_identifier", "reporter_name", "onset_date"):
        if (
            _NON_ANSWER.match(stripped)
            or _SIMPLE_YESNO.match(stripped)
            or _UNKNOWN_ANSWER.search(lowered)
        ):
            return {}
        if field_key in ("patient_identifier", "reporter_name"):
            cleaned = _clean_name_value(stripped)
            # A combined answer (name + date of birth) never stores the date
            # as part of the name, and an introductory phrase such as "and my
            # date of birth is" is never part of the name either.
            dob_value, _year = _extract_dob_with_year(stripped)
            if dob_value:
                if cleaned == dob_value.strip():
                    return {}
                cleaned = cleaned.replace(dob_value, "").strip(" ,;-&")
            cleaned = _DOB_NAME_LEAD.sub("", cleaned).strip(" ,;-&")
            cleaned = _HINDI_DOB_NAME_TAIL.sub("", cleaned).strip(" ,;-&")
            # Re-run the cleaner after the split so wrappers exposed by it
            # ("… है और मेरा जन्मदिन …" -> "… है") are stripped too; the
            # cleaner is idempotent for already-clean values.
            cleaned = _clean_name_value(cleaned)
            if not dob_value:
                # A FAILED date attempt (the observed "राजू शर्मा बीस में
                # दो हजार": the month word was misrecognised) is never part
                # of the name: the day-word + year-lead tail is stripped,
                # the name stays intact, and the date of birth remains
                # unset for one bounded clarification (day, month, year) —
                # nothing is invented.
                span = _failed_date_attempt_span(cleaned)
                if span:
                    cleaned = _clean_name_value(cleaned[: span[0]].strip())
            return {field_key: cleaned} if cleaned else {}
        return {field_key: stripped}
    return {}


class AnswerLLMProvider:
    """Deterministic provider: base interpretation plus answer-field extraction.

    Implements the same ``interpret`` surface the conversational workflow
    consumes. Intent values produced by the wrapped provider are preserved for
    answer turns (typically ``"unknown"``); only ``extracted_fields`` is
    enriched. The one exception is a fresh first report the base provider could
    not classify, which is surfaced as ``"ae_report"`` so the case opens.
    """

    def __init__(self) -> None:
        self._base = MockLLMProvider()

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> LLMResult:
        base = self._base.interpret(message, context)
        if not message.strip():
            return base

        context = context or {}
        collected = context.get("collected") or {}
        fresh = not collected.get("product") and not collected.get("adverse_event")

        if base.intent in _BASE_INTERPRETED:
            # A fresh first report the base provider classified from its fixed
            # phrases still needs the reporter's own complete wording: the
            # canned short event must never replace the real description.
            if base.intent == "ae_report" and fresh:
                initial = _parse_initial_report(message)
                merged = dict(base.extracted_fields or {})
                if _is_obvious_question(message):
                    # A question is never the reporter's own description, even
                    # when the fixed-phrase provider matched its wording: the
                    # canned event is dropped instead of being replaced by the
                    # question text.
                    merged.pop("adverse_event", None)
                elif "adverse_event" in merged or "adverse_event" in initial:
                    merged["adverse_event"] = message.strip()
                for key, value in initial.items():
                    merged.setdefault(key, value)
                base.extracted_fields = merged
            return base

        pending = context.get("next_field")
        fields = _parse_answer(
            message,
            allow_bare_dob=bool(pending)
            and pending[0] in ("patient_age", "patient_identifier"),
        )
        # A number spoken while the reporter's own contact number is the
        # pending question is the reporter's, never the patient's: the two
        # identities stay separate on the report.
        if (
            pending
            and pending[0] == "reporter_contact_phone"
            and fields.get("contact_phone")
        ):
            fields["reporter_contact_phone"] = fields.pop("contact_phone")

        # A fresh first report may not match the base provider's fixed phrases
        # (e.g. a misheard product name). Extract product + description so the
        # case opens instead of looping on "which product?".
        if fresh:
            for key, value in _parse_initial_report(message).items():
                fields.setdefault(key, value)

        if not fields:
            if pending:
                fields = _parse_for_pending_field(pending[0], message)
        elif (
            pending
            and pending[0] == "patient_identifier"
            and "adverse_event" not in fields
            and "product" not in fields
        ):
            # The name-and-date-of-birth question may be answered with both
            # values at once: the parser above captured the date, so the name
            # is filled in from the same answer (never the other way round).
            for key, value in _parse_for_pending_field(
                "patient_identifier", message
            ).items():
                fields.setdefault(key, value)

        # An existing medical-care "yes" is never downgraded to "no"/"unknown"
        # by a later answer unless the reporter explicitly corrects it.
        if (
            fields.get("medical_care") in ("no", "unknown")
            and collected.get("medical_care") == "yes"
            and not _EXPLICIT_CORRECTION.search(message)
        ):
            fields = {
                key: value for key, value in fields.items() if key != "medical_care"
            }

        if not fields:
            return base

        merged = dict(base.extracted_fields or {})
        merged.update(fields)
        base.extracted_fields = merged
        if (
            fresh
            and not _is_obvious_question(message)
            and ("product" in merged or "adverse_event" in merged)
        ):
            base.intent = "ae_report"
        return base