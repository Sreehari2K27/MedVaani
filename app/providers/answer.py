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
    r"|नहीं|नही",
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
    r"|सही\s+है|सही|जरूर|ज़रूर",
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
    r"\b(lantern|lantus|lantis|lantos|lanthus|lan\s+tus|solo\s*star|solostar"
    r"|lanta\s+sol(?:o|ar)\s+star|lantus\s+sol(?:o|ar)\s+star)\b"
    r"|सोलो\s*स्टार"
    # Narrow Devanagari transliterations of "Lantus" as STT emits them;
    # exact aliases only, no fuzzy matching (matras break \b, so bare
    # Devanagari like the सोलो स्टार token above).
    r"|लेंटस|लैंटस|लैन्टस|लेन्टस",
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
    r"|पता\s+नहीं|नहीं\s+पता|याद\s+नहीं|नहीं\s+याद",
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
    r"|^\s*(?:मैं|मै)\s+(?:हूँ|हूं)\s*[.!?]?\s*$",
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
    r"|^\s*(?:koi\s+aur|कोई\s+और)\s*[.!?]?\s*$",
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
    "नाइन": "9",
    "जीरो": "0",
    "ज़ीरो": "0",
}

# Keep Latin and Devanagari letters; everything else (digits, punctuation)
# separates tokens. The Devanagari danda "।"/"॥" lives inside the Devanagari
# block but is punctuation, so it is replaced before splitting.
_SPOKEN_PHONE_TOKENS = re.compile(r"[^a-zऀ-ॿ]+")


def _spoken_digit_lookup(token: str) -> str | None:
    """The single digit a spoken token dictates, or ``None``."""
    if token in _SPOKEN_DIGIT_WORDS:
        return _SPOKEN_DIGIT_WORDS[token]
    if token in _HINDI_ZERO_DIGIT_WORDS:
        return _HINDI_ZERO_DIGIT_WORDS[token]
    # Hindi number words (एक, दो, ... नौ) from the shared table; only
    # single-digit values qualify, never multi-digit ages.
    value = _DEVANAGARI_NUMBERS.get(token)
    if value is not None and value <= 9:
        return str(value)
    return _TRANSLITERATED_DIGIT_WORDS.get(token)


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
            run.append(digit)
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
    has_at = bool(re.search(r"\bat\b|@", work))
    has_dot = bool(re.search(r"\bdot\b|\.", work))
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
        if token == "dot":
            out.append(".")
            continue
        if token == "underscore":
            out.append("_")
            continue
        if token in ("hyphen", "dash"):
            out.append("-")
            continue
        digit = _spoken_digit_lookup(token)
        if digit is not None:
            out.append(digit)
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


def _extract_dob_date(text: str) -> tuple[Optional[date], Optional[str]]:
    """Return the calendar date of birth found in ``text`` and its verbatim form.

    Only a fully specified, valid calendar date is returned; "born 1990"
    (year only) yields ``(None, ...)`` because a day and month are never
    invented. Implausible values (unknown month, day/month out of range, year
    outside 1900..current year) are rejected rather than guessed at.
    """
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
    for match in _SPOKEN_DOB.finditer(text):
        day = _spoken_day(match.group(1))
        month = _MONTH_NAMES.get(match.group(2).lower())
        year, year_text = _spoken_year(match.group(3))
        if day is None or month is None or year is None or not _dob_year_ok(year):
            continue
        try:
            parsed = date(year, month, day)
        except ValueError:
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
        record(
            "contact_phone",
            stripped if stripped.isdigit() else "".join(_DIGITS.findall(stripped)),
        )
    else:
        spoken_phone = _extract_spoken_phone(stripped)
        if spoken_phone:
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
            if len(digits) >= _MIN_PHONE_DIGITS:
                return {field_key: digits}
            return {field_key: spoken_phone} if spoken_phone else {}
        if email:
            return {"contact_email": email}
        if len(digits) >= _MIN_PHONE_DIGITS:
            return {"contact_phone": digits}
        return {"contact_phone": spoken_phone} if spoken_phone else {}
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