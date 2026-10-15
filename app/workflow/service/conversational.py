"""Deterministic conversational AE workflow service.

Processing order: USER MESSAGE → LLM interpretation → SAFETY CHECK → AE STATE ENGINE UPDATE → next info → response
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from app.providers.base import LLMProvider, LLMResult
from app.providers.answer import (
    _AFFIRM,
    _EXPLICIT_CORRECTION,
    _NEGATION,
    _NON_ANSWER,
    _SIMPLE_YESNO,
    _UNKNOWN_ANSWER,
    _known_other_medicine,
    _normalize_product,
    _normalize_spoken_email,
    _seriousness_value,
    normalized_dob_in,
)
from app.safety import SafetyLayer, SafetyResult, SafetySignal
from app.workflow.onset import (
    format_iso_date,
    normalize_onset_statement,
    resolve_reference,
)
from app.workflow.ae_engine import (
    AECaseError,
    AEStateEngine,
    WorkflowState,
    _classify_confirmation_reply as _exact_reply_verdict,
    _question_for,
)


@dataclass
class WorkflowTurnResult:
    """Result of one conversational turn."""

    case_id: str
    current_state: str
    intent: str = "unknown"
    extracted_information: Dict[str, Any] = field(default_factory=dict)
    safety_status: str = "ok"
    priority: str = "normal"
    human_review_required: bool = False
    collected_information: Dict[str, Any] = field(default_factory=dict)
    missing_information: List[str] = field(default_factory=list)
    response_text: str = ""
    completed: bool = False
    next_question: Optional[List[str]] = None


def _field_map_from_llm(llm_result: LLMResult) -> Dict[str, Any]:
    """Extract mappable fields from LLM result."""
    fields: Dict[str, Any] = {}
    extracted = llm_result.extracted_fields or {}

    # Map common fields
    for key in (
        "product",
        "adverse_event",
        "onset_date",
        "seriousness",
        "outcome",
        "patient_identifier",
        "patient_age",
        "reporter_role",
        "reporter_name",
        "reporter_contact_phone",
        "contact_phone",
        "contact_email",
        "preferred_contact_method",
        "consent_to_contact",
        "medical_care",
        "dose_and_use",
        "prescribing_physician",
        "concomitant_medications",
    ):
        if key in extracted and extracted[key] is not None:
            fields[key] = extracted[key]

    # Events - if single event string, map to adverse_event if not set
    if "adverse_event" not in fields and llm_result.events:
        # take first event as description
        first = llm_result.events[0]
        if isinstance(first, str):
            fields["adverse_event"] = first

    return fields


def extract_ae_fields_from_llm(llm_result: LLMResult) -> Dict[str, Any]:
    """Public helper to extract AE fields from LLM interpretation."""
    return _field_map_from_llm(llm_result)


# Fields with an explicit read-back confirmation step. Names and critical
# contact values are read back and only committed after the person confirms.
# Opt-in per field so every field not yet integrated keeps the legacy capture
# behaviour.
CONFIRMATION_FIELDS = frozenset(
    {"patient_identifier", "contact_phone", "contact_email"}
)
# Default confirmation scope: only the patient name is confirmed. Phone and
# email confirmation remain available by passing CONFIRMATION_FIELDS (or any
# explicit set) to the workflow constructor.
NAME_CONFIRMATION_FIELDS = frozenset({"patient_identifier"})

# The voice/text conversation's confirmation scope: every identity and
# contact value the documented flow reads back (reporter name and number,
# patient details, patient contact number, email address). The date of birth
# is read back together with the patient name, so it is not confirmed alone.
VOICE_CONFIRMATION_FIELDS = frozenset(
    {
        "patient_identifier",
        "reporter_name",
        "reporter_contact_phone",
        "contact_phone",
        "contact_email",
    }
)

_CONFIRM_FIELD_LABELS = {
    "patient_identifier": "name",
    "reporter_name": "name",
    "contact_phone": "phone number",
    "reporter_contact_phone": "contact number",
    "contact_email": "email address",
}

# ---- documented AE voice conversation steps ---------------------------------
# The wording below is fixed by the flow specification: it is spoken (and
# displayed) verbatim, so it is never assembled or paraphrased elsewhere.

# Step 1 - fixed introductory message + Step 2 invitation, shown once per
# session and never repeated after the person has described the event.
INTRODUCTION_MESSAGE = (
    "Thank you for taking the time to speak with us. "
    "I'm here to help record the event for review. "
    "I'm not a substitute for a healthcare professional, and I can't provide "
    "medical advice. Please tell me what happened, in your own words."
)

# Step 2 - invite the person to describe the event (also the adverse-event
# question, so the invitation is asked whenever no description exists yet).
EVENT_DESCRIPTION_QUESTION = (
    "Please tell me what happened, in your own words. "
    "Take your time\u2014I'm listening."
)

# Step 3 - one-time empathy offered with the medication question.
MEDICATION_EMPATHY = "I'm sorry to hear that you've experienced this."

# Shown after a combined name/date-of-birth read-back is rejected: only the
# incorrect field is asked for again, never the whole question.
DETAILS_CORRECTION_QUESTION = (
    "Was it the name or the date of birth that was incorrect?"
)
DETAILS_CORRECTION_CLARIFICATION = (
    "Sorry, please tell me the name or the date of birth that was incorrect."
)

# The person chose the date of birth to correct but did not supply a full
# date: ask only for the missing date, never restarting the whole question.
INCOMPLETE_DOB_CLARIFICATION = "Sorry, please tell me a full date of birth."

# Field-choice phrases on the details-correction turn ("date of birth",
# "the date", "the name"). A value that actually parses as a date is handled
# as a corrected date before these are consulted.
_DOB_CHOICE = re.compile(
    r"(?i)\b(?:date\s+of\s+birth|birth\s+date|dob|birth|the\s+date|date)\b"
)
_NAME_CHOICE = re.compile(r"(?i)\b(?:the\s+)?name\b")

# A reply that says the captured email is wrong without supplying a new
# address ("the email address is incorrect").
_EMAIL_INCORRECTION = re.compile(
    r"(?i)\b(?:incorrect|wrong|not\s+right|isn'?t\s+right|isn'?t\s+correct|"
    r"not\s+correct|not\s+my\s+address)\b"
)

# A later statement that carries an explicit correction cue could be a
# correction of the original event description rather than another symptom;
# it is clarified once instead of being silently combined with the original.
EVENT_CORRECTION_CLARIFICATION = (
    "Just to check - was that another symptom, or a correction to what you "
    "described?"
)

# Step 4 - verify the medication before the interview continues.
SCOPE_VERIFICATION_QUESTION = (
    "Just to make sure I have the right medication, is it Lantus SoloStar?"
)

# Step 7 - explicit consent to contact the patient about the report.
CONSENT_QUESTION = (
    "Would it be okay for the appropriate team to contact the patient about "
    "this report if any follow-up is needed?"
)

# Step 8 - the email question is conditional on consent; the offer is asked
# first so a "no" skips email collection entirely.
EMAIL_OFFER_QUESTION = (
    "May I have the patient's email address, if available?"
)
EMAIL_REQUEST_QUESTION = (
    "Could you please spell out your email address, one letter or number at a "
    "time? You can say 'at the rate' for @ and 'dot' for a period."
)

# Step 9 - severity / life-threatening screen.
SEVERITY_QUESTION = "Would you say that the event was severe or life-threatening?"

# Step 10A - urgent safety response for a severe or life-threatening answer.
SEVERITY_URGENT_MESSAGE = (
    "Please seek immediate medical attention now. "
    "If you've already consented to follow-up contact, we'll arrange to "
    "collect any additional information at a more convenient time."
)

# Step 12 - closing message; the consent sentence is only added when contact
# consent was not declined (never implying consent that was refused).
CLOSING_MESSAGE = "Thank you for sharing this information with us. We've prepared the report for review."
CLOSING_CONSENT_SENTENCE = (
    "If you provided consent for follow-up contact, the appropriate team may "
    "contact you if needed."
)

# Optional closing question. Opt-in per workflow so every existing caller
# (legacy journey harness included) keeps its current completion behaviour;
# the production voice endpoint enables it explicitly.
ADDITIONAL_INFORMATION_QUESTION = (
    "Is there anything else you'd like to share that may help us document "
    "the event?"
)

# Step 10B.2 follow-up: asked only when the person says yes to the question
# above, so the free-text answer itself is never mistaken for "yes".
ADDITIONAL_INFORMATION_FOLLOWUP = (
    "Of course. Please tell me any additional details you'd like included."
)

# "No", "No, nothing else.", "No thanks", "Nothing more", "That's all" ...
_ADDITIONAL_DECLINE = re.compile(
    r"(?i)^\s*(?:no|nope|nah|nothing|none|nil|that'?s\s+(?:all|it))\b[\s,.!-]*"
    r"(?:nothing\s+(?:else|more)|that'?s\s+(?:all|it)|thanks?|thank\s+you|"
    r"i'?m\s+(?:good|fine|okay|ok)|all\s+good|else)?[\s.!]*$"
)


def _is_decline(message: str) -> bool:
    """Whether a reply to the closing question means "nothing else to add"."""
    stripped = (message or "").strip()
    if not stripped:
        return True
    if _NON_ANSWER.match(stripped):
        return True
    if _ADDITIONAL_DECLINE.match(stripped):
        return True
    if _exact_reply_verdict(stripped) == "no":
        return True
    return bool(_UNKNOWN_ANSWER.search(stripped))


def _is_plain_yes(message: str) -> bool:
    """Whether a reply is only an agreement, carrying no new information.

    "Yes" / "yes, that's right" invite the details; "Yes, I also felt dizzy"
    already carries them and is recorded instead of being turned into a
    second question.
    """
    stripped = (message or "").strip()
    if not stripped:
        return False
    verdict, _ = _classify_confirmation_turn(stripped)
    if verdict != "yes":
        return False
    remainder = _AFFIRM_LEAD.sub("", stripped, count=1).strip(" .,!?")
    if not remainder:
        return True
    return bool(_AFFIRM_CONTINUATION.match(remainder))


# A reply asking us to repeat/clarify rather than confirming or correcting.
# Hindi / Romanized-Hindi equivalents ("dobara" = again, "samajh nahi aaya" =
# didn't understand, "kya" = what).
_CLARIFY_PHRASE = re.compile(
    r"(?i)\b(repeat|again|pardon|sorry|remind\s+me|come\s+again|maybe|perhaps|"
    r"what(?:'s|\s+is)?\s+(?:it|that|the)\b"
    r"|dobara|samajh\s+nahi\s+aaya|kya"
    r"|दोबारा|समझ\s+नहीं\s+आया|क्या)"
)

# "no, it's Beta Jones" -> the correction payload is everything after the no.
_NEGATION_LEAD = re.compile(
    r"(?i)^\s*(?:no|nope|not\s+really|incorrect)[,\s]+"
    r"(?:it'?s|actually|the\s+name\s+is|that(?:'s|\s+is))?\s*(.+)$"
)

# Leading filler before the actual value: "My name is Beta Jones".
_VALUE_LEAD = re.compile(
    r"(?i)^(?:my\s+name\s+is|name\s+is|this\s+is|it(?:'s|\s+is)|i\s*'?m|i\s+am)\s+"
)

# Hindi name wrappers, stripped by ``_clean_correction`` so a fuller
# correction reads back as the name itself: "जी, मेरा नाम जया बच्चन है"
# -> "जया बच्चन".
_HINDI_NAME_LEAD = re.compile(r"^(?:जी[,\s]+)?(?:मेरा\s+नाम|नाम)\s+")
_HINDI_NAME_TRAIL = re.compile(r"\s+है$")

# A leading affirmative token, so a value hiding after it can be inspected:
# "yes alpha beta" is not the same reply as "yes". Hindi forms are stripped
# here too so "haan"/"ji haan" confirm a read-back instead of reading as a
# possible corrected value.
_AFFIRM_LEAD = re.compile(
    r"(?i)^\s*(?:yes|yeah|yep|yup|sure|ok|okay|correct|right|true|fine|confirmed"
    r"|ji haan|haan|han)"
    r"\b[\s,]*"
)
# Continuations that turn the remainder back into a plain affirmation
# ("yes it is", "yes I am sure that's right").
_AFFIRM_CONTINUATION = re.compile(
    r"(?i)^\s*(?:i\s*'?m|i\s+am|it(?:'s|\s+is)|that(?:'s|\s+is)|we(?:'re|\s+are)"
    r"|you(?:'re|\s+are)|you\s+got\s+it|got\s+it|isn'?t|indeed|thanks?|thank\s+you"
    r"|please|correct|right|fine|ok|okay|sure|true)\b"
)


def _clean_correction(text: str) -> str:
    """Strip conversational filler from a corrected value.

    Applies the existing ``_FILLER_LEAD`` and ``_VALUE_LEAD`` rules for at
    most two passes, so filler never becomes part of the stored value
    ("no, only alpha beta" -> "alpha beta"; "sure, my name is Beta Jones"
    -> "Beta Jones").
    """
    cleaned = (text or "").strip()
    for _ in range(2):
        before = cleaned
        cleaned = _FILLER_LEAD.sub("", cleaned, count=1).strip()
        cleaned = _VALUE_LEAD.sub("", cleaned, count=1).strip()
        if cleaned == before:
            break
    cleaned = _HINDI_NAME_LEAD.sub("", cleaned, count=1).strip()
    cleaned = _HINDI_NAME_TRAIL.sub("", cleaned).strip()
    return cleaned.rstrip(" .!?")


def _affirm_with_value_replied(text: str) -> bool:
    """Whether an affirmative reply also carries a possible new value.

    "yes it is" affirms; "yes alpha beta" may be a correction and must be
    clarified instead of silently committing the old value or dropping the
    new one.
    """
    remainder = _AFFIRM_LEAD.sub("", text.strip(), count=1)
    remainder = re.sub(
        r"^(?:but|and|plus|also)[,\s]+", "", remainder, count=1, flags=re.IGNORECASE
    ).strip(" .,!?")
    if not remainder or _AFFIRM_CONTINUATION.match(remainder):
        return False
    # A remainder that is itself a plain (or repeated) affirmation, possibly
    # with trailing punctuation ("yes, yes"; "सही है।"; "हाँ हाँ") is not a
    # possible corrected value; such replies confirm.
    if _AFFIRM.fullmatch(remainder.strip(" .,!?।")):
        return False
    tokens = [token.strip(" .,!?।") for token in remainder.split()]
    tokens = [token for token in tokens if token]
    if tokens and all(_AFFIRM.fullmatch(token) for token in tokens):
        return False
    return _looks_like_value(remainder)


def _looks_like_value(text: str) -> bool:
    """Whether a reply plausibly is a corrected value (e.g. a name)."""
    candidate = text.strip()
    if len(candidate) < 3 or len(candidate.split()) > 6:
        return False
    # A plausible name carries script letters: Latin, or Devanagari letters
    # (vowels/consonants only — not digits or signs), so Indic-script name
    # corrections are recognised while arbitrary non-name strings (pure
    # digits, punctuation, other scripts) stay excluded.
    if not re.search(r"[A-Za-z\u0904-\u0939\u0958-\u0961]", candidate):
        return False
    if (
        _NON_ANSWER.match(candidate)
        or _UNKNOWN_ANSWER.search(candidate)
        or _CLARIFY_PHRASE.search(candidate)
    ):
        return False
    return True


def _looks_like_contact_number(text: str) -> bool:
    """Whether a reply plausibly is a corrected contact number.

    A number is digits and dialling punctuation only; the letters rule that
    protects name read-backs never applies to it.
    """
    candidate = (text or "").strip()
    if not re.fullmatch(r"[\d\s\-().+]{7,}", candidate):
        return False
    return len(re.sub(r"\D", "", candidate)) >= 7


def _classify_confirmation_turn(
    message: str, field: Optional[str] = None
) -> tuple[str, str]:
    """Classify a reply to a pending read-back.

    Returns ``(verdict, payload)`` where verdict is one of ``"yes"``,
    ``"no"``, ``"correction"`` or ``"ambiguous"`` and payload is what to
    feed to ``respond_to_confirmation`` (canonical yes/no or the corrected
    text). Ambiguous replies must not advance the workflow; the caller
    clarifies instead. ``field`` is the pending field, so a contact number
    offered as a correction is read as digits rather than as an unclear
    reply.
    """
    stripped = (message or "").strip()
    if not stripped:
        return "ambiguous", ""
    exact = _exact_reply_verdict(stripped)
    if exact in ("yes", "no"):
        return exact, exact
    if "?" in stripped:
        return "ambiguous", ""
    if (
        _UNKNOWN_ANSWER.search(stripped)
        or _NON_ANSWER.match(stripped)
        or _CLARIFY_PHRASE.search(stripped)
    ):
        return "ambiguous", ""

    def _is_value_correction(candidate: str) -> bool:
        if not candidate:
            return False
        if _looks_like_value(candidate):
            return True
        return field in (
            "contact_phone",
            "reporter_contact_phone",
        ) and _looks_like_contact_number(candidate)

    lead = _NEGATION_LEAD.match(stripped)
    if lead:
        candidate = _clean_correction(lead.group(1))
        # A negation-led reply whose remainder is itself a negation ("no,
        # nahi") is a clear rejection, not a corrected value.
        if candidate and _NEGATION.search(candidate) and not _AFFIRM.search(candidate):
            return "no", "no"
        if _is_value_correction(candidate):
            return "correction", candidate
        return "no", "no"
    affirm = bool(_AFFIRM.search(stripped))
    negate = bool(_NEGATION.search(stripped))
    if affirm and not negate:
        if _affirm_with_value_replied(stripped):
            # "yes alpha beta" may be a correction: clarify rather than
            # silently accept a value in either direction.
            return "ambiguous", ""
        return "yes", "yes"
    if negate and not affirm:
        return "no", "no"
    if affirm and negate:
        return "ambiguous", ""
    candidate = _clean_correction(stripped)
    if _is_value_correction(candidate):
        return "correction", candidate
    return "ambiguous", ""


def _clarification_for(field_name: str, awaiting_correction: bool) -> str:
    label = _CONFIRM_FIELD_LABELS.get(field_name, field_name.replace("_", " "))
    if awaiting_correction:
        return f"Sorry, I didn't catch that - please tell me the correct {label}."
    return (
        "Sorry, I didn't catch that - please say yes if that's correct, "
        "or no if you'd like me to change it."
    )


# One short, differently-worded clarification per field, offered instead of
# repeating the same question indefinitely when a reply is ambiguous.
_FIELD_CLARIFICATIONS: Dict[str, str] = {
    "product": "Sorry, I didn't catch the product name. Which medicine were you using?",
    "adverse_event": "Sorry, could you describe the side effect in a few words?",
    "reporter_role": "Just to check - are you reporting your own experience, or someone else's?",
    "patient_identifier": "Could you say your first and last name again?",
    # The approved flow collects a date of birth: the clarification never
    # offers an age as a substitute.
    "patient_age": "Could you say the date of birth again?",
    "consent_to_contact": "Would it be alright if we contact you later - yes or no?",
    "contact_method": "Which is easier for you - phone, email, or both?",
    "contact_phone": "Could you repeat the phone number, digit by digit?",
    "reporter_contact_phone": "Could you repeat the contact number, digit by digit?",
    "contact_email": "Could you repeat the email address?",
    "onset_date": "About when did it start - for example today, yesterday, or a few days ago?",
    "seriousness": "Do you consider this serious - yes, no, or not sure?",
    "outcome": "Are your symptoms better, about the same, or worse now?",
    "medical_care": "Did you get any medical care for it - yes or no?",
}


# Fields that are safe to park for human review once one clarification has
# gone unanswered. The value is the structured field recorded first; the
# review flag itself is raised separately via the engine.
_REVIEW_FALLBACK_FIELDS: Dict[str, str] = {
    "seriousness": "unknown",
    # An age that cannot be clarified after the targeted re-ask is parked as
    # "unknown" for human review instead of looping the generic question.
    "patient_age": "unknown",
    # Same bound for a phone number that stays unparseable (spoken digits in
    # any script): parked as "unknown" for the human reviewer; consent and
    # every other captured field are untouched.
    "contact_phone": "unknown",
}

# Fields whose repeated non-answers park the case for human review with the
# field left genuinely empty (audit step 3): an event description or a
# suspected medicine is never fabricated, and its clarification loop must
# stay bounded exactly like _REVIEW_FALLBACK_FIELDS.
_REVIEW_ONLY_FIELDS = frozenset({"adverse_event", "product"})

# Hard bound on failed read-back attempts per confirmation field: every turn
# that starts with a pending confirmation and does not commit a value counts
# as one attempt. After this many, the existing HUMAN_REVIEW fallback is used
# instead of repeating the read-back forever. The report is never discarded
# and never marked complete.
_MAX_CONFIRMATION_ATTEMPTS = 3

# What the person is told when the read-back loop is parked. Consent may not
# have been captured yet, so follow-up contact is never promised; required
# information may still be missing, so nothing is claimed to be saved in full,
# complete, or submitted — only that some details need human review.
CONFIRMATION_REVIEW_NOTICE = (
    "Thank you for your patience. We could not confirm some details, "
    "so your case will be reviewed by a member of our team. Take care."
)

# One targeted clarification when a single answer carries a date of birth and
# an age that disagree by more than a year: both values are held until the
# reporter picks one (neither is ever silently chosen).
_DOB_AGE_CLARIFICATION = (
    "Your date of birth ({dob}) and your age ({age} years) don't match. "
    "Which should I record - your date of birth or your age?"
)


def _dob_age_clarification(dob: str, age: str) -> str:
    return _DOB_AGE_CLARIFICATION.format(dob=dob, age=age)


# Bounded prototype-scope message when the suspected cause is a medicine
# other than the configured demo product. No claim of any submission is made.
def _product_scope_message(product: str) -> str:
    return (
        "This MedVaani prototype is configured for Lantus SoloStar only. "
        f"For {product}, please contact the medicine's manufacturer "
        "or your local adverse-event reporting channel."
    )


# Appended after the scope message: routine AE intake ends for the session.
# The report is never marked complete and no submission is ever claimed.
_SCOPE_TERMINATION_SENTENCE = "This prototype cannot continue this report."


def _short_clarification(field_name: str) -> str:
    return _FIELD_CLARIFICATIONS.get(
        field_name,
        f"Sorry, could you tell me a bit more about the {field_name.replace('_', ' ')}?",
    )


def _merge_event_description(existing: str, new: str) -> str:
    """Append a newly reported symptom without replacing the original."""
    current = (existing or "").strip()
    addition = (new or "").strip()
    if not addition:
        return current
    if not current:
        return addition
    if addition.lower() in current.lower():
        return current
    if current.lower() in addition.lower():
        return addition
    return f"{current}; {addition}"


_FILLER_LEAD = re.compile(
    r"(?i)^(?:sure|yes|yeah|yep|ok|okay|of\s+course|well|um|uh|so|only)[,\s]+"
)


def _clean_confirm_value(field_name: str, value: str) -> str:
    """Strip conversational filler before reading a value back."""
    cleaned = _FILLER_LEAD.sub("", (value or "").strip(), count=1)
    if field_name in ("patient_identifier", "reporter_name"):
        cleaned = _VALUE_LEAD.sub("", cleaned, count=1).strip()
        return cleaned.rstrip(" .!?")
    return cleaned.strip()


class ConversationalAEWorkflow:
    """Conversational workflow orchestrator."""

    _ACKS = (
        "Thanks for that.",
        "Got it, thank you.",
        "Okay, thank you.",
        "Thanks for the details.",
    )

    def __init__(
        self,
        engine: AEStateEngine,
        llm_provider: LLMProvider,
        confirm_fields: Optional[Any] = None,
        ask_additional_information: bool = False,
    ) -> None:
        self.engine = engine
        self.llm_provider = llm_provider
        # Opt-in per-field read-back confirmation; empty by default so every
        # existing caller (legacy journey harness included) is unchanged.
        self._confirm_fields = frozenset(confirm_fields or ())
        # Opt-in closing question; off by default to preserve legacy completion.
        self._ask_additional_information = bool(ask_additional_information)
        self._additional_info_asked = False
        self._additional_info_done = False
        self._warning_shown = False
        self._ack_index = 0
        # How many times a field question has gone unanswered/ambiguous, so a
        # short clarification can be offered instead of repeating forever.
        self._field_attempts: Dict[str, int] = {}
        self._last_asked_field: Optional[str] = None
        # Failed read-back attempts per pending confirmation field, bounded
        # so a stuck confirmation loop parks the report for human review
        # instead of looping forever. Counters reset when a value commits.
        self._confirmation_attempts: Dict[str, int] = {}
        # A conflicting DOB/age answer holds both raw values until the
        # reporter clarifies which one to record.
        self._dob_age_conflict: Optional[Dict[str, str]] = None
        # Set once a suspected out-of-scope medicine is captured: routine AE
        # intake for this session is over (urgent safety handling still runs
        # first on every turn). Lives on the per-session workflow instance.
        self._out_of_scope_product: Optional[str] = None
        # Step 1: the fixed introduction is handled once per session; it is
        # only spoken when the person has not already described the event.
        self._introduction_shown = False
        # Step 3: the empathy line is offered once, with the medication question.
        self._medication_empathy_shown = False
        # A combined name/date-of-birth read-back was rejected: only the
        # incorrect field is asked for again.
        self._awaiting_details_correction = False
        # Which detail ("dob" or "name") the person chose to correct; the
        # other field stays untouched.
        self._correction_target: Optional[str] = None
        # A correction-like second event description has been clarified once.
        self._event_correction_clarified = False
        # Step 4: pending "is it Lantus SoloStar?" verification of a captured
        # demo product. Only a demo-product capture arms it; an out-of-scope
        # medicine keeps the existing immediate scope/termination behaviour.
        self._scope_verification_pending = False
        self._product_verified = False
        # Step 4 clarification: the recorded medication was denied without an
        # alternative being named, so the medication question is asked again
        # (the refuted value may be replaced by the next answer).
        self._reask_medication = False
        # Step 8: consent was given, so the optional email question is offered
        # (and, on a yes, requested) exactly once.
        self._email_offer_pending = False
        self._email_request_pending = False
        # Step 10B.2: a plain "yes" to the closing question asks for the
        # details before anything is recorded as additional information.
        self._awaiting_additional_details = False
        # First reliable conversation timestamp/timezone seen, used as a
        # fallback reference for onset normalization across turns.
        self._conversation_timestamp: Optional[Any] = None
        self._conversation_timezone: Optional[Any] = None

    def _closing_message(self) -> str:
        """Step 12 wording, honest about the actual consent answer."""
        message = CLOSING_MESSAGE
        if self.engine.case.consent_to_contact is False:
            # Consent was declined: never imply that follow-up contact was
            # consented to.
            return message
        return f"{message} {CLOSING_CONSENT_SENTENCE}"

    def _closing_message(self) -> str:
        """Step 12 wording, honest about the actual consent answer."""
        message = CLOSING_MESSAGE
        if self.engine.case.consent_to_contact is False:
            # Consent was declined: never imply that follow-up contact was
            # consented to.
            return message
        return f"{message} {CLOSING_CONSENT_SENTENCE}"

    def _discard_pending_confirmation(self) -> None:
        """Drop a pending read-back without committing it.

        Used only where the value is optional (the email address): the
        candidate is discarded, never silently retained as verified.
        """
        field = self.engine.case.pending_confirmation
        if field is None:
            return
        self.engine.case.pending_confirmation = None
        self.engine.case.confirmation_candidate.pop(field, None)
        self.engine.case.confirmed_fields.discard(field)
        self.engine.case.confirmation_response = None

    def _abandon_optional_email(self) -> None:
        """Leave the optional email unprovided and continue the interview.

        The email is optional: after the retry limit the candidate is
        discarded, the email is recorded as declined/unavailable, and the
        next approved question is asked. Follow-up consent is never touched.
        """
        self._discard_pending_confirmation()
        try:
            self.engine.update("contact_email_declined", True)
        except AECaseError:
            pass
        self._confirmation_attempts.pop("contact_email", None)
        self._email_offer_pending = False
        self._email_request_pending = False

    def _display_question(self, question: str, text_mode: bool) -> str:
        """Question wording for the current channel.

        Text conversations carry no audio, so the spoken-only "heard"
        confirmation wording becomes "understood"; the voice flow keeps its
        existing wording unchanged.
        """
        if text_mode:
            return question.replace("I heard ", "I understood ")
        return question

    def _introduction_parts(self, description_captured: bool) -> List[str]:
        """Step 1: the fixed introduction, handled exactly once per session.

        It is only spoken when the person has not already described the event:
        a client that displayed the opening before the first utterance is
        never given a second copy, and a first message that already IS the
        description never has the opening repeated after it.
        """
        if self._introduction_shown:
            return []
        self._introduction_shown = True
        if description_captured:
            return []
        return [INTRODUCTION_MESSAGE]

    def _additional_info_due(self) -> bool:
        """Step 10B.2 is reached once the severity screen and the incident
        date have both been answered (never on the urgent branch)."""
        case = self.engine.case
        if case.seriousness not in ("yes", "no"):
            return False
        if not case.onset_date:
            return False
        return self.engine.current_state not in (
            WorkflowState.HUMAN_REVIEW,
            WorkflowState.CASE_COMPLETED,
        )

    def _active_step(self) -> Optional[tuple[str, str]]:
        """The step the next turn answers, when it is not a plain field ask.

        The documented flow inserts conversational steps that are not plain
        case fields: the medication verification (Step 4), the details
        correction after a rejected combined read-back, the optional email
        question (Step 8) and the closing additional-information question
        (Step 10B.2). While one is pending, it - not the engine's next
        missing field - is the question on the table.
        """
        if self._correction_target is not None:
            # Only the chosen detail is asked for again.
            key = (
                "patient_age"
                if self._correction_target == "dob"
                else "patient_identifier"
            )
            return key, _question_for(key, self.engine.case)
        if self._awaiting_details_correction:
            return "patient_identifier", DETAILS_CORRECTION_QUESTION
        if self._reask_medication:
            # Step 4 clarification: the recorded medication was denied, so
            # ask for the medication name again with the existing Step 3
            # wording. Nothing out-of-scope is claimed and nothing invented.
            return "product", _question_for("product", self.engine.case)
        if self._scope_verification_pending:
            return "product_verification", SCOPE_VERIFICATION_QUESTION
        if self._email_offer_pending:
            return "contact_email", EMAIL_OFFER_QUESTION
        if self._email_request_pending:
            return "contact_email", EMAIL_REQUEST_QUESTION
        if (
            self._ask_additional_information
            and self._additional_info_asked
            and not self._additional_info_done
        ):
            if self._awaiting_additional_details:
                # Step 10B.2: the person said yes; now invite the details.
                return "additional_information", ADDITIONAL_INFORMATION_FOLLOWUP
            return "additional_information", ADDITIONAL_INFORMATION_QUESTION
        return None

    def process_turn(self, message: str, context: Optional[Dict[str, Any]] = None) -> WorkflowTurnResult:
        """Process a single conversational turn."""
        # Build context for LLM (don't pass full engine internals)
        # The very first free-text message is a description, not an answer to
        # the (not yet asked) first question: only expose a pending question to
        # the provider once the interview has demonstrably started (a product
        # or event captured, or the initial message already preserved
        # verbatim). This keeps a vague first message from being swallowed as
        # the product/event answer - and never fabricates either from it.
        interview_started = bool(
            self.engine.case.product
            or self.engine.case.adverse_event
            or self.engine.case.unclassified_initial_input
        )
        active_step = self._active_step()
        # A step armed by this turn's own answer is answered on the next
        # turn: the consent answer must never be read as the email answer.
        scope_verification_at_start = self._scope_verification_pending
        email_offer_at_start = self._email_offer_pending
        email_request_at_start = self._email_request_pending
        reask_medication_at_start = self._reask_medication
        pending_question = (
            active_step
            if active_step is not None
            else (self.engine.next_field() if interview_started else None)
        )
        # Seriousness is only ever an answer to the severity question: a
        # symptom description containing "serious"/"severe" must never be
        # recorded as (or escalate to) an unasked severity answer.
        severity_question_pending = (
            pending_question is not None and pending_question[0] == "seriousness"
        )
        # Follow-up consent is only ever an answer to the consent question:
        # an email reply (or any other turn) can never change it.
        consent_question_pending = (
            pending_question is not None
            and pending_question[0] == "consent_to_contact"
        )
        turn_context = {
            "case_state": self.engine.current_state.value,
            "case_id": self.engine.case.case_id,
            "priority": self.engine.case.priority,
            "collected": self._collected_snapshot(),
            "next_field": pending_question,
        }
        if context:
            turn_context.update(context)
        # Text turns carry no audio: confirmation wording says "understood"
        # instead of the spoken-only "heard" (voice wording is unchanged).
        text_mode = bool(turn_context.get("text_mode"))

        # Remember the conversation's own timestamp/timezone (from the STT
        # request) so a relative onset answer can be resolved later. First
        # reliable value wins; nothing is assumed from the server clock.
        if turn_context.get("conversation_timestamp") and self._conversation_timestamp is None:
            self._conversation_timestamp = turn_context["conversation_timestamp"]
        if turn_context.get("conversation_timezone") and self._conversation_timezone is None:
            self._conversation_timezone = turn_context["conversation_timezone"]

        # True while this turn is the answer to the closing question. Used to
        # stop the closing free text from overwriting already-captured fields.
        collecting_additional = (
            self._ask_additional_information
            and self._additional_info_asked
            and not self._additional_info_done
        )

        # 1. LLM interpretation
        llm_result = self.llm_provider.interpret(message, turn_context)

        # 2. SAFETY CHECK
        safety_result = SafetyLayer.classify_input(message, self.engine.case)

        # 3-GATE. A product-info intent from the LLM is advisory only: below,
        # it diverts a turn away from the AE interview. Honour it only when
        # the deterministic safety layer also classifies the turn as a
        # product-information request, or when the case has not captured any
        # intake data yet (first-turn product-info demos). An active report
        # is never diverted by an unverified LLM intent; any fields the turn
        # still extracted are applied normally below.
        if (
            llm_result.intent == "product_info"
            and safety_result.signal is not SafetySignal.PRODUCT_INFORMATION_REQUEST
            and self._case_has_captured_fields()
        ):
            llm_result = replace(llm_result, intent="unknown")

        # 3. Handle safety - if urgent, apply immediately
        if safety_result.signal == SafetySignal.URGENT_LIFE_THREATENING:
            # Preserve what the person reported before escalating: the report
            # (including this description) is kept and flagged for human review.
            self._capture_urgent_description(message)
            SafetyLayer.apply_safety_to_case(safety_result, self.engine)
            return WorkflowTurnResult(
                case_id=self.engine.case.case_id,
                current_state=self.engine.current_state.value,
                intent="urgent_medical",
                extracted_information={},
                safety_status="urgent_life_threatening",
                priority=self.engine.case.priority,
                human_review_required=True,
                collected_information=self._collected_snapshot(),
                missing_information=[],
                response_text=safety_result.message,
                completed=False,
                next_question=None,
            )

        if safety_result.signal == SafetySignal.PERSONALIZED_TREATMENT_REQUEST:
            # Do not change engine state; return safe bounded response
            return WorkflowTurnResult(
                case_id=self.engine.case.case_id,
                current_state=self.engine.current_state.value,
                intent="treatment_query",
                extracted_information={},
                safety_status="treatment_request_blocked",
                priority=self.engine.case.priority,
                human_review_required=False,
                collected_information=self._collected_snapshot(),
                missing_information=self.engine.missing_information(),
                response_text=safety_result.message,
                completed=False,
                next_question=self.engine.next_field() if self.engine.next_field() else None,
            )

        if safety_result.signal == SafetySignal.PRODUCT_INFORMATION_REQUEST or llm_result.intent == "product_info":
            return WorkflowTurnResult(
                case_id=self.engine.case.case_id,
                current_state=self.engine.current_state.value,
                intent="product_info",
                extracted_information={},
                safety_status="product_info_routed_to_mi_rag",
                priority=self.engine.case.priority,
                human_review_required=False,
                collected_information=self._collected_snapshot(),
                missing_information=self.engine.missing_information(),
                response_text="This is a product information request. It will be routed to the MI/RAG layer (not implemented yet).",
                completed=False,
                next_question=self.engine.next_field() if self.engine.next_field() else None,
            )

        # 4-SCOPE. Once a suspected out-of-scope medicine has been captured,
        # routine AE intake for this session is over: restate the bounded
        # scope message and change nothing else. The urgent-safety check
        # above always runs first, so urgent replies still escalate.
        if self._out_of_scope_product is not None:
            return WorkflowTurnResult(
                case_id=self.engine.case.case_id,
                current_state=self.engine.current_state.value,
                intent=llm_result.intent or safety_result.signal.value.lower(),
                extracted_information={},
                safety_status="ok",
                priority=self.engine.case.priority,
                human_review_required=False,
                collected_information=self._collected_snapshot(),
                missing_information=self.engine.missing_information(),
                response_text=(
                    _product_scope_message(self._out_of_scope_product)
                    + " "
                    + _SCOPE_TERMINATION_SENTENCE
                ),
                completed=False,
                next_question=None,
            )

        # 4. NORMAL/AE case - apply extracted fields to engine
        extracted = _field_map_from_llm(llm_result)
        applied_any = False
        clarification: Optional[str] = None
        confirmation_review = False
        # Set when the Step 4 clarification received a new medication name.
        product_replaced = False
        # Set when a suspect product other than the configured demo product
        # (Lantus SoloStar) is captured; the reply then carries the bounded
        # prototype-scope message (never a submission claim).
        product_scope_notice: Optional[str] = None
        # Set when this turn answers Step 9 with a severe / life-threatening
        # event: safety handling then takes over the rest of the conversation.
        severity_urgent = False

        # 4-VERIFY (Step 4). The captured medication is verified before the
        # interview continues. "Yes" confirms it, "no" ends routine intake for
        # the mentioned medicine using the existing out-of-scope behaviour, a
        # restated medicine is re-checked (and canonicalised), and anything
        # else is clarified - uncertainty is never read as confirmation.
        scope_verification_turn = False
        if scope_verification_at_start:
            scope_verification_turn = True
            verdict, payload = _classify_confirmation_turn(message, "product")
            corrected: Optional[str] = None
            if verdict == "correction":
                named = (payload or "").strip()
                canonical = _normalize_product(named) or _known_other_medicine(named)
                negation_led = bool(_NEGATION_LEAD.match((message or "").strip()))
                if canonical is None and not negation_led:
                    # A value that names no medicine answers nothing here: it
                    # is never stored as the product (the captured product is
                    # never overwritten) and never ends the report. Clarify the
                    # yes-or-no verification instead.
                    verdict = "ambiguous"
                else:
                    corrected = canonical or named
            if verdict == "yes":
                self._scope_verification_pending = False
                self._product_verified = True
                self._confirmation_attempts.pop("product_verification", None)
            elif verdict == "no":
                recorded = self.engine.case.product or ""
                if _normalize_product(recorded) is not None:
                    # A bare "no": the recorded demo product itself was
                    # denied and no alternative medicine was named. Nothing
                    # out-of-scope can be described and nothing is invented,
                    # so the medication question is asked again; the loop is
                    # bounded by the same attempt limit as the other
                    # confirmation loops and falls back to human review.
                    self._scope_verification_pending = False
                    self._reask_medication = True
                    attempts = (
                        self._confirmation_attempts.get("product_verification", 0)
                        + 1
                    )
                    self._confirmation_attempts["product_verification"] = attempts
                    if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                        self.engine.flag_for_review(
                            reason=(
                                "The medication could not be verified after "
                                f"{attempts} attempts. All collected information "
                                "is preserved and the report is paused for "
                                "qualified human review."
                            )
                        )
                        confirmation_review = True
                        clarification = None
                        self._reask_medication = False
                else:
                    # The recorded medicine is not the demo product: the
                    # existing out-of-scope termination applies unchanged.
                    self._scope_verification_pending = False
                    self._out_of_scope_product = (
                        recorded or "the mentioned medication"
                    )
                    product_scope_notice = self._out_of_scope_product
            elif verdict == "correction" and corrected is not None:
                try:
                    self.engine.update("product", corrected)
                except AECaseError:
                    corrected = None
                if corrected is None:
                    clarification = _clarification_for("product", False)
                elif corrected == "Lantus SoloStar":
                    self._scope_verification_pending = False
                    self._product_verified = True
                    self._confirmation_attempts.pop(
                        "product_verification", None
                    )
                else:
                    self._scope_verification_pending = False
                    self._out_of_scope_product = corrected
                    product_scope_notice = corrected
                if clarification is None:
                    applied_any = True
            else:
                attempts = (
                    self._confirmation_attempts.get("product_verification", 0) + 1
                )
                self._confirmation_attempts["product_verification"] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                    self.engine.flag_for_review(
                        reason=(
                            "The medication could not be verified after "
                            f"{attempts} attempts. All collected information "
                            "is preserved and the report is paused for "
                            "qualified human review."
                        )
                    )
                    confirmation_review = True
                    clarification = None
                    self._scope_verification_pending = False
                else:
                    clarification = _clarification_for("product", False)

        # 4-DOB. A single answer with both a date of birth and an age that
        # disagree by more than a year: hold both values, apply neither, and
        # ask exactly one targeted clarification (never silently choose).
        conflict_prompt: Optional[str] = None
        raw_conflict = (llm_result.extracted_fields or {}).get("dob_age_conflict")
        if isinstance(raw_conflict, dict) and not scope_verification_turn:
            conflict_dob = str(raw_conflict.get("dob") or "").strip()
            conflict_age = str(raw_conflict.get("age") or "").strip()
            if conflict_dob and conflict_age and not self.engine.case.patient_age:
                self._dob_age_conflict = {"dob": conflict_dob, "age": conflict_age}
                conflict_prompt = _dob_age_clarification(conflict_dob, conflict_age)

        # 4a. Explicit confirmation sub-state. While a read-back is pending
        # for an integrated field, the turn is a confirmation reply (yes / no
        # / correction), not a new field answer; anything ambiguous is
        # clarified without advancing. While the combined name read-back is
        # still waiting for the date of birth the turn answers that question
        # instead, so no read-back is offered before both values are known.
        #
        # A rejected combined read-back only asks for the field that was
        # wrong (name or date of birth); a date repeated while the combined
        # read-back is pending corrects the date and never the name.
        confirmation_handled = False
        confirmation_rejected = False
        pending_field = self.engine.case.pending_confirmation
        awaiting_dob = (
            pending_field == "patient_identifier"
            and not self.engine.case.patient_age
        )
        details_correction_turn = self._awaiting_details_correction
        handled_by_details_flow = False
        confirmation_active = (
            pending_field is not None
            and pending_field in self._confirm_fields
            and not awaiting_dob
            and not scope_verification_turn
            and not details_correction_turn
        )
        repeated_dob = None
        if confirmation_active and pending_field == "patient_identifier":
            repeated_dob = normalized_dob_in(message)
        if self._correction_target is not None:
            # Only the chosen detail is being corrected: a full date updates
            # the date of birth (the name stays), a name-like value updates
            # the name (the date stays); anything incomplete is re-asked for
            # that one field, bounded like the other confirmation loops.
            confirmation_handled = True
            handled_by_details_flow = True
            dob_value = normalized_dob_in(message)
            verdict, payload = _classify_confirmation_turn(message, pending_field)
            if dob_value is not None:
                try:
                    self.engine.update("patient_age", dob_value)
                except AECaseError:
                    pass
                self._correction_target = None
                applied_any = True
                self._confirmation_attempts.pop("patient_identifier", None)
            elif (
                self._correction_target == "name"
                and verdict == "correction"
                and _looks_like_value(payload)
                and len(payload.split()) >= 2
            ):
                try:
                    self.engine.respond_to_confirmation(payload)
                    self._correction_target = None
                    applied_any = True
                    self._confirmation_attempts.pop("patient_identifier", None)
                except AECaseError:
                    clarification = DETAILS_CORRECTION_CLARIFICATION
            elif self._correction_target == "dob":
                attempts = (
                    self._confirmation_attempts.get("patient_identifier", 0) + 1
                )
                self._confirmation_attempts["patient_identifier"] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                    self.engine.flag_for_review(
                        reason=(
                            "The date of birth could not be corrected after "
                            f"{attempts} attempts. All collected information "
                            "is preserved and the report is paused for "
                            "qualified human review."
                        )
                    )
                    confirmation_review = True
                    clarification = None
                    self._correction_target = None
                else:
                    clarification = INCOMPLETE_DOB_CLARIFICATION
            else:
                # Name target with no usable value: return to the choice
                # question, which keeps its own attempt bound.
                self._correction_target = None
                clarification = DETAILS_CORRECTION_CLARIFICATION
        elif details_correction_turn:
            confirmation_handled = True  # this turn consumes the reply only
            handled_by_details_flow = True
            dob_value = normalized_dob_in(message)
            verdict, payload = _classify_confirmation_turn(message, pending_field)
            if dob_value is not None:
                # Only the date of birth is corrected; the name is retained.
                try:
                    self.engine.update("patient_age", dob_value)
                except AECaseError:
                    pass
                self._awaiting_details_correction = False
                applied_any = True
                self._confirmation_attempts.pop("patient_identifier", None)
            elif _DOB_CHOICE.search(message):
                # "date of birth" / "the date" names the field to correct;
                # the literal phrase is never stored as a value.
                self._correction_target = "dob"
                self._awaiting_details_correction = False
                applied_any = True
            elif _NAME_CHOICE.search(message):
                # "the name" names the field to correct; the date stays.
                self._correction_target = "name"
                self._awaiting_details_correction = False
                applied_any = True
            elif (
                verdict == "correction"
                and _looks_like_value(payload)
                and len(payload.split()) >= 2
            ):
                # Only the name is corrected (first and last name per the
                # question); the date of birth is retained. A single-word
                # mumble ("hmm") is ambiguity, never a corrected name.
                try:
                    self.engine.respond_to_confirmation(payload)
                    self._awaiting_details_correction = False
                    applied_any = True
                    self._confirmation_attempts.pop("patient_identifier", None)
                except AECaseError:
                    clarification = DETAILS_CORRECTION_CLARIFICATION
            elif verdict == "no":
                # A repeated clear rejection is a successful interpretation:
                # the choice is simply re-asked, without counting a failed
                # attempt (mirroring the read-back rejection semantics).
                clarification = DETAILS_CORRECTION_CLARIFICATION
            else:
                attempts = (
                    self._confirmation_attempts.get("patient_identifier", 0) + 1
                )
                self._confirmation_attempts["patient_identifier"] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                    self.engine.flag_for_review(
                        reason=(
                            "The patient details could not be corrected after "
                            f"{attempts} attempts. All collected information "
                            "is preserved and the report is paused for "
                            "qualified human review."
                        )
                    )
                    confirmation_review = True
                    clarification = None
                    self._awaiting_details_correction = False
                else:
                    clarification = DETAILS_CORRECTION_CLARIFICATION
        elif confirmation_active and repeated_dob is not None:
            # A date of birth repeated (or corrected) while the combined
            # read-back is pending answers the date question, never the name.
            try:
                self.engine.update("patient_age", repeated_dob)
            except AECaseError:
                pass
            confirmation_handled = True
            applied_any = True
            handled_by_details_flow = True
            self._confirmation_attempts.pop("patient_identifier", None)
        elif confirmation_active:
            awaiting_correction = self.engine.case.confirmation_response == "no"
            verdict, payload = _classify_confirmation_turn(message, pending_field)
            if (
                verdict == "correction"
                and pending_field == "patient_identifier"
                and isinstance(payload, str)
                and payload
                and len(payload.split()) == 1
                and re.fullmatch(r"[A-Za-z0-9 .'\-]+", payload)
            ):
                # A single Latin token ("hmm") is a mumble, never a first and
                # last name: clarify instead of replacing the name candidate.
                verdict = "ambiguous"
            if pending_field == "contact_email":
                # Any reply carrying a usable address (written or spelled)
                # is a correction/capture of the candidate - never a yes/no
                # answer - and the normalized form is what gets confirmed.
                normalized_email = _normalize_spoken_email(message)
                if normalized_email:
                    verdict, payload = "correction", normalized_email
                elif verdict == "correction":
                    # "the email address is incorrect" rejects the candidate
                    # (the corrected address is then asked for); anything else
                    # is ambiguous and is never stored as an address.
                    if _EMAIL_INCORRECTION.search(message):
                        verdict, payload = "no", "no"
                    else:
                        verdict = "ambiguous"
            if verdict == "ambiguous" or (
                verdict == "yes" and awaiting_correction
            ):
                clarification = _clarification_for(pending_field, awaiting_correction)
            elif (
                verdict == "no"
                and pending_field == "patient_identifier"
                and self.engine.case.patient_age
            ):
                # Rejecting the combined read-back does not restart the whole
                # question: only the incorrect field is asked for again.
                self._awaiting_details_correction = True
                confirmation_handled = True
                confirmation_rejected = True  # clear interpretation, not a miss
                applied_any = True
            else:
                try:
                    self.engine.respond_to_confirmation(payload)
                    confirmation_handled = True
                    confirmation_rejected = verdict == "no"
                    applied_any = True
                except AECaseError:
                    clarification = _clarification_for(
                        pending_field, awaiting_correction
                    )

        # 4a-2. Bound the read-back loop. A turn that starts with a pending
        # confirmation and yields an ambiguous or unrecognized reply counts as
        # one failed attempt for that field. A clear rejection is a successful
        # interpretation (the candidate is correctly discarded and the field
        # re-asked), so it neither increments nor resets the counter; a
        # committed value clears it. After _MAX_CONFIRMATION_ATTEMPTS the
        # report is preserved exactly as collected and paused for qualified
        # human review (existing fallback) instead of repeating forever.
        # Nothing is discarded, no contact detail is invented, and completion
        # stays off.
        if confirmation_active and not handled_by_details_flow:
            if confirmation_handled and confirmation_rejected:
                pass
            elif confirmation_handled and self.engine.case.pending_confirmation is None:
                self._confirmation_attempts.pop(pending_field, None)
            else:
                attempts = self._confirmation_attempts.get(pending_field, 0) + 1
                self._confirmation_attempts[pending_field] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                    if pending_field == "contact_email":
                        # The email is optional: after the retry limit it is
                        # left unprovided and the interview continues safely
                        # (the candidate is discarded, never kept as
                        # verified, and consent is untouched).
                        self._abandon_optional_email()
                    else:
                        label = _CONFIRM_FIELD_LABELS.get(
                            pending_field, pending_field.replace("_", " ")
                        )
                        self.engine.flag_for_review(
                            reason=(
                                f"The {label} could not be confirmed after "
                                f"{attempts} attempts. All collected information "
                                "is preserved and the report is paused for "
                                "qualified human review."
                            )
                        )
                        confirmation_review = True
                        clarification = None

        # 4b. Apply each field if present (engine validates). The original
        # product and side-effect description are never overwritten once
        # captured. A confirmation reply consumes the turn, so no other field
        # is applied on that turn; a Step 4 medication-verification reply is
        # handled entirely by its own branch above.
        if not confirmation_handled and not scope_verification_turn:
            for field_name, field_value in extracted.items():
                if field_name == "seriousness" and not severity_question_pending:
                    # Context-gated: seriousness is only recorded when the
                    # severity question is the one on the table. A symptom
                    # description (even one containing "serious" or "severe")
                    # is never an answer to it and never triggers the urgent
                    # branch.
                    continue
                if field_name == "consent_to_contact" and not consent_question_pending:
                    # Context-gated: follow-up consent is only recorded when
                    # the consent question is the one on the table. Saying
                    # "no" to an email or correcting an email can never
                    # withdraw or overwrite the consent already given.
                    continue
                if (
                    field_name == "product"
                    and self.engine.case.product
                    and not self._reask_medication
                ):
                    # The original product is never overwritten once captured.
                    continue
                if field_name == "adverse_event":
                    # Never replace the original description; merge in a newly
                    # reported symptom so every symptom is preserved. The
                    # closing answer (Step 10B.2) is kept separately from the
                    # original description, so it never extends it either.
                    if collecting_additional and self.engine.case.adverse_event:
                        continue
                    if (
                        self.engine.case.adverse_event
                        and not self._event_correction_clarified
                        and isinstance(field_value, str)
                        and field_value.strip()
                        and _EXPLICIT_CORRECTION.search(message)
                    ):
                        # A correction-like statement is clarified once instead
                        # of being silently combined with the original. The
                        # wording is never lost afterwards: it is merged
                        # verbatim and the original is never overwritten.
                        self._event_correction_clarified = True
                        clarification = EVENT_CORRECTION_CLARIFICATION
                        continue
                    if isinstance(field_value, str) and field_value.strip():
                        merged = _merge_event_description(
                            self.engine.case.adverse_event or "", field_value
                        )
                        if merged != (self.engine.case.adverse_event or ""):
                            try:
                                self.engine.update("adverse_event", merged)
                                applied_any = True
                            except AECaseError:
                                pass
                    continue
                if collecting_additional and getattr(
                    self.engine.case, field_name, None
                ):
                    # The closing answer may fill empty fields but must never
                    # overwrite or corrupt what was already captured.
                    continue
                if (
                    field_name == "product"
                    and isinstance(field_value, str)
                    and "lantus" in field_value.lower()
                ):
                    field_value = "Lantus SoloStar"
                if field_name == "concomitant_medications" and isinstance(
                    field_value, list
                ):
                    # Append (deduplicated) so each turn's concomitant
                    # medicine is preserved across turns.
                    existing = list(
                        getattr(self.engine.case, "concomitant_medications", None)
                        or []
                    )
                    additions = [item for item in field_value if item not in existing]
                    if not additions:
                        continue
                    field_value = existing + additions
                try:
                    if field_name in self._confirm_fields:
                        if self.engine.case.pending_confirmation is not None:
                            # Under confirmation only the reply may move it.
                            continue
                        candidate = _clean_confirm_value(field_name, str(field_value))
                        if not candidate:
                            continue
                        self.engine.begin_confirmation(field_name, candidate)
                    else:
                        self.engine.update(field_name, field_value)
                    applied_any = True
                    if field_name == "product":
                        # A Step 4 clarification may replace the medication
                        # that was denied; every later product answer keeps
                        # the existing no-overwrite rule.
                        if self._reask_medication:
                            product_replaced = True
                        self._reask_medication = False
                    if (
                        field_name == "product"
                        and isinstance(field_value, str)
                        and field_value != "Lantus SoloStar"
                        and not field_value.startswith("Unknown")
                    ):
                        # Another medicine is the suspected cause: say the
                        # prototype is Lantus-only; never attribute the event
                        # to Lantus and never claim a submission. The session
                        # becomes out of scope and routine intake ends.
                        product_scope_notice = field_value
                        self._out_of_scope_product = field_value
                    elif (
                        field_name == "product"
                        and field_value == "Lantus SoloStar"
                        and not self._product_verified
                        and not self._scope_verification_pending
                    ):
                        # Step 4: the demo product is verified with the person
                        # before the interview continues.
                        self._scope_verification_pending = True
                    if field_name == "consent_to_contact" and field_value is True:
                        # Step 8: consent opens the optional email question,
                        # which is offered exactly once and may be declined.
                        if (
                            not self.engine.case.contact_email
                            and not self.engine.case.contact_email_declined
                        ):
                            self._email_offer_pending = True
                    if field_name == "contact_email":
                        # The address is on the table: no further email ask.
                        self._email_offer_pending = False
                        self._email_request_pending = False
                    if field_name == "seriousness" and field_value == "yes":
                        # Step 10A: a severe or life-threatening event is
                        # treated as urgent regardless of how it was phrased.
                        severity_urgent = True
                except Exception:
                    # Skip invalid fields per engine validation rules
                    pass

        # 4-REASK (Step 4 clarification). A clarification turn that yields no
        # new medication counts as one failed verification attempt, bounded
        # exactly like the read-back loops: after the limit the report is
        # preserved and paused for qualified human review instead of asking
        # forever. A captured medication clears both counters.
        if product_replaced:
            self._confirmation_attempts.pop("product_reask", None)
            self._confirmation_attempts.pop("product_verification", None)
        elif self._reask_medication and reask_medication_at_start:
            attempts = self._confirmation_attempts.get("product_reask", 0) + 1
            self._confirmation_attempts["product_reask"] = attempts
            if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                self.engine.flag_for_review(
                    reason=(
                        "The medication could not be captured after "
                        f"{attempts} clarification attempts. All collected "
                        "information is preserved and the report is paused "
                        "for qualified human review."
                    )
                )
                confirmation_review = True
                clarification = None
                self._reask_medication = False

        # 4-EMAIL (Step 8). Consent was given, so the email address is offered
        # first: a no skips it entirely, a yes asks for the address, and an
        # address volunteered on either turn is captured and read back by the
        # normal confirmation path.
        if email_offer_at_start:
            if self.engine.case.pending_confirmation == "contact_email":
                self._email_offer_pending = False
            elif self.engine.case.contact_email:
                self._email_offer_pending = False
            elif _is_decline(message):
                self._email_offer_pending = False
                self.engine.update("contact_email_declined", True)
            else:
                verdict, _ = _classify_confirmation_turn(message)
                if verdict == "yes":
                    self._email_offer_pending = False
                    self._email_request_pending = True
                elif verdict == "no":
                    self._email_offer_pending = False
                    self.engine.update("contact_email_declined", True)
                else:
                    attempts = (
                        self._confirmation_attempts.get("contact_email", 0) + 1
                    )
                    self._confirmation_attempts["contact_email"] = attempts
                    if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                        # Optional email: leave it unprovided and continue.
                        try:
                            self.engine.update("contact_email_declined", True)
                        except AECaseError:
                            pass
                        self._email_offer_pending = False
                        self._confirmation_attempts.pop("contact_email", None)
                    else:
                        clarification = _clarification_for("contact_email", False)
        elif email_request_at_start:
            if self.engine.case.pending_confirmation == "contact_email":
                self._email_request_pending = False
            elif self.engine.case.contact_email:
                self._email_request_pending = False
            elif _is_decline(message) or _classify_confirmation_turn(message)[0] == "no":
                # The address is optional: declining it never blocks the report.
                self._email_request_pending = False
                self.engine.update("contact_email_declined", True)
            else:
                attempts = (
                    self._confirmation_attempts.get("contact_email", 0) + 1
                )
                self._confirmation_attempts["contact_email"] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
                    # Optional email: leave it unprovided and continue.
                    try:
                        self.engine.update("contact_email_declined", True)
                    except AECaseError:
                        pass
                    self._email_request_pending = False
                    self._confirmation_attempts.pop("contact_email", None)
                else:
                    clarification = _short_clarification("contact_email")

        # 4-URGENT (Step 10A). A severe or life-threatening answer takes
        # priority over every routine question: the person is asked to seek
        # immediate medical attention, the case is escalated for qualified
        # human review (existing urgent handling), routine intake stops, and
        # the incident date and additional-information steps are skipped.
        if severity_urgent:
            self.engine.mark_urgent(
                reason=(
                    "The reporter answered that the event was severe or "
                    "life-threatening."
                )
            )
            response_parts = self._introduction_parts(
                description_captured=bool(self.engine.case.adverse_event)
            )
            response_parts.append(SEVERITY_URGENT_MESSAGE)
            response_parts.append(self._closing_message())
            return WorkflowTurnResult(
                case_id=self.engine.case.case_id,
                current_state=self.engine.current_state.value,
                intent="urgent_medical",
                extracted_information=extracted,
                safety_status="urgent_life_threatening",
                priority=self.engine.case.priority,
                human_review_required=True,
                collected_information=self._collected_snapshot(),
                missing_information=[],
                response_text=" ".join(response_parts),
                completed=False,
                next_question=None,
            )

        # Preserve the person's initial free-text description verbatim when it
        # could not yet be classified as an adverse event (audit step 1):
        # recorded separately on the case so nothing is invented, no
        # structured field is filled, and later answers never overwrite it.
        raw_message = (message or "").strip()
        if (
            raw_message
            and self.engine.case.unclassified_initial_input is None
            and self.engine.case.adverse_event is None
            and not applied_any
            and not _NON_ANSWER.match(raw_message)
            and not _SIMPLE_YESNO.match(raw_message)
        ):
            # Direct set (same precedent as the onset-derived fields): this
            # is a verbatim record, not a validated workflow answer, so it
            # never enters the confirmation or field-update paths.
            self.engine.case.unclassified_initial_input = raw_message

        # Once the age/DOB question is answered, the held conflicting values
        # are no longer needed: the reporter has clarified by answering.
        if self._dob_age_conflict is not None and self.engine.case.patient_age:
            self._dob_age_conflict = None

        # 4b-2. Interpret a captured onset statement against the conversation
        # timestamp/timezone. Runs once per onset capture; if no reliable
        # reference exists the original wording is kept and the date is marked
        # unknown (never invented). Ambiguity is retained, not guessed at, so
        # the conversation is not extended with a confirmation turn.
        self._normalize_onset(turn_context)

        # 4c. Finalize once every required field is satisfied. The engine owns
        # the completion rules; this only invokes its guarded complete(). When
        # the optional closing question is enabled it is asked exactly once at
        # Step 10B.2 (after the severity screen and the incident date), a plain
        # "yes" first invites the details, and the case completes once every
        # required field is satisfied.
        ready_to_complete = (
            self.engine.can_complete()
            and self.engine.current_state is not WorkflowState.CASE_COMPLETED
        )
        additional_info_provided = False
        if self._ask_additional_information:
            if not self._additional_info_done and self._additional_info_due():
                if not self._additional_info_asked:
                    self._additional_info_asked = True
                elif _is_decline(message):
                    self.engine.decline_additional_information()
                    self._additional_info_done = True
                    self._awaiting_additional_details = False
                elif self._awaiting_additional_details or not _is_plain_yes(message):
                    try:
                        self.engine.record_additional_information(message)
                        additional_info_provided = True
                    except AECaseError:
                        pass
                    self._additional_info_done = True
                    self._awaiting_additional_details = False
                else:
                    # A bare "yes" is not the additional information itself:
                    # ask for the details before recording anything.
                    self._awaiting_additional_details = True
            if self._additional_info_done and self.engine.can_complete():
                self.engine.complete()
        elif ready_to_complete:
            self.engine.complete()

        # 5. Determine next info
        missing = self.engine.missing_information()
        next_field = self.engine.next_field()

        # 5-pre. A documented conversational step (the Step 4 medication
        # verification, the details correction after a rejected combined
        # read-back, the Step 8 email question, or the Step 10B.2 closing
        # question) replaces the next plain field question while it is
        # pending. A pending read-back keeps priority: it owns its question -
        # except while the person is choosing which detail to correct. The
        # closing question keeps the envelope behaviour it has always
        # had: it is asked in the response, while ``next_question`` stays
        # null because no case field is outstanding for it.
        active_step = None
        if (
            self._awaiting_details_correction
            or self._correction_target is not None
            or self.engine.case.pending_confirmation is None
        ):
            active_step = self._active_step()
        closing_step_active = bool(
            active_step is not None and active_step[0] == "additional_information"
        )
        if active_step is not None:
            next_field = active_step

        # 5a. If the same field is still pending and the reply advanced nothing,
        # offer one short, differently-worded clarification rather than looping
        # on the identical question indefinitely. Conversational steps above own
        # their own clarification and bound, so the field loop never runs for
        # them.
        if active_step is not None:
            pass
        elif next_field is not None:
            field_key = next_field[0]
            if (
                self._last_asked_field == field_key
                and not applied_any
                and not clarification
            ):
                self._field_attempts[field_key] = (
                    self._field_attempts.get(field_key, 0) + 1
                )
            else:
                self._field_attempts[field_key] = 0
            if self._field_attempts.get(field_key, 0) >= 1 and not clarification:
                if (
                    field_key in _REVIEW_FALLBACK_FIELDS
                    and self._field_attempts[field_key] >= 2
                ):
                    # One clarification has already gone unanswered. Park the
                    # field with its safe fallback value and stop the loop by
                    # sending the case to human review. Urgency/priority is
                    # untouched: a separate path owns that.
                    fallback_value = _REVIEW_FALLBACK_FIELDS[field_key]
                    if not getattr(self.engine.case, field_key, None):
                        try:
                            self.engine.update(field_key, fallback_value)
                        except AECaseError:
                            pass
                    self.engine.flag_for_review(
                        reason=(
                            f"The answer about {field_key.replace('_', ' ')} was "
                            "unclear after a clarification; recorded as unknown "
                            "and paused for qualified human review."
                        )
                    )
                    next_field = None
                elif (
                    field_key in _REVIEW_ONLY_FIELDS
                    and self._field_attempts[field_key] >= 2
                ):
                    # A still-unclear event description or suspected product
                    # is bounded the same way, but the field is left empty:
                    # no clinical description or medicine name is ever
                    # invented. The out-of-scope stop path is unaffected (it
                    # only triggers when a value was actually captured).
                    self.engine.flag_for_review(
                        reason=(
                            f"The {field_key.replace('_', ' ')} was still "
                            "unclear after a clarification; left empty and "
                            "paused for qualified human review."
                        )
                    )
                    next_field = None
                else:
                    next_field = (field_key, _short_clarification(field_key))
            self._last_asked_field = field_key
        else:
            self._last_asked_field = None

        # 5b. A held DOB/age conflict asks its own single clarification this
        # turn, replacing the generic age question. Questioning pauses (e.g.
        # under human review) keep their own behaviour. A pending
        # conversational step keeps priority over it.
        if conflict_prompt is not None and next_field is not None and active_step is None:
            next_field = ("patient_age", conflict_prompt)

        # 5c. An out-of-scope suspect product ends routine intake for this
        # session: no further questions are asked (urgent safety handling
        # above has already had its chance).
        if self._out_of_scope_product is not None:
            next_field = None

        # 6. Build response
        completed = self.engine.current_state is WorkflowState.CASE_COMPLETED
        # Step 1: the opening is handled once per session and is never spoken
        # after the person has already described the event.
        response_parts = self._introduction_parts(
            description_captured=bool(self.engine.case.adverse_event)
        )
        opening_emitted = bool(response_parts)
        if confirmation_review:
            # Read-back attempts are exhausted: say so plainly. The case is
            # under human review (never completed) with everything preserved.
            response_parts.append(CONFIRMATION_REVIEW_NOTICE)
        else:
            if clarification:
                # Ambiguous confirmation reply: clarify, then re-ask (the
                # engine still returns the read-back, so nothing advanced).
                response_parts.append(clarification)
            is_ae = llm_result.intent == "ae_report" or safety_result.signal == SafetySignal.NORMAL_AE_REPORTING
            # Step 3: the medication question carries the one-time empathy
            # line, which replaces the generic acknowledgement on that turn so
            # the conversation never repeats sympathy statements.
            empathy_next = (
                next_field is not None
                and next_field[0] == "product"
                and not self._medication_empathy_shown
            )
            if additional_info_provided:
                response_parts.append("Thank you for sharing that.")
            elif is_ae and not empathy_next:
                if not self._warning_shown:
                    response_parts.append(
                        "Thanks for telling me, and I'm sorry you're going through this. "
                        "I'll take a few details so we can record it safely."
                    )
            elif applied_any and not collecting_additional and not completed:
                response_parts.append(self._ACKS[self._ack_index % len(self._ACKS)])
                self._ack_index += 1

            if product_scope_notice:
                # Shown on the turn the other medicine is captured as the
                # suspect product; the session is then out of scope and
                # routine intake ends (urgent turns never reach here).
                response_parts.append(_product_scope_message(product_scope_notice))
                response_parts.append(_SCOPE_TERMINATION_SENTENCE)

            if next_field:
                _, question = next_field
                if empathy_next:
                    # Step 3: acknowledge the experience once, then ask.
                    response_parts.append(MEDICATION_EMPATHY)
                    self._medication_empathy_shown = True
                # The opening already ends with the event-description
                # invitation: never ask the same question twice.
                if not (opening_emitted and next_field[0] == "adverse_event"):
                    response_parts.append(
                        self._display_question(question, text_mode)
                    )
            elif completed:
                # Step 11/12: the report is presented by the existing report
                # mechanism on completion; the closing line follows it.
                response_parts.append(self._closing_message())

            # The physician-consultation note follows the question instead of
            # preceding it, so it never crowds the medication question.
            if is_ae and not self._warning_shown:
                response_parts.append(
                    "Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects."
                )
                response_parts.append(
                    "Do not self-diagnose or change your medication without medical advice."
                )
                self._warning_shown = True

        response_text = " ".join(response_parts) if response_parts else "Thank you."

        return WorkflowTurnResult(
            case_id=self.engine.case.case_id,
            current_state=self.engine.current_state.value,
            intent=llm_result.intent or safety_result.signal.value.lower(),
            extracted_information=extracted,
            safety_status="ok",
            priority=self.engine.case.priority,
            human_review_required=self.engine.current_state is WorkflowState.HUMAN_REVIEW,
            collected_information=self._collected_snapshot(),
            missing_information=missing,
            response_text=response_text,
            completed=(self.engine.current_state is WorkflowState.CASE_COMPLETED),
            next_question=(
                None
                if closing_step_active
                else (
                    [next_field[0], self._display_question(next_field[1], text_mode)]
                    if next_field
                    else None
                )
            ),
        )

    def _capture_urgent_description(self, message: str) -> None:
        """Preserve the person's own wording before the urgent escalation.

        The urgent statement is merged into the case description (never
        overwriting an existing report) so the report kept for human review
        includes what was said at the moment of escalation. No state change
        happens here; escalation itself stays owned by SafetyLayer.
        """
        merged = _merge_event_description(
            self.engine.case.adverse_event or "", (message or "").strip()
        )
        if merged and merged != (self.engine.case.adverse_event or ""):
            try:
                self.engine.update("adverse_event", merged)
            except Exception:
                # Safety escalation must never be blocked by field validation.
                pass

    def _normalize_onset(self, turn_context: Dict[str, Any]) -> None:
        case = self.engine.case
        if not case.onset_date or case.onset_date_source is not None:
            return
        reference = resolve_reference(
            turn_context.get("conversation_timestamp") or self._conversation_timestamp,
            turn_context.get("conversation_timezone") or self._conversation_timezone,
        )
        result = normalize_onset_statement(case.onset_date, reference)
        case.onset_date_normalized = result.normalized_iso
        case.onset_period = result.period
        case.onset_date_source = result.source

    def _case_has_captured_fields(self) -> bool:
        """Whether the case has captured any real intake data yet.

        ``seriousness`` defaults to ``"unknown"`` on a new case, so only a
        deliberate non-default value counts as progress.
        """
        case = self.engine.case
        return any(
            (
                case.product,
                case.adverse_event,
                case.patient_identifier,
                case.patient_age,
                case.patient_sex,
                case.reporter_role,
                case.reporter_name,
                case.reporter_contact_phone,
                case.contact_phone,
                case.contact_email,
                case.preferred_contact_method,
                case.consent_to_contact is not None,
                case.onset_date,
                case.medical_care,
                case.seriousness != "unknown",
                case.outcome,
                case.dose_and_use,
                case.prescribing_physician,
                case.concomitant_medications,
            )
        )

    def _collected_snapshot(self) -> Dict[str, Any]:
        case = self.engine.case
        return {
            "product": case.product,
            "adverse_event": case.adverse_event,
            # Verbatim first message kept when it could not be classified yet
            # (never promoted into adverse_event automatically).
            "unclassified_initial_input": case.unclassified_initial_input,
            "patient_identifier": case.patient_identifier,
            "patient_age": case.patient_age,
            "reporter_role": case.reporter_role,
            "reporter_name": case.reporter_name,
            "reporter_contact_phone": case.reporter_contact_phone,
            "contact_phone": case.contact_phone,
            "contact_email": case.contact_email,
            "contact_email_declined": case.contact_email_declined,
            "preferred_contact_method": case.preferred_contact_method,
            "consent_to_contact": case.consent_to_contact,
            "onset_date": case.onset_date,
            "onset_date_normalized": case.onset_date_normalized,
            "onset_date_display": format_iso_date(case.onset_date_normalized),
            "onset_period": case.onset_period,
            "onset_date_source": case.onset_date_source,
            "medical_care": case.medical_care,
            "seriousness": case.seriousness,
            "outcome": case.outcome,
            "dose_and_use": case.dose_and_use,
            "prescribing_physician": case.prescribing_physician,
            "concomitant_medications": list(case.concomitant_medications),
            "additional_information": case.additional_information,
            "additional_information_declined": case.additional_information_declined,
            "additional_information_status": self.engine.additional_information_status,
            "priority": case.priority,
        }
