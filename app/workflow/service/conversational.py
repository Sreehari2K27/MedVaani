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
    _NEGATION,
    _NON_ANSWER,
    _SIMPLE_YESNO,
    _UNKNOWN_ANSWER,
    _seriousness_value,
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

_CONFIRM_FIELD_LABELS = {
    "patient_identifier": "name",
    "contact_phone": "phone number",
    "contact_email": "email address",
}

# Optional closing question. Opt-in per workflow so every existing caller
# (legacy journey harness included) keeps its current completion behaviour;
# the production voice endpoint enables it explicitly.
ADDITIONAL_INFORMATION_QUESTION = (
    "Before we finish, is there anything else you'd like to share about what happened?"
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
    r"(?i)^(?:my\s+name\s+is|name\s+is|this\s+is|it'?s|i\s*'?m|i\s+am)\s+"
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
    r"|you(?:'re|\s+are)|isn'?t|indeed|thanks?|thank\s+you|please"
    r"|correct|right|fine|ok|okay|sure|true)\b"
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


def _classify_confirmation_turn(message: str) -> tuple[str, str]:
    """Classify a reply to a pending read-back.

    Returns ``(verdict, payload)`` where verdict is one of ``"yes"``,
    ``"no"``, ``"correction"`` or ``"ambiguous"`` and payload is what to
    feed to ``respond_to_confirmation`` (canonical yes/no or the corrected
    text). Ambiguous replies must not advance the workflow; the caller
    clarifies instead.
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
    lead = _NEGATION_LEAD.match(stripped)
    if lead:
        candidate = _clean_correction(lead.group(1))
        # A negation-led reply whose remainder is itself a negation ("no,
        # nahi") is a clear rejection, not a corrected value.
        if candidate and _NEGATION.search(candidate) and not _AFFIRM.search(candidate):
            return "no", "no"
        if _looks_like_value(candidate):
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
    if _looks_like_value(candidate):
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
    "patient_age": "Could you tell me your age in years, or your date of birth?",
    "consent_to_contact": "Would it be alright if we contact you later - yes or no?",
    "contact_method": "Which is easier for you - phone, email, or both?",
    "contact_phone": "Could you repeat the phone number, digit by digit?",
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
        # First reliable conversation timestamp/timezone seen, used as a
        # fallback reference for onset normalization across turns.
        self._conversation_timestamp: Optional[Any] = None
        self._conversation_timezone: Optional[Any] = None

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
        turn_context = {
            "case_state": self.engine.current_state.value,
            "case_id": self.engine.case.case_id,
            "priority": self.engine.case.priority,
            "collected": self._collected_snapshot(),
            "next_field": self.engine.next_field() if interview_started else None,
        }
        if context:
            turn_context.update(context)

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
        # Set when a suspect product other than the configured demo product
        # (Lantus SoloStar) is captured; the reply then carries the bounded
        # prototype-scope message (never a submission claim).
        product_scope_notice: Optional[str] = None

        # 4-DOB. A single answer with both a date of birth and an age that
        # disagree by more than a year: hold both values, apply neither, and
        # ask exactly one targeted clarification (never silently choose).
        conflict_prompt: Optional[str] = None
        raw_conflict = (llm_result.extracted_fields or {}).get("dob_age_conflict")
        if isinstance(raw_conflict, dict):
            conflict_dob = str(raw_conflict.get("dob") or "").strip()
            conflict_age = str(raw_conflict.get("age") or "").strip()
            if conflict_dob and conflict_age and not self.engine.case.patient_age:
                self._dob_age_conflict = {"dob": conflict_dob, "age": conflict_age}
                conflict_prompt = _dob_age_clarification(conflict_dob, conflict_age)

        # 4a. Explicit confirmation sub-state. While a read-back is pending
        # for an integrated field, the turn is a confirmation reply (yes / no
        # / correction), not a new field answer; anything ambiguous is
        # clarified without advancing.
        confirmation_handled = False
        confirmation_rejected = False
        pending_field = self.engine.case.pending_confirmation
        if pending_field is not None and pending_field in self._confirm_fields:
            awaiting_correction = self.engine.case.confirmation_response == "no"
            verdict, payload = _classify_confirmation_turn(message)
            if verdict == "ambiguous" or (
                verdict == "yes" and awaiting_correction
            ):
                clarification = _clarification_for(pending_field, awaiting_correction)
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
        if pending_field is not None and pending_field in self._confirm_fields:
            if confirmation_handled and confirmation_rejected:
                pass
            elif confirmation_handled and self.engine.case.pending_confirmation is None:
                self._confirmation_attempts.pop(pending_field, None)
            else:
                attempts = self._confirmation_attempts.get(pending_field, 0) + 1
                self._confirmation_attempts[pending_field] = attempts
                if attempts >= _MAX_CONFIRMATION_ATTEMPTS:
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
        # is applied on that turn.
        if not confirmation_handled:
            for field_name, field_value in extracted.items():
                if field_name == "product" and self.engine.case.product:
                    # The original product is never overwritten once captured.
                    continue
                if field_name == "adverse_event":
                    # Never replace the original description; merge in a newly
                    # reported symptom so every symptom is preserved.
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
                except Exception:
                    # Skip invalid fields per engine validation rules
                    pass

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
        # the optional closing question is enabled it is asked exactly once
        # after intake is otherwise complete, and the case completes on the
        # following turn (the answer is recorded verbatim).
        ready_to_complete = (
            self.engine.can_complete()
            and self.engine.current_state is not WorkflowState.CASE_COMPLETED
        )
        additional_info_provided = False
        if ready_to_complete:
            if self._ask_additional_information and not self._additional_info_done:
                if not self._additional_info_asked:
                    self._additional_info_asked = True
                else:
                    if _is_decline(message):
                        self.engine.decline_additional_information()
                    else:
                        self.engine.record_additional_information(message)
                        additional_info_provided = True
                    self._additional_info_done = True
                    self.engine.complete()
            else:
                self.engine.complete()

        # 5. Determine next info
        missing = self.engine.missing_information()
        next_field = self.engine.next_field()

        # 5a. If the same field is still pending and the reply advanced nothing,
        # offer one short, differently-worded clarification rather than looping
        # on the identical question indefinitely.
        if next_field is not None:
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
        # under human review) keep their own behaviour.
        if conflict_prompt is not None and next_field is not None:
            next_field = ("patient_age", conflict_prompt)

        # 5c. An out-of-scope suspect product ends routine intake for this
        # session: no further questions are asked (urgent safety handling
        # above has already had its chance).
        if self._out_of_scope_product is not None:
            next_field = None

        # 6. Build response
        response_parts = []
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
            if additional_info_provided:
                response_parts.append("Thank you for sharing that.")
            elif is_ae:
                if not self._warning_shown:
                    response_parts.append(
                        "Thanks for telling me, and I'm sorry you're going through this. "
                        "I'll take a few details so we can record it safely."
                    )
                    response_parts.append("Important: Please consult a qualified physician or healthcare professional for evaluation and treatment of the reported side effects. Do not self-diagnose or change your medication without medical advice.")
                    self._warning_shown = True
            elif applied_any and not collecting_additional:
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
                response_parts.append(question)
            elif self.engine.current_state is WorkflowState.CASE_COMPLETED:
                if self.engine.case.consent_to_contact is False:
                    response_parts.append(
                        "Thank you for sharing this information. We've recorded what "
                        "you've provided. Since you haven't consented to follow-up "
                        "contact, we won't contact you using these details."
                    )
                else:
                    response_parts.append(
                        "Thank you - I've recorded everything we need for now. "
                        "A member of our team will follow up if needed. Take care."
                    )
            elif self._additional_info_asked and not self._additional_info_done:
                response_parts.append(ADDITIONAL_INFORMATION_QUESTION)

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
            next_question=list(next_field) if next_field else None,
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
            "contact_phone": case.contact_phone,
            "contact_email": case.contact_email,
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
