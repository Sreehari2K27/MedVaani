"""Deterministic adverse event (AE) workflow and state engine (prototype).

Python/FastAPI owns the state machine, required prototype fields, validation,
missing-information / next-question computation, escalation to human review,
and case status/completion. There is deliberately no LLM in this engine.

Clinical boundary:
    This engine makes NO clinical or causality decisions. It never diagnoses,
    determines causality, or recommends treatment, and it never judges whether
    free text is medically serious. Values such as ``seriousness`` and
    ``outcome`` are plain structured claims supplied by the caller/reporter,
    validated only against allowed values. Urgency is never inferred here;
    escalation is an explicit external signal via ``mark_urgent()``.

Data boundary:
    Prototype only. Use fictional / demo data. Not designed to store real
    patient data.

Design:
    ``AECase`` is a plain data record. ``AEStateEngine`` owns every rule
    (required fields, transitions, completion, escalation). A future LLM can
    provide intent detection and extraction without controlling the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from app.workflow.onset import format_iso_date


class WorkflowState(str, Enum):
    CASE_CREATED = "CASE_CREATED"
    PRODUCT_IDENTIFIED = "PRODUCT_IDENTIFIED"
    EVENT_IDENTIFIED = "EVENT_IDENTIFIED"
    PATIENT_IDENTIFIED = "PATIENT_IDENTIFIED"
    REPORTER_IDENTIFIED = "REPORTER_IDENTIFIED"
    CONTACT_CAPTURED = "CONTACT_CAPTURED"
    CONSENT_CAPTURED = "CONSENT_CAPTURED"
    EVENT_TIMING_CAPTURED = "EVENT_TIMING_CAPTURED"
    SERIOUSNESS_SCREENED = "SERIOUSNESS_SCREENED"
    OUTCOME_CAPTURED = "OUTCOME_CAPTURED"
    OPTIONAL_DETAILS = "OPTIONAL_DETAILS"
    CASE_REVIEW = "CASE_REVIEW"
    CASE_COMPLETED = "CASE_COMPLETED"
    HUMAN_REVIEW = "HUMAN_REVIEW"


# Normal-progression states in interview order. CASE_COMPLETED and
# HUMAN_REVIEW are terminal / forced and never reached by auto-advance.
# The documented AE voice flow asks for the event description first, then the
# medication, collects the contact number before consent, and screens
# seriousness before the incident date, so EVENT_IDENTIFIED precedes
# PRODUCT_IDENTIFIED, CONTACT_CAPTURED precedes CONSENT_CAPTURED and
# SERIOUSNESS_SCREENED precedes EVENT_TIMING_CAPTURED.
_PROGRESSION: tuple[WorkflowState, ...] = (
    WorkflowState.CASE_CREATED,
    WorkflowState.EVENT_IDENTIFIED,
    WorkflowState.PRODUCT_IDENTIFIED,
    WorkflowState.REPORTER_IDENTIFIED,
    WorkflowState.PATIENT_IDENTIFIED,
    WorkflowState.CONTACT_CAPTURED,
    WorkflowState.CONSENT_CAPTURED,
    WorkflowState.SERIOUSNESS_SCREENED,
    WorkflowState.EVENT_TIMING_CAPTURED,
    WorkflowState.OUTCOME_CAPTURED,
    WorkflowState.OPTIONAL_DETAILS,
    WorkflowState.CASE_REVIEW,
)

# Which field keys must be captured to satisfy each state. Every key listed
# for a state must be satisfied (_state_satisfied uses all(...)); the contact
# group is special-cased in _group_satisfied: it is the patient contact
# number, which the flow collects before consent is ever asked.
_STATE_REQUIREMENTS: dict[WorkflowState, tuple[str, ...]] = {
    WorkflowState.PRODUCT_IDENTIFIED: ("product",),
    WorkflowState.EVENT_IDENTIFIED: ("adverse_event",),
    WorkflowState.PATIENT_IDENTIFIED: ("patient_identifier", "patient_age"),
    WorkflowState.REPORTER_IDENTIFIED: (),  # handled specially (role / name)
    WorkflowState.CONTACT_CAPTURED: ("contact_group",),
    WorkflowState.CONSENT_CAPTURED: ("consent_to_contact",),
    WorkflowState.EVENT_TIMING_CAPTURED: ("onset_date",),
    WorkflowState.SERIOUSNESS_SCREENED: ("seriousness",),
    # The outcome is no longer an interview question: once the applicable
    # information is collected, the report is generated without asking how
    # the person is feeling. A volunteered outcome is still captured; when
    # none is given the field stays unset (unknown).
    WorkflowState.OUTCOME_CAPTURED: (),
    WorkflowState.OPTIONAL_DETAILS: (),
    WorkflowState.CASE_REVIEW: (),
    WorkflowState.CASE_CREATED: (),
}

def _question_for(key: str, case: "AECase") -> str:
    """Plain, person-aware phrasing for the next missing field.

    Wording follows the documented AE voice flow: one clear spoken question
    per field, second person ("you") unless the person is reporting on behalf
    of someone else, in which case the patient is referred to in the third
    person. Technical PV wording ("adverse event", "reporter") is never shown
    to the person.
    """
    third_person = case.reporter_role == "someone_else"
    if key == "product":
        # The one-time empathy line is prepended by the conversation service,
        # so the spoken question reads as the approved compassionate
        # transition into medication identification.
        return (
            "To help us record your report correctly, could you tell me the "
            "name of the medication involved?"
        )
    if key == "adverse_event":
        return (
            "Please tell me what happened, in your own words. "
            "Take your time\u2014I'm listening."
        )
    if key == "reporter_role":
        return (
            "May I ask whether you're the person who experienced this event, "
            "or whether you're reporting it on someone else's behalf?"
        )
    if key == "patient_identifier":
        # No date format is dictated: accepted formats are handled by the
        # extraction layer and stored dates are normalized to DD-MMM-YYYY.
        if third_person:
            return (
                "Could you please tell me the patient's first and last name, "
                "along with their date of birth?"
            )
        return (
            "Could you please tell me your first and last name, along with "
            "your date of birth?"
        )
    if key == "patient_age":
        # Continuation of the combined name-and-date-of-birth question when
        # only the name was given: the date of birth is still outstanding.
        if third_person:
            return "Could you please tell me the patient's date of birth?"
        return "Could you please tell me your date of birth?"
    if key == "reporter_name":
        return "Could you please tell me your full name?"
    if key == "reporter_contact_phone":
        return "Could you please share your contact number?"
    if key == "consent_to_contact":
        return (
            "Would it be okay for the appropriate team to contact the patient "
            "about this report if any follow-up is needed?"
        )
    if key == "contact_method":
        return "Would you prefer to be contacted by phone, by email, or both?"
    if key == "contact_phone":
        if third_person:
            return "Could you please share the patient's contact number?"
        return "Could you please share the best contact number for you?"
    if key == "contact_email":
        # Kept in sync with EMAIL_REQUEST_QUESTION in the conversation
        # service: dictated/spelled addresses are invited explicitly, with
        # the letter-pause instruction so adjacent same-sound letters
        # (e.g. a doubled E) are dictated as separate tokens.
        return (
            "Could you please spell out your email address, one letter or "
            "number at a time, with a short pause between letters? You can "
            "say 'at the rate' for @ and 'dot' for a period."
        )
    if key == "onset_date":
        return "Could you tell me when the event occurred?"
    if key == "seriousness":
        return "Would you say that the event was severe or life-threatening?"
    if key == "outcome":
        return (
            "How is the patient feeling now\u2014have the symptoms improved, "
            "stayed the same, or gotten worse?"
            if third_person
            else "How are you feeling now\u2014have your symptoms improved, "
            "stayed the same, or gotten worse?"
        )
    return f"Could you tell me a bit more about the {key.replace('_', ' ')}?"


OUTCOME_OPTIONS = frozenset(
    {"recovered", "recovering", "worsening", "not_recovered", "unresolved", "unknown", "fatal"}
)

_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "reporter_role": frozenset({"self", "someone_else"}),
    "priority": frozenset({"normal", "high"}),
    "seriousness": frozenset({"yes", "no", "unknown"}),
    "outcome": OUTCOME_OPTIONS,
    "patient_sex": frozenset({"male", "female", "other", "unknown"}),
    "preferred_contact_method": frozenset({"phone", "email", "both"}),
    "medical_care": frozenset({"yes", "no", "unknown"}),
}

# Public alias of the per-field allowed-value domains, shared with the
# provider-output validator so enum rules never drift between layers.
ALLOWED_FIELD_VALUES: dict[str, frozenset[str]] = _ALLOWED_VALUES

_IMMUTABLE_FIELDS = frozenset({"case_id", "status", "human_review_reason"})

# The structured identity fields whose stored value is the properly
# capitalized display form: the raw transcript is never modified (the
# audit trail keeps the reporter's own words); only these structured
# fields are formatted.
_NAME_FIELDS = frozenset({"patient_identifier", "reporter_name"})


def _title_case_latin_name(value: str) -> str:
    """Capitalize the first alphabetic character of each Latin-script word
    ("raju sharma" -> "Raju Sharma"). The remaining letters are never
    changed (McDonald, Lantus SoloStar keep their internal capitalization)
    and non-Latin scripts (no case mapping) are untouched. No component is
    invented, expanded, or removed; punctuation and whitespace are kept."""
    titled = []
    for word in value.split(" "):
        if word and word[0].isascii() and word[0].isalpha() and word[0].islower():
            titled.append(word[0].upper() + word[1:])
        else:
            titled.append(word)
    return " ".join(titled)

# Normalised replies to a read-back question ("I heard X. Is that correct?").
# Anything outside these sets is treated as a corrected value.
_AFFIRMATIVE_REPLIES = frozenset(
    {
        "yes", "y", "yeah", "yep", "yup",
        "correct", "right",
        "that's correct", "that is correct", "that's right", "that is right",
        "ok", "okay", "confirmed", "true",
        # Standalone affirmative replies in the eight newly supported
        # languages, exact full-reply matches only: a sentence is never a
        # confirmation. The observed Malayalam forms ("അതെ",
        # "അതെ തന്നെ") are pinned; "ശരിയാണ്" ("that's correct") is the
        # observed browser reply that must confirm, "അതേ" is its
        # standard spelling variant, "അത് ശരിയാണ്"/"അത് ശരി" are the
        # pronoun-led echoes of the read-back question, and "യെസ്"/"യസ്"
        # pronoun-led echoes of the read-back question, "യെസ്"/"യസ്"
        # are the Malayalam-script transliterations of spoken "yes" that
        # the browser STT emits, "हो"/"हो बरोबर आहे"/"होय ते बरोबर आहे"
        # are the observed Marathi affirmative and its compounds, and
        # "hoy"/"ho"/"barobar" are their Latin-script transliterations.
        "होय", "बरोबर",
        "হ্যাঁ", "ঠিক আছে", "ਹਾਂ", "ਠੀਕ ਹੈ", "હા", "સાચું",
        "ಹೌದು", "ಸರಿ", "అవును", "సరే", "సరి", "ஆம்", "சரி",
        "അതെ", "അതെ തന്നെ", "അതേ", "ശരി", "ശരിയാണ്",
        "അത് ശരിയാണ്", "അത് ശരി",
        "യെസ്", "യസ്",
        "हो", "हो बरोबर आहे", "बरोबर आहे", "होय बरोबर आहे", "होय ते बरोबर आहे",
        "hoy", "ho", "barobar",
        "ഓക്കെ", "ആ അതെ", "ആ ശരിയാണ്",
    }
)

_NEGATIVE_REPLIES = frozenset(
    {
        "no", "n", "nope", "wrong", "incorrect",
        "that's wrong", "that is wrong", "not correct", "not right",
        # Standalone negative replies in the eight newly supported languages
        # (exact full-reply matches only; Hindi keeps its existing path).
        "नाही",
        "না", "নয়", "ਨਹੀਂ", "ਨਾ", "ના", "નહીં",
        "ಇಲ್ಲ", "కాదు", "లేదు", "இல்லை", "ഇല്ല", "അല്ല",
    }
)


def _classify_confirmation_reply(reply: str) -> str:
    """Classify a read-back answer as ``"yes"``, ``"no"``, or ``"correction"``.

    Deterministic and text-only: the caller/LLM layer normalises free speech
    into short answers; any unrecognised reply is taken as a corrected value.
    """
    normalized = reply.strip().lower().rstrip(" .!?,").rstrip("\u0964").strip()
    if normalized in _AFFIRMATIVE_REPLIES:
        return "yes"
    if normalized in _NEGATIVE_REPLIES:
        return "no"
    return "correction"


def _new_case_id() -> str:
    return f"AE-{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex[:8]}-demo"


class AECaseError(Exception):
    """Base error for the AE workflow engine."""


class AECaseValidationError(AECaseError):
    """A field update was rejected: unknown or out-of-range value."""


class AECaseIncompleteError(AECaseError):
    """Completion was attempted while required information was missing."""


@dataclass
class AECase:
    """Plain structured record for one adverse event (prototype/demo only)."""

    case_id: str
    status: WorkflowState = WorkflowState.CASE_CREATED
    priority: str = "normal"
    product: str | None = None
    adverse_event: str | None = None
    patient_identifier: str | None = None
    patient_age: str | None = None
    patient_sex: str | None = None
    reporter_role: str | None = None
    reporter_name: str | None = None
    # The reporter's own contact number, kept apart from the patient's
    # ``contact_phone`` so a report on someone else's behalf never mixes the
    # two identities or the two numbers.
    reporter_contact_phone: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    # Explicit refusal of the optional email question. Consent to follow-up
    # contact can still be "yes" while the email address stays unavailable.
    contact_email_declined: bool = False
    preferred_contact_method: str | None = None
    consent_to_contact: bool | None = None
    onset_date: str | None = None
    # Derived onset interpretation (never replaces the original wording):
    #   onset_date_normalized: ISO YYYY-MM-DD, or None when unresolved.
    #   onset_period: morning / afternoon / evening / night (time stays
    #     approximate; an exact clock time is never invented).
    #   onset_date_source: "explicit" | "normalized" | "unknown".
    onset_date_normalized: str | None = None
    onset_period: str | None = None
    onset_date_source: str | None = None
    medical_care: str | None = None
    seriousness: str = "unknown"
    outcome: str | None = None
    # Voluntarily supplied details preserved verbatim (never required fields).
    dose_and_use: str | None = None
    prescribing_physician: str | None = None
    concomitant_medications: list[str] = field(default_factory=list)
    # Free-form answer to the closing "anything else?" question, kept verbatim
    # and separate from the structured fields above. Declining is recorded
    # explicitly so the report can state "None provided" rather than nothing.
    additional_information: str | None = None
    additional_information_declined: bool = False
    # The person's first free-text message, kept verbatim when it could not
    # yet be classified as an adverse event. Never a substitute for
    # adverse_event and never filled from later structured answers.
    unclassified_initial_input: str | None = None
    human_review_reason: str | None = None
    # Confirmation tracking
    confirmed_fields: set[str] = field(default_factory=set)
    pending_confirmation: str | None = None  # field awaiting user confirmation
    confirmation_candidate: dict[str, str] = field(default_factory=dict)  # pending values to confirm
    confirmation_response: str | None = None  # user's response to confirmation (yes/no/correction)


class AEStateEngine:
    """Deterministic state machine driving one adverse event case."""

    def __init__(self, case_id: str) -> None:
        self.case = AECase(case_id=case_id)

    @classmethod
    def create_case(cls, case_id: str | None = None) -> "AEStateEngine":
        case_id = case_id if case_id is not None else _new_case_id()
        return cls(case_id)

    # ---- state tracking ---------------------------------------------------

    @property
    def current_state(self) -> WorkflowState:
        return self.case.status

    def _recompute_status(self) -> None:
        if self.case.status in (
            WorkflowState.HUMAN_REVIEW,
            WorkflowState.CASE_COMPLETED,
        ):
            return
        new_status = WorkflowState.CASE_CREATED
        for state in _PROGRESSION:
            if not self._state_satisfied(state):
                break
            new_status = state
        self.case.status = new_status

    def _state_satisfied(self, state: WorkflowState) -> bool:
        if state is WorkflowState.REPORTER_IDENTIFIED:
            return self._reporter_identity_satisfied()
        return all(
            self._group_satisfied(key) for key in _STATE_REQUIREMENTS.get(state, ())
        )

    def _reporter_identity_satisfied(self) -> bool:
        # An unconfirmed name never satisfies identity, even if an older
        # value is still on the case.
        if self.case.pending_confirmation in ("patient_identifier", "reporter_name"):
            return False
        role = self.case.reporter_role
        if role == "someone_else":
            # Reporting on someone else's behalf: the reporter's own name and
            # contact number are both collected before patient details.
            if self.case.pending_confirmation == "reporter_contact_phone":
                return False
            return bool(self.case.reporter_name) and bool(
                self.case.reporter_contact_phone
            )
        if role == "self":
            # Reporter is the patient: the patient identity is not duplicated.
            return bool(self.case.patient_identifier)
        return False

    def _group_satisfied(self, key: str) -> bool:
        # A field whose read-back is still pending is never satisfied: the
        # candidate is held in confirmation_candidate, not on the case, so
        # progression and completion stay blocked until "yes".
        if self.case.pending_confirmation == key:
            return False
        if key == "contact_group":
            # The patient contact number is collected before consent is asked
            # (see the documented flow); it is a required interview answer.
            return self._contact_complete()
        if key == "seriousness":
            return self.case.seriousness in {"yes", "no"}
        if key == "consent_to_contact":
            # Captured once answered either way; only True unlocks contact.
            return self.case.consent_to_contact is not None
        return bool(getattr(self.case, key))

    # ---- required information / next question -----------------------------

    def _contact_complete(self) -> bool:
        """Whether the patient contact number has been captured.

        The contact number is asked for before consent is ever raised, so it
        is required regardless of the consent answer. A pending read-back
        keeps the field unsatisfied until the number is confirmed.
        """
        if self.case.pending_confirmation == "contact_phone":
            return False
        return bool(self.case.contact_phone)

    def _contact_pending(self) -> list[str]:
        """Keys still needed to reach the patient (asked before consent)."""
        return [] if self._contact_complete() else ["contact_phone"]

    def _unsatisfied_groups(self) -> list[str]:
        # No early "pause" shortcut here: a pending confirmation keeps its
        # field unsatisfied via _group_satisfied, so the field stays in
        # missing_information and completion stays blocked.
        groups: list[str] = []
        for state in _PROGRESSION:
            if state is WorkflowState.REPORTER_IDENTIFIED:
                role = self.case.reporter_role
                if role is None:
                    groups.append("reporter_role")
                elif role == "someone_else":
                    # Reporter branch: name, then the reporter's own contact
                    # number, both before any patient detail is asked.
                    if (
                        not self.case.reporter_name
                        or self.case.pending_confirmation == "reporter_name"
                    ):
                        groups.append("reporter_name")
                    elif (
                        not self.case.reporter_contact_phone
                        or self.case.pending_confirmation
                        == "reporter_contact_phone"
                    ):
                        groups.append("reporter_contact_phone")
                continue
            if state is WorkflowState.CONTACT_CAPTURED:
                groups.extend(self._contact_pending())
                continue
            for key in _STATE_REQUIREMENTS.get(state, ()):
                if not self._group_satisfied(key):
                    groups.append(key)
        return groups

    def missing_information(self) -> list[str]:
        """Keys of required prototype fields not yet captured (in interview order)."""
        return self._unsatisfied_groups()

    def _confirmation_question(self) -> str | None:
        if getattr(self.case, 'pending_confirmation', None) is None:
            return None
        field = self.case.pending_confirmation
        # Read the interpreted value back in plain language; the value itself is
        # interpolated unchanged (names, digits, addresses are never altered).
        if field == "patient_identifier":
            name = self.case.confirmation_candidate.get('patient_identifier', self.case.patient_identifier or '')
            dob = self.case.patient_age
            if not dob:
                # Only the name is known so far: the date of birth is still
                # collected (the combined question) before any read-back.
                return None
            if self.case.reporter_role == "someone_else":
                return (
                    "Let me read that back to make sure I've recorded it "
                    f"correctly: {name}, {dob}. Is that right?"
                )
            return f"I have your details as {name}, {dob}. Is that correct?"
        if field == "reporter_name":
            name = self.case.confirmation_candidate.get('reporter_name', self.case.reporter_name or '')
            return f"I have your name as {name}. Have I got that right?"
        if field in ("contact_phone", "reporter_contact_phone"):
            phone = self.case.confirmation_candidate.get(
                field, getattr(self.case, field, '') or ''
            )
            # The count is derived from the exact candidate being confirmed,
            # so a doubled or missing digit is audible at confirmation time
            # while the number itself is never read aloud.
            digits = "".join(ch for ch in str(phone) if ch.isdigit())
            tail = digits[-4:] if len(digits) >= 4 else digits
            return f"I have a {len(digits)}-digit number ending in {tail}. Is that correct?"
        if field == "contact_email":
            email = self.case.confirmation_candidate.get('contact_email', self.case.contact_email or '')
            return f"I heard your email address as {email}. Is that correct?"
        # Generic fallback
        val = self.case.confirmation_candidate.get(field) or getattr(self.case, field, '')
        return f"I heard {val}. Is that correct?"

    def next_field(self) -> tuple[str, str] | None:
        """Return ``(field_key, question)`` for the next required field.

        Returns ``None`` once questioning should pause (human review,
        completion-ready case, or completed case).
        """
        if self.case.status in (WorkflowState.HUMAN_REVIEW, WorkflowState.CASE_COMPLETED):
            return None
        if getattr(self.case, 'pending_confirmation', None) is not None:
            if self.case.confirmation_response == "no":
                # Read-back rejected: re-ask for a corrected value before
                # anything else can advance.
                return self.case.pending_confirmation, _question_for(
                    self.case.pending_confirmation, self.case
                )
            if (
                self.case.pending_confirmation == "patient_identifier"
                and not self.case.patient_age
            ):
                # Name held for the combined name-and-date-of-birth read-back:
                # collect the outstanding date of birth first (the read-back
                # happens once both values are known).
                return "patient_age", _question_for("patient_age", self.case)
            q = self._confirmation_question()
            if q:
                return ('confirmation', q)
            return None
        groups = self._unsatisfied_groups()
        if not groups:
            return None
        key = groups[0]
        return key, _question_for(key, self.case)

    # ---- updates ----------------------------------------------------------

    def update(self, field: str, value: object) -> "AEStateEngine":
        """Accept one structured field value and recompute the workflow state."""
        if field in _IMMUTABLE_FIELDS:
            raise AECaseValidationError(f"Field '{field}' cannot be updated directly.")
        if not hasattr(self.case, field):
            raise AECaseValidationError(f"Unknown field '{field}'.")

        pending = self.case.pending_confirmation
        if pending == field:
            if self.case.confirmation_response == "no":
                # The read-back was rejected: this value is a correction and
                # is captured as a new candidate for another read-back.
                self.begin_confirmation(field, value)
                return self
            raise AECaseValidationError(
                f"'{field}' has a confirmation pending; answer yes or no first."
            )

        if field == "concomitant_medications":
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise AECaseValidationError("concomitant_medications must be a list of strings.")
            setattr(self.case, field, list(value))
        elif field == "consent_to_contact":
            if not isinstance(value, bool):
                raise AECaseValidationError("consent_to_contact must be a boolean.")
            self.case.consent_to_contact = value
        elif field == "contact_email_declined":
            if not isinstance(value, bool):
                raise AECaseValidationError(
                    "contact_email_declined must be a boolean."
                )
            self.case.contact_email_declined = value
        elif field in _ALLOWED_VALUES:
            self._set_checked(field, value, _ALLOWED_VALUES[field])
        else:
            if not isinstance(value, str) or not value.strip():
                raise AECaseValidationError(f"'{field}' must be a non-empty string.")
            stored = value.strip()
            if field in _NAME_FIELDS:
                stored = _title_case_latin_name(stored)
            setattr(self.case, field, stored)

        if field == "reporter_role" and value == "self":
            # Reporter is the patient: do not keep a duplicate reporter name
            # or a duplicate reporter contact number.
            self.case.reporter_name = None
            self.case.reporter_contact_phone = None

        if field == "onset_date":
            # A new/corrected statement invalidates the derived interpretation;
            # the workflow re-normalizes it against the conversation reference.
            self.case.onset_date_normalized = None
            self.case.onset_period = None
            self.case.onset_date_source = None

        self._recompute_status()
        return self

    def _set_checked(self, field: str, value: object, allowed: frozenset[str]) -> None:
        if not isinstance(value, str) or value not in allowed:
            raise AECaseValidationError(
                f"'{field}' must be one of: {', '.join(sorted(allowed))}."
            )
        setattr(self.case, field, value)

    # ---- explicit confirmation ---------------------------------------------

    def begin_confirmation(self, field: str, candidate: str) -> str | None:
        """Capture ``candidate`` for ``field`` and enter the read-back state.

        The candidate is stored in ``confirmation_candidate`` and is NOT
        written to the case until the read-back is answered affirmatively.
        While a confirmation is pending, ``next_field()`` asks the read-back
        question and the workflow cannot advance past (or complete on) this
        field. Returns the read-back question to ask, or ``None`` when the
        read-back itself is still waiting for another value (the combined
        name-and-date-of-birth read-back).
        """
        if field in _IMMUTABLE_FIELDS:
            raise AECaseValidationError(f"Field '{field}' cannot be confirmed.")
        if not hasattr(self.case, field):
            raise AECaseValidationError(f"Unknown field '{field}'.")
        pending = self.case.pending_confirmation
        if pending is not None and pending != field:
            raise AECaseValidationError(
                f"A confirmation is already pending for '{pending}'."
            )
        if not isinstance(candidate, str) or not candidate.strip():
            raise AECaseValidationError(f"'{field}' must be a non-empty string.")
        self.case.pending_confirmation = field
        stored_candidate = candidate.strip()
        if field in _NAME_FIELDS:
            stored_candidate = _title_case_latin_name(stored_candidate)
        self.case.confirmation_candidate = {field: stored_candidate}
        self.case.confirmation_response = None
        self.case.confirmed_fields.discard(field)
        return self._confirmation_question()

    def respond_to_confirmation(self, reply: str) -> str | None:
        """Answer a pending read-back. Returns the next question to ask.

        - affirmative (``yes`` / ``correct`` ...): the candidate is committed
          to the case and the workflow advances.
        - negative (``no`` / ``wrong`` ...): the candidate is discarded and
          the field is asked again so the reporter can correct it.
        - anything else: treated as a corrected value and read back again.
        """
        field = self.case.pending_confirmation
        if field is None:
            raise AECaseValidationError("No confirmation is pending.")
        if not isinstance(reply, str) or not reply.strip():
            raise AECaseValidationError("Confirmation reply must be a non-empty string.")

        verdict = _classify_confirmation_reply(reply)
        if verdict == "yes":
            value = self.case.confirmation_candidate.get(field) or getattr(
                self.case, field, None
            )
            if not isinstance(value, str) or not value.strip():
                raise AECaseValidationError(
                    f"No captured value to confirm for '{field}'."
                )
            setattr(self.case, field, value.strip())
            self.case.confirmed_fields.add(field)
            self.case.confirmation_candidate.pop(field, None)
            self.case.pending_confirmation = None
            self.case.confirmation_response = "yes"
            self._recompute_status()
        elif verdict == "no":
            # Keep the confirmation open with a "needs correction" marker so
            # the field stays unsatisfied until a corrected value is captured
            # and confirmed in a later turn.
            self.case.confirmation_candidate.pop(field, None)
            self.case.confirmed_fields.discard(field)
            self.case.confirmation_response = "no"
        else:
            self.begin_confirmation(field, reply.rstrip(" .!?"))

        next_field = self.next_field()
        return None if next_field is None else next_field[1]

    # ---- optional closing detail -------------------------------------------

    def record_additional_information(self, text: str) -> None:
        """Store the reporter's free-form closing answer verbatim.

        This never writes into the structured fields: extraction into those is
        the caller's job and must only fill empty fields.
        """
        if not isinstance(text, str) or not text.strip():
            raise AECaseValidationError(
                "Additional information must be a non-empty string."
            )
        self.case.additional_information = text.strip()
        self.case.additional_information_declined = False

    def decline_additional_information(self) -> None:
        """Record that the reporter had nothing else to add."""
        self.case.additional_information = None
        self.case.additional_information_declined = True

    @property
    def additional_information_status(self) -> str | None:
        if self.case.additional_information_declined:
            return "declined"
        if self.case.additional_information:
            return "provided"
        return None

    # ---- completion / escalation ------------------------------------------

    def can_complete(self) -> bool:
        if self.case.status is WorkflowState.HUMAN_REVIEW:
            return False
        if self.case.pending_confirmation is not None:
            return False
        return not self._unsatisfied_groups()

    def complete(self) -> "AEStateEngine":
        if self.case.status is WorkflowState.HUMAN_REVIEW:
            raise AECaseIncompleteError(
                "Case is under human review and cannot be auto-completed."
            )
        if self.case.pending_confirmation is not None:
            raise AECaseIncompleteError(
                f"Confirmation is pending for '{self.case.pending_confirmation}'."
            )
        missing = self._unsatisfied_groups()
        if missing:
            raise AECaseIncompleteError(
                f"Missing required information: {', '.join(missing)}."
            )
        self.case.status = WorkflowState.CASE_COMPLETED
        return self

    def mark_urgent(self, reason: str | None = None) -> "AEStateEngine":
        """Pause normal questioning and route the case to high-priority HUMAN_REVIEW.

        Called only when the application has already classified the input as
        potentially urgent. This engine never infers urgency itself. All
        collected information is preserved.
        """
        self.case.priority = "high"
        self.case.status = WorkflowState.HUMAN_REVIEW
        self.case.human_review_reason = reason
        return self

    def flag_for_review(self, reason: str | None = None) -> "AEStateEngine":
        """Pause normal questioning for HUMAN_REVIEW without escalating priority.

        Used when an answer cannot be resolved after clarification (e.g. an
        ambiguous seriousness reply recorded as ``unknown``): the interview
        stops and a qualified reviewer decides. Priority/urgency semantics are
        untouched - only ``mark_urgent`` raises priority.
        """
        self.case.status = WorkflowState.HUMAN_REVIEW
        self.case.human_review_reason = reason
        return self

    def resume_from_review(self) -> "AEStateEngine":
        """Leave HUMAN_REVIEW and recompute normal progression.

        Used only when a field that was parked for review is re-engaged by
        the reporter (a corrected candidate or a successful confirmation):
        the status returns to the normal progression given the fields
        actually satisfied, and the interview continues. Urgent escalations
        (``mark_urgent``) are never resumed through this path; the recorded
        ``human_review_reason`` is kept for traceability.
        """
        if self.case.status is not WorkflowState.HUMAN_REVIEW:
            return self
        self.case.status = WorkflowState.CASE_CREATED
        self._recompute_status()
        return self

    # ---- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Clean JSON-ready representation of the case for the prototype."""
        next_field = self.next_field()
        return {
            "case_id": self.case.case_id,
            "status": self.case.status.value,
            "priority": self.case.priority,
            "report_complete": self.case.status is WorkflowState.CASE_COMPLETED,
            "product": self.case.product,
            "adverse_event": self.case.adverse_event,
            "unclassified_initial_input": self.case.unclassified_initial_input,
            "patient": {
                "identifier": self.case.patient_identifier,
                "age": self.case.patient_age,
                "sex": self.case.patient_sex,
            },
            "reporter": {
                "role": self.case.reporter_role,
                "name": self.case.reporter_name,
                "same_as_patient": self.case.reporter_role == "self",
            },
            "contact": {
                "phone": self.case.contact_phone,
                "reporter_phone": self.case.reporter_contact_phone,
                "email": self.case.contact_email,
                "email_declined": self.case.contact_email_declined,
                "preferred_method": self.case.preferred_contact_method,
                "consent_to_contact": self.case.consent_to_contact,
            },
            "onset_date": self.case.onset_date,
            "onset_date_normalized": self.case.onset_date_normalized,
            "onset_date_display": format_iso_date(self.case.onset_date_normalized),
            "onset_period": self.case.onset_period,
            "onset_date_source": self.case.onset_date_source,
            "medical_care": self.case.medical_care,
            "seriousness": self.case.seriousness,
            "outcome": self.case.outcome,
            "dose_and_use": self.case.dose_and_use,
            "prescribing_physician": self.case.prescribing_physician,
            "concomitant_medications": list(self.case.concomitant_medications),
            "additional_information": self.case.additional_information,
            "additional_information_status": self.additional_information_status,
            "human_review_reason": self.case.human_review_reason,
            "pending_confirmation": self.case.pending_confirmation,
            "missing_information": self._unsatisfied_groups(),
            "next_question": None if next_field is None else list(next_field),
        }