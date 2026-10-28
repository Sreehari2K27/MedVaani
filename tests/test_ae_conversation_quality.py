"""Focused regression tests for the AE conversation-quality fixes.

Covers the observed live-conversation defects: opening shown once, concise
empathy-plus-medication wording, natural reporter-role answers, name/DOB
extraction and DD-MMM-YYYY normalization, combined-confirmation correction
handling, and report integrity (no silent consent, no completion with
unconfirmed identity). All data is synthetic.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.providers.answer import AnswerLLMProvider, normalized_dob_in
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    DETAILS_CORRECTION_QUESTION,
    EVENT_DESCRIPTION_QUESTION,
    INTRODUCTION_MESSAGE,
    MEDICATION_EMPATHY,
    SCOPE_VERIFICATION_QUESTION,
    VOICE_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)

# (affirmative, negative) replies per locale for the Step-4 medication
# confirmation. All are conservative standalone answers; the Malayalam forms
# are the observed browser-test phrases.
_CONFIRM_LANGUAGE_PHRASES = {
    "en-IN": ("yes", "no"),
    "hi-IN": ("हाँ", "नहीं"),
    "bn-IN": ("হ্যাঁ", "না"),
    "gu-IN": ("હા", "ના"),
    "kn-IN": ("ಹೌದು", "ಇಲ್ಲ"),
    "ml-IN": ("അതെ തന്നെ", "ഇല്ല"),
    "mr-IN": ("होय", "नाही"),
    "pa-IN": ("ਹਾਂ", "ਨਹੀਂ"),
    "ta-IN": ("ஆம்", "இல்லை"),
    "te-IN": ("అవును", "కాదు"),
}


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine,
        AnswerLLMProvider(),
        confirm_fields=VOICE_CONFIRMATION_FIELDS,
        ask_additional_information=True,
    )
    return engine, workflow


def _to_verified_self(case_id: str):
    """Workflow past Step 4 with the reporter-role question on the table."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")
    workflow.process_turn("yes")  # Step 4: medication verified
    return engine, workflow


def _to_combined_readback(case_id: str):
    """Workflow with a pending combined name/date-of-birth read-back."""
    engine = AEStateEngine.create_case(case_id)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_age", "27-May-2000")
    engine.begin_confirmation("patient_identifier", "Alpha Beta")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    return engine, workflow


# ---- 1. introduction once per session ---------------------------------------


def test_introduction_appears_once_per_session():
    engine, workflow = _workflow("q-intro")

    first = workflow.process_turn("Hello")
    assert first.response_text.startswith(INTRODUCTION_MESSAGE)
    # The opening ends with the event-description invitation, so the
    # adverse-event question is not asked twice.
    assert first.response_text.endswith("Please tell me what happened, in your own words.")
    assert first.next_question == ["adverse_event", EVENT_DESCRIPTION_QUESTION]

    described = workflow.process_turn("I took Lantus and have a headache")
    assert INTRODUCTION_MESSAGE not in described.response_text

    verified = workflow.process_turn("yes")
    assert INTRODUCTION_MESSAGE not in verified.response_text


def test_first_message_that_is_the_description_has_no_opening_repetition():
    engine, workflow = _workflow("q-intro-desc")
    result = workflow.process_turn("I took Lantus and have a headache")

    # The person has already described the event: no opening after it, and
    # the medication verification (Step 4) is asked instead.
    assert INTRODUCTION_MESSAGE not in result.response_text
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"


# ---- 2. concise empathy + medication wording --------------------------------


def test_post_event_response_is_empathy_then_medication_question():
    engine, workflow = _workflow("q-med")
    workflow.process_turn("Hello")

    result = workflow.process_turn("I have a headache")
    text = result.response_text
    question = (
        "To help us record your report correctly, could you tell me the name "
        "of the medication involved?"
    )

    empathy_at = text.index(MEDICATION_EMPATHY)
    question_at = text.index(question)
    # A short empathy statement immediately followed by the medication name.
    assert empathy_at < question_at
    assert text[empathy_at:question_at].strip() == MEDICATION_EMPATHY
    # No long safety paragraph before the question; the consultation note
    # still appears, after it.
    assert "Important: Please consult" not in text[:question_at]
    assert "Important: Please consult" in text[question_at:]
    assert result.next_question is not None
    assert result.next_question[0] == "product"


# ---- 3. reporter-role natural-language understanding ------------------------

SELF_ANSWERS = [
    "Myself",
    "I am the patient",
    "I experienced it",
    "I am reporting my own experience",
    "And the patient who experienced this event",
    "I am the one who experienced it",
]

SOMEONE_ELSE_ANSWERS = [
    "For my mother",
    "On behalf of someone else",
    "I'm reporting this for another person",
]


@pytest.mark.parametrize("answer", SELF_ANSWERS)
def test_natural_self_reporter_answers_select_the_patient_branch(answer):
    result = AnswerLLMProvider().interpret(
        answer, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "self"


@pytest.mark.parametrize("answer", SOMEONE_ELSE_ANSWERS)
def test_natural_someone_else_answers_select_the_reporter_branch(answer):
    result = AnswerLLMProvider().interpret(
        answer, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "someone_else"


def test_natural_self_answer_advances_to_the_patient_details_question():
    engine, workflow = _to_verified_self("q-role-nlu")

    result = workflow.process_turn("And the patient who experienced this event")

    assert engine.case.reporter_role == "self"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier"


# ---- multilingual reporter-role answers --------------------------------------

# (self, other) replies per locale. All are conservative standalone phrases;
# the Malayalam self forms are the observed browser-test utterances.
_ROLE_LANGUAGE_PHRASES = {
    "en-IN": ("It happened to me", "Reporting this for someone else"),
    "hi-IN": ("मुझे ही हुआ", "मेरी पत्नी को हुआ"),
    "bn-IN": ("আমার সাথেই ঘটেছে", "অন্য কারো জন্য জানাচ্ছি"),
    "gu-IN": ("મને જ સંબંધિત છે", "બીજા કોઈ માટે જણાવું છું"),
    "kn-IN": ("ನನಗೆ ಸಂಬಂಧಿಸಿದದ್ದು", "ಬೇರೊಬ್ಬರಿಗಾಗಿ ಹೇಳುತ್ತಿದ್ದೇನೆ"),
    "ml-IN": ("എനിക്ക് തന്നെയാണ് സംബന്ധിച്ചത്", "മറ്റൊരാൾക്ക് വേണ്ടി"),
    "mr-IN": ("मलाच संबंधित आहे", "दुसऱ्यासाठी सांगत आहे"),
    "pa-IN": ("ਮੈਨੂੰ ਹੀ ਸਬੰਧਤ ਹੈ", "ਕਿਸੇ ਹੋਰ ਲਈ ਦੱਸ ਰਿਹਾ ਹਾਂ"),
    "ta-IN": ("எனக்கு தான் சம்பந்தப்பட்டது", "வேறு ஒருவருக்காக சொல்கிறேன்"),
    "te-IN": ("నాకే సంబంధించినది", "వేరొకరి కోసం చెబుతున్నాను"),
}


@pytest.mark.parametrize("language", sorted(_ROLE_LANGUAGE_PHRASES))
def test_self_reporter_answers_in_every_locale_select_the_patient_branch(language):
    self_answer, _ = _ROLE_LANGUAGE_PHRASES[language]
    result = AnswerLLMProvider().interpret(
        self_answer, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "self", language


@pytest.mark.parametrize("language", sorted(_ROLE_LANGUAGE_PHRASES))
def test_someone_else_answers_in_every_locale_select_the_reporter_branch(language):
    _, other_answer = _ROLE_LANGUAGE_PHRASES[language]
    result = AnswerLLMProvider().interpret(
        other_answer, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "someone_else", language


def test_other_person_precedence_over_self_cues_in_mixed_answers():
    # A mixed answer that names another person must stay someone_else:
    # _ROLE_OTHER is checked before _ROLE_SELF.
    for text in (
        "No, it happened to my wife",
        "नहीं, मेरी पत्नी को हुआ",
        "ഇല്ല, എന്റെ ഭാര്യക്ക് സംഭവിച്ചത്",
    ):
        result = AnswerLLMProvider().interpret(
            text, {"next_field": ["reporter_role", "Are you the patient?"]}
        )
        assert result.extracted_fields.get("reporter_role") == "someone_else", text


def test_unrelated_answer_records_no_reporter_role():
    result = AnswerLLMProvider().interpret(
        "blah blah something random",
        {"next_field": ["reporter_role", "Are you the patient?"]},
    )
    assert "reporter_role" not in result.extracted_fields


# The three exact Malayalam utterances from the browser test.
MALAYALAM_SELF_UTTERANCES = [
    "എനിക്ക് തന്നെയാണ് സംബന്ധിച്ചത്",
    "എന്റെ സ്വന്തം അനുഭവമാണ്",
    "എന്റെ സ്വന്തം അനുഭവം തന്നെ",
]


@pytest.mark.parametrize("utterance", MALAYALAM_SELF_UTTERANCES)
def test_malayalam_self_utterances_classify_as_self(utterance):
    result = AnswerLLMProvider().interpret(
        utterance, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "self", utterance


@pytest.mark.parametrize("utterance", MALAYALAM_SELF_UTTERANCES)
def test_malayalam_self_utterances_advance_to_the_patient_details_question(
    utterance,
):
    engine, workflow = _to_verified_self("q-role-ml")

    result = workflow.process_turn(utterance)

    # The role is recorded and the approved patient-details question follows
    # instead of a repeated reporter-role prompt.
    assert engine.case.reporter_role == "self", utterance
    assert result.next_question is not None, utterance
    assert result.next_question[0] == "patient_identifier", utterance


# ---- 4-6. name/DOB extraction and DD-MMM-YYYY normalization ----------------


def test_name_and_date_of_birth_are_extracted_separately():
    result = AnswerLLMProvider().interpret(
        "Alpha Beta and my date of birth is twenty seven May two thousand",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )

    assert result.extracted_fields["patient_identifier"] == "Alpha Beta"
    assert result.extracted_fields["patient_age"] == "27-May-2000"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("27/05/2000", "27-May-2000"),
        ("27-05-2000", "27-May-2000"),
        ("Twenty seven May two thousand", "27-May-2000"),
        ("My date of birth is twenty seven May two thousand", "27-May-2000"),
        ("my dob is 1/1/2000", "01-Jan-2000"),
        # Ordinal forms spoken in the live conversation.
        ("twenty seventh may two thousand", "27-May-2000"),
        ("my date of birth is twenty seventh may two thousand", "27-May-2000"),
        ("27th may 2000", "27-May-2000"),
    ],
)
def test_date_of_birth_inputs_normalize_to_dd_mmm_yyyy(text, expected):
    assert normalized_dob_in(text) == expected


def test_transcript_name_and_ordinal_dob_are_extracted_cleanly():
    result = AnswerLLMProvider().interpret(
        "my name is alpha beta and my date of birth is twenty seventh "
        "may two thousand",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )

    # The introductory phrases never become part of the name, and the
    # spoken ordinal date is normalized.
    assert result.extracted_fields["patient_identifier"] == "alpha beta"
    assert result.extracted_fields["patient_age"] == "27-May-2000"


def test_unsupported_dates_are_never_normalized_or_guessed():
    assert normalized_dob_in("maybe next year") is None
    assert normalized_dob_in("31/02/2000") is None
    assert normalized_dob_in("born 1990") is None  # year only: never invented


def test_normalized_dob_appears_in_readback_and_structured_report():
    engine, workflow = _to_verified_self("q-dob-report")
    workflow.process_turn("myself")

    readback = workflow.process_turn(
        "Alpha Beta and my date of birth is twenty seven May two thousand"
    )
    assert (
        "I have your details as Alpha Beta, 27-May-2000. Is that correct?"
        in readback.response_text
    )

    confirmed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "27-May-2000"
    assert engine.to_dict()["patient"]["age"] == "27-May-2000"
    # Unconfirmed identity is never treated as a completed report.
    assert confirmed.completed is False


# ---- 7-9. confirmation and correction handling ------------------------------


def test_no_to_combined_readback_asks_only_which_field_was_wrong():
    engine, workflow = _to_combined_readback("q-correction-choice")

    result = workflow.process_turn("no")

    assert result.next_question is not None
    assert result.next_question == ["patient_identifier", DETAILS_CORRECTION_QUESTION]
    # Nothing was restarted or committed.
    assert engine.case.patient_identifier is None
    assert engine.case.patient_age == "27-May-2000"


def test_date_correction_after_no_keeps_the_name():
    engine, workflow = _to_combined_readback("q-correction-dob")
    workflow.process_turn("no")

    result = workflow.process_turn("the date is 27/05/1998")

    assert "I have your details as Alpha Beta, 27-May-1998." in result.response_text
    assert engine.case.patient_age == "27-May-1998"
    assert engine.case.patient_identifier is None  # still a candidate

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "27-May-1998"


def test_name_correction_after_no_keeps_the_date_of_birth():
    engine, workflow = _to_combined_readback("q-correction-name")
    workflow.process_turn("no")

    result = workflow.process_turn("no it is Gamma Ray")

    assert "I have your details as Gamma Ray, 27-May-2000." in result.response_text
    assert engine.case.patient_age == "27-May-2000"

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Gamma Ray"


def test_malayalam_combined_name_dob_and_denial_flow():
    # The audited browser sequence: a Malayalam combined name + spoken-DOB
    # answer, a corrected year, and Malayalam denials of the read-back. The
    # denials never overwrite the name, and the existing which-field
    # clarification asks for the corrected information.
    engine, workflow = _to_verified_self("q-ml-name-dob-reject")

    # Step 5: the reporter-role answer (the audited first utterance).
    role = workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")
    assert engine.case.reporter_role == "self"
    assert role.next_question is not None
    assert role.next_question[0] == "patient_identifier"

    combined = workflow.process_turn(
        "എൻ്റെ പേരാണ് രാജു പാണ്ഡെ പിന്നെ ജന്മതിഥി ഇരുപത് മെയ് രണ്ടായിരം"
    )
    assert engine.case.patient_age == "20-May-2000"
    assert engine.case.confirmation_candidate == {"patient_identifier": "രാജു പാണ്ഡെ"}
    assert "രാജു പാണ്ഡെ" in combined.response_text
    assert "20-May-2000" in combined.response_text

    # The corrected year (… ആറ് = 2006) replaces the candidate date and is
    # read back; the name candidate is untouched.
    corrected = workflow.process_turn("ഇരുപത് മെയ് രണ്ടായിരം ആറ്")
    assert engine.case.patient_age == "20-May-2006"
    assert "20-May-2006" in corrected.response_text
    assert engine.case.confirmation_candidate["patient_identifier"] == "രാജു പാണ്ഡെ"

    # Malayalam denials are rejections, never corrected names.
    for denial in ("അതല്ല", "അല്ല"):
        workflow.process_turn(denial)
        candidate = engine.case.confirmation_candidate["patient_identifier"]
        assert candidate == "രാജു പാണ്ഡെ", (denial, candidate)
        assert engine.case.patient_identifier is None, denial

    # "Both are wrong" reaches the existing name-or-DOB clarification without
    # storing the instruction as either field and without completing anything.
    both = workflow.process_turn("രണ്ടും സെറ്റാക്കും")
    assert both.completed is False
    assert both.human_review_required is False
    assert engine.case.confirmation_candidate["patient_identifier"] == "രാജു പാണ്ഡെ"
    assert "name" in both.response_text and "date of birth" in both.response_text


@pytest.mark.parametrize(
    "answer",
    [
        # The evidenced Malayalam self-identification variants from the
        # browser transcript: "it happened to me" phrasings that used to
        # loop the reporter-role question.
        "എനിക്കാണ് സംഭവിച്ചത്",
        "എനിക്ക് ആണ് സംഭവിച്ചത്",
        "എന്റെ തന്നെ അനുഭവമാണ്",
    ],
)
def test_malayalam_self_variants_select_the_patient_branch(answer):
    engine, workflow = _workflow("q-role-ml-self")
    workflow.process_turn("I had a stomach ache after taking this medication")
    workflow.process_turn("Lantus")
    workflow.process_turn("yes")  # Step 4

    result = workflow.process_turn(answer)

    assert engine.case.reporter_role == "self", answer
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier", answer


def test_bare_affirmative_is_never_self_identification():
    # The role question offers two alternatives; a bare yes is ambiguous and
    # is never stored as "self" (no default role is ever assigned). One
    # unclear reply offers the clarication; the bound itself has its own
    # test.
    for answer in ("അതെ", "ശരിയാണ്"):
        engine, workflow = _to_verified_self(f"q-role-bare-{answer[:4]}")
        result = workflow.process_turn(answer)
        assert engine.case.reporter_role is None, answer
        assert result.next_question is not None
        assert result.next_question[0] == "reporter_role", answer


def test_unclear_role_replies_park_for_review_instead_of_looping():
    # The reporter-role loop is bounded exactly like the other clarification
    # loops: one targeted clarification, and the second unclear reply parks
    # the case with the role unset (no default role) for the human reviewer
    # to resolve.
    engine, workflow = _to_verified_self("q-role-bound")
    workflow.process_turn("hmm")

    parked = workflow.process_turn("not sure")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.human_review_required is True
    assert parked.completed is False
    assert engine.case.reporter_role is None
    # Everything already captured is preserved.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event


def test_confirmed_name_snapshot_commits_normally():
    engine, workflow = _to_verified_self("q-ml-snapshot-commit")
    workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")
    workflow.process_turn(
        "എൻ്റെ പേരാണ് സുരേഷ് ഗോപി പിന്നെ ജന്മതിഥി ഇരുപത് മേ രണ്ടായിരത്തി ആറ്"
    )
    # While the combined read-back is pending (not parked), the held,
    # unconfirmed name stays hidden from the snapshot.
    assert workflow._collected_snapshot()["patient_identifier"] is None

    workflow.process_turn("ശരിയാണ്")

    assert engine.case.patient_identifier == "സുരേഷ് ഗോപി"
    # രണ്ടായിരത്തി ആറ് = 2006: the date decodes exactly as spoken.
    assert engine.case.patient_age == "20-May-2006"
    assert workflow._collected_snapshot()["patient_identifier"] == "സുരേഷ് ഗോപി"


def test_unclear_event_description_is_verbatim_and_review_flagged():
    # Option B: the garbled Malayalam description is preserved byte-for-byte
    # (never rewritten, translated, or clinically interpreted) and carries
    # the reviewer-facing unclear note, which changes no workflow state on
    # its own. A description matching a recognized cue is not flagged.
    engine, workflow = _workflow("q-ml-unclear-event")
    unclear = "ഈ മേഖലയില് പാര്ട്ടി നല്കി"

    first = workflow.process_turn(unclear)

    # The unclassifiable first narrative is preserved verbatim, without
    # being promoted into the event field on its own.
    assert engine.case.unclassified_initial_input == unclear
    assert engine.case.adverse_event is None
    assert first.human_review_required is False  # the note changes no state

    # When the reporter's answer to the event question is the same unclear
    # wording, it is captured verbatim and the note is set for the reviewer.
    workflow.process_turn(unclear)
    snapshot = workflow._collected_snapshot()
    assert snapshot["adverse_event"] == unclear
    assert snapshot["event_description_unclear"] is True

    # A recognized cue-bearing description is never flagged.
    clear_engine, clear_workflow = _workflow("q-clear-event")
    clear_first = clear_workflow.process_turn("I have a headache")
    clear_snapshot = clear_workflow._collected_snapshot()
    assert clear_snapshot["adverse_event"] == "I have a headache"
    assert clear_snapshot["event_description_unclear"] is False
    assert clear_first.human_review_required is False


def test_held_malayalam_name_is_visible_in_human_review_snapshot():
    # The observed flow: the captured name is held for the combined read-back
    # while the transliterated-English DOB keeps failing; the bounded
    # fallback parks the case before the read-back, and the reviewer must
    # still see the captured name (it is never committed without a "yes").
    engine, workflow = _to_verified_self("q-ml-held-name")
    workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")

    workflow.process_turn(
        "എൻ്റെ പേര് സുരേഷ് ഗോപി ജന്മതിയി ട്വന്റി ടു മേ ടു തൗസൻഡ്"
    )
    assert engine.case.confirmation_candidate == {"patient_identifier": "സുരേഷ് ഗോപി"}
    # During the normal pending DOB question the held name stays hidden:
    # neither the flat collected key nor the review-time held key exposes it.
    snapshot = workflow._collected_snapshot()
    assert snapshot["patient_identifier"] is None
    assert snapshot["patient_identifier_held"] is None

    workflow.process_turn("ട്വന്റി ടു മേ ടു തൗസൻഡ്")
    parked = workflow.process_turn("ട്വന്റി ടു മേ ടു തൗസൻഡ്")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.human_review_required is True
    assert parked.completed is False
    # The authoritative field is untouched; the held name is review-visible
    # under its own key, never presented as a confirmed value.
    assert engine.case.patient_identifier is None
    snapshot = workflow._collected_snapshot()
    assert snapshot["patient_identifier"] is None
    assert snapshot["patient_identifier_held"] == "സുരേഷ് ഗോപി"
    # The DOB bounded fallback and unanswered consent are unchanged.
    assert snapshot["patient_age"] == "unknown"
    assert snapshot["consent_to_contact"] is None


def test_malayalam_mixed_digit_dob_is_captured_and_confirmed():
    # Gnani STT commonly transcribes spoken numbers as digits: the observed
    # browser failure had the spoken Malayalam date in mixed digit form
    # ("ഇരുപത് മേയ് 2006"), which parsed no date and drove the bounded
    # fallback to park the DOB as "unknown". The mixed forms now capture,
    # read back and confirm like the all-word forms.
    engine, workflow = _to_verified_self("q-ml-mixed-dob")
    workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")
    workflow.process_turn("എൻ്റെ പേരാണ് രാജു പി കെ")

    answered = workflow.process_turn("ഇരുപത് മേയ് 2006")
    assert engine.case.patient_age == "20-May-2006"
    assert "20-May-2006" in answered.response_text
    assert engine.case.confirmation_candidate == {"patient_identifier": "രാജു പി കെ"}

    committed = workflow.process_turn("yes")
    assert engine.case.patient_age == "20-May-2006"
    assert engine.case.patient_identifier == "രാജു പി കെ"


def test_malayalam_name_selection_reaches_corrected_name_collection():
    # The observed failure: selecting പേര് ("name") in the which-field state
    # repeated the same question. The selection must move to corrected-name
    # collection, and confirmation commits the name without touching the DOB.
    engine, workflow = _to_verified_self("q-ml-name-select")
    role = workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")
    assert role.next_question is not None
    assert role.next_question[0] == "patient_identifier"

    combined = workflow.process_turn(
        "എൻ്റെ പേര് രാജു രാജുവാണ് പിന്നെ ജന്മതിഥി ഇരുപത് മെയ് രണ്ടായിരം"
    )
    assert engine.case.patient_age == "20-May-2000"
    # Only the introductory lead is stripped: the repeated-name content is
    # preserved (no unsafe automatic deletion).
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "രാജു രാജുവാണ്"
    }

    workflow.process_turn("no")
    selected = workflow.process_turn("പേര്")

    # The valid selection clears the failed-count bookkeeping and moves to
    # corrected-name collection instead of repeating the which-field question.
    assert "patient_identifier" not in workflow._confirmation_attempts
    assert selected.next_question is not None
    assert selected.next_question != [
        "patient_identifier",
        DETAILS_CORRECTION_QUESTION,
    ]

    corrected = workflow.process_turn("രാജു പാണ്ഡെ")
    assert "രാജു പാണ്ഡെ" in corrected.response_text
    assert engine.case.patient_age == "20-May-2000"  # DOB unchanged

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "രാജു പാണ്ഡെ"
    assert engine.case.patient_age == "20-May-2000"


def test_malayalam_dob_selection_reaches_corrected_date_collection():
    engine, workflow = _to_verified_self("q-ml-dob-select")
    workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")
    workflow.process_turn(
        "എൻ്റെ പേര് രാജു രാജുവാണ് പിന്നെ ജന്മതിഥി ഇരുപത് മെയ് രണ്ടായിരം"
    )
    workflow.process_turn("no")

    selected = workflow.process_turn("ജന്മതിഥി")
    # The DOB selection reaches corrected-date collection.
    assert "patient_identifier" not in workflow._confirmation_attempts
    assert selected.next_question is not None
    assert selected.next_question != [
        "patient_identifier",
        DETAILS_CORRECTION_QUESTION,
    ]

    corrected = workflow.process_turn("27/05/1998")
    assert "27-May-1998" in corrected.response_text
    assert engine.case.confirmation_candidate["patient_identifier"] == "രാജു രാജുവാണ്"

    committed = workflow.process_turn("yes")
    assert engine.case.patient_age == "27-May-1998"
    assert engine.case.patient_identifier == "രാജു രാജുവാണ്"


# (name-choice, dob-choice) replies per locale in the which-field state. All
# labels are the exact DOB/name phrasing each locale's own flow uses.
_FIELD_CHOICE_LANGUAGE_PHRASES = {
    "en-IN": ("the name", "date of birth"),
    "hi-IN": ("नाम", "जन्म तिथि"),
    "bn-IN": ("নাম", "জন্ম তারিখ"),
    "gu-IN": ("નામ", "જન્મ તારીખ"),
    "kn-IN": ("ಹೆಸರು", "ಜನ್ಮ ದಿನಾಂಕ"),
    "ml-IN": ("പേര്", "ജന്മതിഥി"),
    "mr-IN": ("नाव", "जन्मतारीख"),
    "pa-IN": ("ਨਾਮ", "ਜਨਮ ਤਾਰੀਖ"),
    "ta-IN": ("பெயர்", "பிறந்த தேதி"),
    "te-IN": ("పేరు", "పుట్టిన తేదీ"),
}


@pytest.mark.parametrize("language", sorted(_FIELD_CHOICE_LANGUAGE_PHRASES))
def test_field_selection_in_every_locale_leaves_the_choice_question(language):
    name_choice, dob_choice = _FIELD_CHOICE_LANGUAGE_PHRASES[language]
    for choice in (name_choice, dob_choice):
        engine = AEStateEngine.create_case("q-choice-" + language)
        engine.update("product", "Lantus SoloStar")
        engine.update("adverse_event", "headache")
        engine.update("reporter_role", "self")
        engine.update("patient_age", "20-May-2000")
        engine.begin_confirmation("patient_identifier", "Alpha Beta")
        workflow = ConversationalAEWorkflow(
            engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
        )
        workflow.process_turn("no")

        selected = workflow.process_turn(choice)

        # A valid field selection moves the flow on: the which-field question
        # is not repeated and the instruction is never stored as a value.
        assert selected.next_question != [
            "patient_identifier",
            DETAILS_CORRECTION_QUESTION,
        ], (language, selected.next_question)
        assert selected.completed is False
        assert engine.case.patient_identifier is None


@pytest.mark.parametrize("language", sorted(_CONFIRM_LANGUAGE_PHRASES))
def test_negative_reply_in_every_locale_never_becomes_the_name(language):
    _, negative = _CONFIRM_LANGUAGE_PHRASES[language]
    engine = AEStateEngine.create_case("q-name-reject-" + language)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    engine.update("reporter_role", "self")
    engine.update("patient_age", "20-May-2000")
    engine.begin_confirmation("patient_identifier", "Alpha Beta")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )

    workflow.process_turn(negative)

    # The denial is a rejection, never a corrected name; the read-back re-asks
    # for the name (existing correction path).
    assert engine.case.patient_identifier is None
    assert engine.case.pending_confirmation == "patient_identifier"
    assert engine.case.confirmation_candidate["patient_identifier"] == "Alpha Beta"


def test_repeated_date_of_birth_never_corrupts_the_name_or_completes():
    engine, workflow = _to_combined_readback("q-dob-repeat")

    result = workflow.process_turn("27/05/2003")

    # The date is updated/re-read; the name candidate is untouched, and no
    # report is produced from an unconfirmed identity.
    assert engine.case.patient_identifier is None
    assert engine.case.patient_age == "27-May-2003"
    assert "I have your details as Alpha Beta, 27-May-2003." in result.response_text
    assert result.completed is False
    assert engine.can_complete() is False

    repeated = workflow.process_turn("27/05/2003")  # same date again
    assert engine.case.patient_identifier is None
    assert repeated.completed is False


# ---- 10-11. consent and report integrity -----------------------------------


def test_unanswered_consent_is_never_recorded_as_no():
    engine = AEStateEngine.create_case("q-consent-none")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Beta")
    engine.update("patient_age", "27-May-2000")
    engine.update("contact_phone", "9999990000")

    snapshot = engine.to_dict()
    assert snapshot["contact"]["consent_to_contact"] is None
    assert "consent_to_contact" in engine.missing_information()
    assert engine.can_complete() is False


def test_unconfirmed_identity_blocks_completion_and_never_completes_silently():
    engine, workflow = _to_combined_readback("q-unconfirmed")

    engine.update("contact_phone", "9999990000")
    engine.update("consent_to_contact", True)
    engine.update("seriousness", "no")
    engine.update("onset_date", "yesterday")
    engine.update("outcome", "recovering")

    # Identity is still only a candidate: nothing can complete.
    assert engine.case.pending_confirmation == "patient_identifier"
    assert engine.can_complete() is False

    first = workflow.process_turn("yes")
    assert first.completed is True
    assert engine.case.patient_identifier == "Alpha Beta"

    # A case that never confirms stays incomplete under the bounded fallback.
    engine2 = AEStateEngine.create_case("q-unconfirmed-2")
    engine2.update("product", "Lantus SoloStar")
    engine2.update("adverse_event", "rash")
    engine2.update("reporter_role", "self")
    engine2.update("patient_age", "27-May-2000")
    engine2.begin_confirmation("patient_identifier", "Alpha Beta")
    workflow2 = ConversationalAEWorkflow(
        engine2, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    engine2.update("contact_phone", "9999990000")
    engine2.update("consent_to_contact", True)
    engine2.update("seriousness", "no")
    engine2.update("onset_date", "yesterday")
    engine2.update("outcome", "recovering")
    assert engine2.can_complete() is False


# ---- multiple reported events ----------------------------------------------


def test_stomach_ache_description_is_captured_as_the_event():
    engine, workflow = _workflow("q-ache")

    result = workflow.process_turn("I have a severe stomach ache since yesterday")

    assert engine.case.adverse_event == "I have a severe stomach ache since yesterday"
    assert result.next_question is not None


def test_second_symptom_is_merged_and_never_replaces_the_original():
    engine, workflow = _workflow("q-events-both")
    first = "I have a severe stomach ache since yesterday"
    second = "and I also have a severe headache"

    workflow.process_turn(first)
    assert engine.case.adverse_event == first

    workflow.process_turn(second)

    # Both distinct events are preserved verbatim, in order; nothing is lost.
    assert engine.case.adverse_event == f"{first}; {second}"


def test_correction_like_second_description_is_clarified_before_any_merge():
    engine, workflow = _workflow("q-events-correction")
    original = "I have a severe stomach ache since yesterday"

    workflow.process_turn(original)
    result = workflow.process_turn("actually it is only a headache")

    # A correction-like statement is clarified, never silently combined.
    assert "was that another symptom, or a correction" in result.response_text
    assert engine.case.adverse_event == original

    # After the clarification the wording is never lost and the original is
    # never overwritten.
    workflow.process_turn("actually it is only a headache")
    assert engine.case.adverse_event == f"{original}; actually it is only a headache"


# ---- provider failure session integrity ------------------------------------


class _ScriptSTT:
    """Replays transcripts, then fails like the live 403 seen in the browser."""

    def __init__(self, transcripts: list[str], fail_status: int | None = None):
        self._transcripts = list(transcripts)
        self._fail_status = fail_status

    def transcribe(self, audio_path, language_code="en-IN"):
        from app.services.prisma import PrismaAPIError

        if self._transcripts:
            return SimpleNamespace(text=self._transcripts.pop(0))
        if self._fail_status is not None:
            raise PrismaAPIError(
                "Access denied. Check the API key and the account's available "
                "credits.",
                status_code=self._fail_status,
            )
        return SimpleNamespace(text="")


class _WavTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\x00" * 40)
        return path


class _FailTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        from app.services.timbre import TimbreAPIError

        raise TimbreAPIError("Gnani TTS is unavailable", status_code=503)


def _voice_client(stt, tts, store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.voice_conversation import create_voice_conversation_router

    router = create_voice_conversation_router(
        stt=stt,
        tts=tts,
        session_store=store,
        llm_provider=AnswerLLMProvider(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, session_id):
    resp = client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": "en-IN"},
    )
    return resp


def test_stt_provider_failure_does_not_corrupt_the_session():
    from app.config import get_gnani_api_key
    from app.session import SessionStore

    store = SessionStore()
    stt = _ScriptSTT(["I took Lantus and have a headache"], fail_status=403)
    client = _voice_client(stt, _WavTTS(), store)
    session_id = "prov-fail-stt"

    first = _post(client, session_id)
    assert first.status_code == 200
    captured = store.get_or_create(session_id).ae_engine.case.adverse_event
    assert captured

    failed = _post(client, session_id)
    assert failed.status_code == 500
    detail = failed.json()["detail"]
    assert "Access denied" in detail
    # No secret is ever echoed back to the caller.
    key = get_gnani_api_key()
    assert not key or key not in detail

    # The captured case is untouched by the provider failure.
    case = store.get_or_create(session_id).ae_engine.case
    assert case.adverse_event == captured


def test_stt_403_error_preserves_session_and_the_next_turn_proceeds(caplog):
    import logging

    from app.config import get_gnani_api_key
    from types import SimpleNamespace

    from app.services.prisma import PrismaAPIError
    from app.session import SessionStore

    class Flaky403STT:
        """Fails the first request with the observed HTTP 403, then recovers."""

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path, language_code="en-IN"):
            self.calls += 1
            if self.calls == 1:
                # A realistic Gnani 403 body: the provider's own denial reason
                # (distinct from MedVaani's static patient-facing hint) with an
                # error type, as parsed onto PrismaAPIError by _parse_response.
                raise PrismaAPIError(
                    "STT access is denied for this API key for the requested "
                    "language",
                    status_code=403,
                    error_type="AUTHORIZATION_FAILED",
                )
            return SimpleNamespace(
                text="I took Lantus and have a headache",
                detected_language=None,
                timestamp=None,
            )

    store = SessionStore()
    stt = Flaky403STT()
    client = _voice_client(stt, _WavTTS(), store)
    session_id = "prov-403-retry"

    with caplog.at_level(logging.WARNING, logger="app.api.voice_conversation"):
        failed = _post(client, session_id)
    assert failed.status_code == 500
    detail = failed.json()["detail"]
    # The error is reported as an authentication/credits problem, never as a
    # language-recognition failure, and no secret is echoed back.
    assert "Access denied" in detail
    key = get_gnani_api_key()
    assert not key or key not in detail

    # Operator-side diagnostics: the log carries the provider's own denial
    # reason (type and message), the failing status and the STT language, so
    # an intermittent 403 can be diagnosed without a live API call.
    assert "HTTP 403" in caplog.text
    assert "error_type=AUTHORIZATION_FAILED" in caplog.text
    assert (
        "STT access is denied for this API key for the requested language"
        in caplog.text
    )
    assert "language=en-IN" in caplog.text
    # Privacy: the API key is never logged either.
    assert not key or key not in caplog.text

    # The failed turn applied nothing and left no pending confirmation.
    case = store.get_or_create(session_id).ae_engine.case
    assert not case.adverse_event
    assert case.pending_confirmation is None

    # A subsequent valid turn proceeds normally through the approved flow.
    recovered = _post(client, session_id)
    assert recovered.status_code == 200
    body = recovered.json()
    assert body["next_question"] is not None
    assert body["next_question"][0] == "product_verification"
    assert (
        store.get_or_create(session_id).ae_engine.case.adverse_event
        == "I took Lantus and have a headache"
    )
    assert stt.calls == 2


def test_tts_provider_failure_does_not_reset_the_workflow():
    from app.session import SessionStore

    store = SessionStore()
    client = _voice_client(
        _ScriptSTT(["I took Lantus and have a headache"]), _FailTTS(), store
    )
    session_id = "prov-fail-tts"

    failed = _post(client, session_id)
    assert failed.status_code == 502
    assert "Gnani service failure" in failed.json()["detail"]

    # The workflow ran before synthesis: the report data is kept, not reset.
    case = store.get_or_create(session_id).ae_engine.case
    assert case.adverse_event == "I took Lantus and have a headache"


# ---- identity confirmation: spoken decades and the field-correction loop ----


def test_name_and_decade_year_dob_are_extracted_cleanly():
    result = AnswerLLMProvider().interpret(
        "my name is alpha beta and my date of birth is twenty seven june "
        "nineteen ninety eight",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )

    assert result.extracted_fields["patient_identifier"] == "alpha beta"
    assert result.extracted_fields["patient_age"] == "27-Jun-1998"


@pytest.mark.parametrize(
    "text",
    [
        "twenty seventh june nineteen ninety eight",
        "27th june nineteen ninety eight",
        "27 june nineteen ninety eight",
    ],
)
def test_ordinal_and_cardinal_days_parse_to_the_same_date(text):
    assert normalized_dob_in(text) == "27-Jun-1998"


def test_year_only_answer_is_never_stored():
    result = AnswerLLMProvider().interpret(
        "nineteen ninety eight",
        {"next_field": ["patient_age", "Could you please tell me your date of birth?"]},
    )

    # A partial year stays incomplete: nothing is guessed or stored.
    assert "patient_age" not in result.extracted_fields
    assert "dob_age_conflict" not in result.extracted_fields


def test_bare_three_digit_fragment_is_never_stored_as_dob_or_age():
    age_question = {
        "next_field": ["patient_age", "Could you please tell me your date of birth?"]
    }
    # Mis-heard date fragments must never become an age or a DOB.
    assert "patient_age" not in AnswerLLMProvider().interpret("117", age_question).extracted_fields

    engine, workflow = _workflow("q-117")
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Beta")  # name; DOB still outstanding
    fragment = workflow.process_turn("117")

    assert engine.case.patient_age is None
    # The name stays the existing candidate throughout.
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Beta"}
    assert fragment.next_question is not None
    assert fragment.next_question[0] == "patient_age"


def test_dob_choice_updates_only_the_date_and_keeps_the_name():
    engine, workflow = _to_combined_readback("q-dob-choice")
    workflow.process_turn("no")

    asked = workflow.process_turn("date of birth")
    # The literal phrase is never stored as a value; only the date is asked.
    assert asked.next_question is not None
    assert asked.next_question[0] == "patient_age"
    assert engine.case.patient_age == "27-May-2000"

    corrected = workflow.process_turn("twenty seventh june nineteen ninety eight")
    assert "I have your details as Alpha Beta, 27-Jun-1998." in (
        corrected.response_text
    )
    # The name stays the existing candidate; only the date changed.
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Beta"}

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "27-Jun-1998"


def test_incomplete_dob_on_the_target_turn_reasks_without_restarting():
    engine, workflow = _to_combined_readback("q-dob-incomplete")
    workflow.process_turn("no")
    workflow.process_turn("date of birth")

    incomplete = workflow.process_turn("nineteen ninety eight")

    assert engine.case.patient_age == "27-May-2000"  # unchanged, never 117/None
    # The name stays the existing candidate throughout.
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Beta"}
    assert "full date of birth" in incomplete.response_text
    assert incomplete.next_question is not None
    assert incomplete.next_question[0] == "patient_age"
    assert incomplete.completed is False


def test_name_choice_updates_only_the_name_and_keeps_the_date():
    engine, workflow = _to_combined_readback("q-name-choice")
    workflow.process_turn("no")

    asked = workflow.process_turn("the name")
    assert asked.next_question is not None
    assert asked.next_question[0] == "patient_identifier"

    corrected = workflow.process_turn("Gamma Ray")
    assert "I have your details as Gamma Ray, 27-May-2000." in (
        corrected.response_text
    )
    assert engine.case.patient_age == "27-May-2000"

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Gamma Ray"
    assert engine.case.patient_age == "27-May-2000"


@pytest.mark.parametrize(
    "answer",
    ["yes", "yes, you got it correct", "that's correct"],
)
def test_clear_affirmatives_confirm_a_pending_read_back(answer):
    engine, workflow = _to_combined_readback(f"q-affirm-{answer[:4]}")

    result = workflow.process_turn(answer)

    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.pending_confirmation is None
    assert result.next_question is not None
    assert result.next_question[0] == "contact_phone"


# Natural "that's correct" affirmatives per locale, exact standalone replies
# only. The Malayalam ശരിയാണ് is the observed browser reply that used to
# repeat the confirmation prompt; അതേ is its companion spelling, and
# "അത് ശരിയാണ്"/"അത് ശരി" are the pronoun-led echoes of the read-back
# question (the phone-confirmation loop repro). సరి ("correct") parallels
# the already-supported ಸರಿ.
_NATURAL_AFFIRMATIVES = {
    "en-IN": "that's correct",
    "hi-IN": "बिल्कुल सही",
    "bn-IN": "ঠিক আছে",
    "gu-IN": "સાચું",
    "kn-IN": "ಸರಿ",
    "ml-IN": ["ശരിയാണ്", "അതേ", "അത് ശരിയാണ്", "അത് ശരി", "യെസ്", "യസ്"],
    "mr-IN": "बरोबर",
    "pa-IN": "ਠੀਕ ਹੈ",
    "ta-IN": "சரி",
    "te-IN": "సరి",
}


@pytest.mark.parametrize(
    ("locale", "affirmatives"),
    sorted(_NATURAL_AFFIRMATIVES.items()),
    ids=str,
)
def test_natural_affirmatives_in_every_locale_confirm_the_read_back(
    locale, affirmatives
):
    for affirmative in affirmatives if isinstance(affirmatives, list) else [affirmatives]:
        engine, workflow = _to_combined_readback(f"q-affirm-{locale}-{affirmative}")

        result = workflow.process_turn(affirmative)

        # A valid affirmative confirms the pending value and advances the
        # workflow - the confirmation prompt is never repeated.
        assert engine.case.patient_identifier == "Alpha Beta", affirmative
        assert engine.case.pending_confirmation is None, affirmative
        assert result.next_question is not None, affirmative
        assert result.next_question[0] == "contact_phone", affirmative


def test_malayalam_shariyaanu_confirms_the_observed_contaminated_flow():
    # The observed browser sequence: the combined answer kept the DOB-label
    # variant ജന്മതിയി on the name, and the ശരിയാണ് reply then repeated the
    # confirmation prompt (only അതെ was accepted). Both are fixed: the name
    # candidate is clean and ശരിയാണ് confirms instead of re-asking.
    engine, workflow = _to_verified_self("q-ml-shariyaanu")
    workflow.process_turn("എന്റെ സ്വന്തം അനുഭവമാണ്")

    combined = workflow.process_turn(
        "എൻ്റെ പേരാണ് രാജു പി കെ പിന്നെ ജന്മതിയി ഇരുപത് മേ രണ്ടായിരത്തി ആറ്"
    )
    assert engine.case.confirmation_candidate == {"patient_identifier": "രാജു പി കെ"}
    assert engine.case.patient_age == "20-May-2006"
    assert "രാജു പി കെ" in combined.response_text
    assert "20-May-2006" in combined.response_text

    confirmed = workflow.process_turn("ശരിയാണ്")
    assert engine.case.patient_identifier == "രാജു പി കെ"
    assert engine.case.patient_age == "20-May-2006"
    assert engine.case.pending_confirmation is None
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "contact_phone"


def test_ambiguous_and_negative_replies_never_confirm_the_read_back():
    # An unrelated script-word reply and a negation are never treated as
    # affirmative confirmations.
    engine, workflow = _to_combined_readback("q-affirm-negative")

    ambiguous = workflow.process_turn("ശരിയാണോ എന്ന് എനിക്കറിയില്ല")
    assert engine.case.pending_confirmation == "patient_identifier"
    assert ambiguous.next_question is not None
    assert ambiguous.next_question[0] == "confirmation"

    negative = workflow.process_turn("ഇല്ല")
    assert engine.case.patient_identifier is None
    assert negative.next_question is not None
    assert negative.next_question[0] == "patient_identifier"


def test_okke_affirms_the_combined_readback_without_name_corruption():
    # The observed corruption: ഓക്കെ ("ok") used to be classified as a
    # corrected value and REPLACED the held name; it is an exact affirmative
    # and must never become a name candidate.
    engine, workflow = _to_verified_self("q-ml-okke")
    workflow.process_turn("myself")
    workflow.process_turn("രാജു കെ ജന്മതിഥി ഇരുപത് മേയ് രണ്ടായിരം")
    cb = workflow.process_turn("ഓക്കെ")

    assert engine.case.confirmation_candidate == {}
    assert engine.case.pending_confirmation is None
    assert engine.case.patient_identifier == "രാജു കെ"
    assert engine.case.patient_age == "20-May-2000"
    assert cb.next_question is not None
    assert cb.next_question[0] == "contact_phone"


def test_dob_correction_after_rejection_keeps_the_name():
    # The observed sequence: rejection ("അല്ല"), the ജന്മതിഥി selection, the
    # DOB re-entry, and the final confirmation must preserve രാജു കെ and
    # commit the corrected details.
    engine, workflow = _to_verified_self("q-ml-okke-dob")
    workflow.process_turn("myself")
    workflow.process_turn("രാജു കെ ജന്മതിഥി ഇരുപത് മേയ് രണ്ടായിരം")
    workflow.process_turn("അല്ല")
    workflow.process_turn("ജന്മതിഥി")
    workflow.process_turn("ഇരുപത് മേയ് രണ്ടായിരം")
    assert engine.case.confirmation_candidate["patient_identifier"] == "രാജു കെ"
    committed = workflow.process_turn("അതെ")
    assert engine.case.patient_identifier == "രാജു കെ"
    assert engine.case.patient_age == "20-May-2000"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_aa_affirmatives_drive_the_email_flow():
    # "ആ അതെ" affirms availability; "ആ ശരിയാണ്" confirms the dictated
    # read-back and commits; the committed email is retained in the report
    # snapshot and consent is untouched.
    from tests.test_ae_email_flow import _to_email_offer

    engine, workflow = _to_email_offer("q-ml-aa-email")

    offered = workflow.process_turn("ആ അതെ")
    assert workflow._email_request_pending is True
    assert offered.next_question is not None

    workflow.process_turn(
        "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജി മെയിൽ ഡോട്ട് കോം"
    )
    confirmed = workflow.process_turn("ആ ശരിയാണ്")

    assert engine.case.contact_email == "sreehari@gmail.com"
    assert engine.case.consent_to_contact is True
    assert engine.case.pending_confirmation is None
    assert workflow._collected_snapshot()["contact_email"] == "sreehari@gmail.com"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_garbled_single_token_never_replaces_a_held_name():
    # Script-aware mumble guard: an unknown single Malayalam token answering
    # a held name read-back stays ambiguous and never replaces the candidate;
    # legitimate multi-word corrections still replace it.
    engine, workflow = _to_combined_readback("q-ml-mumble")
    garbled = workflow.process_turn("പ്രിയ")
    assert engine.case.confirmation_candidate["patient_identifier"] == "Alpha Beta"
    assert garbled.next_question is not None
    assert garbled.next_question[0] == "confirmation"

    corrected = workflow.process_turn("എന്റെ പേര് രാജു ഗോപി")
    assert (
        engine.case.confirmation_candidate["patient_identifier"]
        == "എന്റെ പേര് രാജു ഗോപി"
    )


def test_clear_affirmative_confirms_the_medication_verification():
    engine, workflow = _workflow("q-affirm-verify")
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")

    result = workflow.process_turn("yes, you got it correct")

    assert result.next_question is not None
    assert result.next_question[0] == "reporter_role"
    assert engine.case.product == "Lantus SoloStar"


def test_repeated_ambiguity_on_the_correction_choice_escalates_to_review():
    engine, workflow = _to_combined_readback("q-choice-bound")
    workflow.process_turn("no")

    workflow.process_turn("hmm")  # ambiguous choice
    workflow.process_turn("not sure")
    final = workflow.process_turn("maybe")

    assert final.human_review_required is True
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert final.completed is False
    # Nothing was discarded or invented.
    assert engine.case.patient_identifier is None
    assert engine.case.patient_age == "27-May-2000"


def test_reporter_branch_dob_correction_keeps_the_patient_name():
    engine = AEStateEngine.create_case("q-reporter-dob-choice")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "someone_else")
    engine.update("patient_age", "12-Mar-1975")
    engine.begin_confirmation("patient_identifier", "Mary Jones")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )

    workflow.process_turn("no")
    workflow.process_turn("date of birth")
    corrected = workflow.process_turn("12/03/1980")

    assert "Mary Jones, 12-Mar-1980." in corrected.response_text
    assert engine.case.patient_age == "12-Mar-1980"

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Mary Jones"
    assert engine.case.patient_age == "12-Mar-1980"


# ---- correction keeps canonical values only (no stale raw DOB) --------------


def _capture_name_and_dob(case_id: str, phrase: str):
    """Workflow whose pending combined read-back carries the given answer."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")
    readback = workflow.process_turn(phrase)
    return engine, workflow, readback


@pytest.mark.parametrize(
    "text,expected",
    [
        ("twenty eighth January twenty twenty", "28-Jan-2020"),
        ("twentieth January two thousand twenty", "20-Jan-2020"),
        ("twenty January two thousand twenty", "20-Jan-2020"),
    ],
)
def test_irregular_ordinal_spoken_dobs_normalize(text, expected):
    assert normalized_dob_in(text) == expected


def test_combined_name_and_irregular_ordinal_dob_extract_cleanly():
    result = AnswerLLMProvider().interpret(
        "alpha beta and my date of birth is twenty eighth January twenty twenty",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )

    # Canonical values only: the raw DOB speech never joins the name.
    assert result.extracted_fields["patient_identifier"] == "alpha beta"
    assert result.extracted_fields["patient_age"] == "28-Jan-2020"


def test_dob_correction_replaces_the_candidate_with_canonical_values():
    engine, workflow, readback = _capture_name_and_dob(
        "q-dob-replace",
        "Alpha Beta and my date of birth is twenty eighth January twenty twenty",
    )
    # The confirmation carries canonical values only.
    assert "I have your details as Alpha Beta, 28-Jan-2020." in readback.response_text
    assert "twenty eighth" not in readback.response_text

    workflow.process_turn("no")
    asked = workflow.process_turn("the date of birth was incorrect")
    assert asked.next_question is not None
    assert asked.next_question[0] == "patient_age"

    corrected = workflow.process_turn("twentieth January two thousand twenty")
    # Canonical values only: the old raw phrase and the old date are gone.
    assert "I have your details as Alpha Beta, 20-Jan-2020." in (
        corrected.response_text
    )
    assert "twenty eighth" not in corrected.response_text
    assert "twenty twenty" not in corrected.response_text
    assert engine.case.patient_age == "20-Jan-2020"
    # The name candidate is untouched by the date correction.
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Beta"}

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "20-Jan-2020"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_rejecting_the_corrected_dob_reopens_the_correction_flow():
    engine, workflow, _ = _capture_name_and_dob(
        "q-dob-reject-again", "alpha beta and my date of birth is twenty eighth January twenty twenty"
    )
    workflow.process_turn("no")
    workflow.process_turn("the date of birth was incorrect")
    corrected = workflow.process_turn("twentieth January two thousand twenty")
    assert "20-Jan-2020" in corrected.response_text

    reopened = workflow.process_turn("no")
    assert reopened.next_question is not None
    assert "was it the name or the date of birth" in reopened.response_text.lower()
    # Nothing was confirmed by the rejection.
    assert engine.case.patient_identifier is None
    assert engine.case.patient_age == "20-Jan-2020"


def test_invalid_dob_never_restores_or_confirms_a_stale_candidate():
    engine, workflow, _ = _capture_name_and_dob(
        "q-dob-invalid", "alpha beta and my date of birth is twenty eighth January twenty twenty"
    )
    workflow.process_turn("no")
    workflow.process_turn("the date of birth was incorrect")

    invalid = workflow.process_turn("maybe next year")

    # The incomplete answer is asked again; nothing is confirmed or restored
    # as a successful correction.
    assert "full date of birth" in invalid.response_text
    assert engine.case.patient_age == "28-Jan-2020"
    assert engine.case.patient_identifier is None
    assert engine.case.pending_confirmation == "patient_identifier"
    assert invalid.completed is False
    assert invalid.next_question is not None
    assert invalid.next_question[0] == "patient_age"


# ---- combined name + typed DD-MMM-YYYY date of birth ------------------------


def test_combined_name_and_ddmmyyyy_dob_extract_in_one_turn():
    result = AnswerLLMProvider().interpret(
        "Alpha Beta 21-Mar-2008",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )

    assert result.extracted_fields["patient_identifier"] == "Alpha Beta"
    assert result.extracted_fields["patient_age"] == "21-Mar-2008"


def _to_patient_details(case_id: str):
    """Workflow with the combined patient-details question on the table."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I had a stomach ache after taking this medication")
    workflow.process_turn("Lantus")
    workflow.process_turn("yes")  # Step 4
    workflow.process_turn("myself")  # Step 5
    return engine, workflow


def test_combined_confirmation_without_a_second_dob_question():
    engine, workflow = _to_patient_details("q-typed-dob")

    confirmed_turn = workflow.process_turn("Alpha Beta 21-Mar-2008")

    # Both values are captured in one turn: the combined read-back follows
    # immediately - the date of birth is never asked for twice.
    assert confirmed_turn.next_question is not None
    assert confirmed_turn.next_question[0] == "confirmation"
    assert (
        "I have your details as Alpha Beta, 21-Mar-2008. Is that correct?"
        in confirmed_turn.response_text
    )
    assert "Could you please tell me your date of birth?" not in (
        confirmed_turn.response_text
    )

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "21-Mar-2008"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_combined_english_spoken_name_and_dob_in_one_turn():
    # The observed failure: the spoken-English date embedded after the
    # lead-in ("… is twenty May two thousand") parsed as no date at all, so
    # the DOB question was asked again and the whole sentence was held as
    # the name candidate. The overlapping-candidate search fixes both.
    engine, workflow = _to_patient_details("q-spoken-english-dob")

    confirmed_turn = workflow.process_turn(
        "My name is Raju PK and my date of birth is twenty May two thousand"
    )

    assert engine.case.patient_age == "20-May-2000"
    assert engine.case.confirmation_candidate == {"patient_identifier": "Raju PK"}
    assert confirmed_turn.next_question is not None
    assert confirmed_turn.next_question[0] == "confirmation"
    assert (
        "I have your details as Raju PK, 20-May-2000. Is that correct?"
        in confirmed_turn.response_text
    )

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Raju PK"
    assert engine.case.patient_age == "20-May-2000"


def test_typed_dob_persists_across_later_turns():
    engine, workflow = _to_patient_details("q-typed-dob-persist")
    workflow.process_turn("Alpha Beta 21-Mar-2008")

    # An unexpected turn neither wipes the date nor fabricates "unknown",
    # and the name candidate is untouched.
    clarification = workflow.process_turn("hmm")

    assert engine.case.patient_age == "21-Mar-2008"
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Beta"}
    assert engine.case.patient_age != "unknown"
    assert clarification.completed is False

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Beta"
    assert engine.case.patient_age == "21-Mar-2008"


def test_no_age_question_is_ever_offered_for_the_patient_details():
    engine, workflow = _to_patient_details("q-no-age-question")

    asked = workflow.process_turn("myself")

    question = asked.next_question[1]
    # The approved flow collects a date of birth, never an age.
    assert "date of birth" in question
    assert "How old" not in question
    assert "age in years" not in question


def test_report_requires_every_mandatory_step_and_the_consent_decision():
    engine, workflow = _to_patient_details("q-completion-gate")
    workflow.process_turn("Alpha Beta 21-Mar-2008")
    workflow.process_turn("yes")

    # Identity captured but not everything: no report yet.
    assert workflow.process_turn("9886094321").completed is False
    assert "consent_to_contact" in engine.missing_information()
    assert "seriousness" in engine.missing_information()

    workflow.process_turn("yes")  # contact confirmed
    assert workflow.process_turn("Yes, I consent").completed is False
    assert workflow.process_turn("I don't have an email address").completed is False
    assert workflow.process_turn("No, it was not severe").completed is False
    assert workflow.process_turn("It started yesterday").completed is False
    final = workflow.process_turn("no")  # additional information declined
    assert final.completed is True


def test_final_report_keeps_the_typed_dob_and_verbatim_event():
    engine, workflow = _to_patient_details("q-final-dob-report")
    workflow.process_turn("Alpha Beta 21-Mar-2008")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    workflow.process_turn("Yes, I consent")
    workflow.process_turn("I don't have an email address")
    workflow.process_turn("No, it was not severe")
    workflow.process_turn("It started yesterday")
    final = workflow.process_turn("no")

    assert final.completed is True
    collected = final.collected_information
    assert collected["adverse_event"] == (
        "I had a stomach ache after taking this medication"
    )
    assert collected["patient_identifier"] == "Alpha Beta"
    assert collected["patient_age"] == "21-Mar-2008"
    assert collected["patient_age"] != "unknown"
    snapshot = engine.to_dict()
    assert snapshot["patient"]["age"] == "21-Mar-2008"
    assert snapshot["report_complete"] is True


# ---- multilingual medication recognition (observed browser failures) --------


def test_malayalam_lantus_alias_captures_product_and_asks_verification():
    # The observed failure: "ഞാൻ എടുത്ത മരുന്നാണ് ലാൻഡസ് കൊതുക" terminated
    # the Lantus workflow as an unrelated medicine. The alias must capture the
    # canonical product, keep the description verbatim, and ask the existing
    # Step-4 verification instead.
    engine, workflow = _workflow("q-malayalam-lantus")
    result = workflow.process_turn("ഞാൻ എടുത്ത മരുന്നാണ് ലാൻഡസ് കൊതുക")

    assert engine.case.product == "Lantus SoloStar"
    # The original adverse-event description is preserved verbatim.
    assert engine.case.adverse_event == "ഞാൻ എടുത്ത മരുന്നാണ് ലാൻഡസ് കൊതുക"
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    assert "Lantus SoloStar?" in result.response_text

    verified = workflow.process_turn("yes")
    assert engine.case.product == "Lantus SoloStar"
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role"


def test_latest_malayalam_transcript_canonicalizes_and_asks_verification():
    # The latest browser failure: the ട/ല spellings ("ലാൻടസ് സോലോ സ്റ്റാർ")
    # had no aliases, so the workflow terminated out-of-scope instead of
    # asking the Step-4 confirmation question.
    engine, workflow = _workflow("q-malayalam-lantus-latest")
    result = workflow.process_turn(
        "ഞാനെടുത്ത മരുന്നിൻ്റെ പേര് ആണ് ലാൻടസ് സോലോ സ്റ്റാർ"
    )

    # The canonical product is captured and the description stays verbatim.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == (
        "ഞാനെടുത്ത മരുന്നിൻ്റെ പേര് ആണ് ലാൻടസ് സോലോ സ്റ്റാർ"
    )
    # The existing Step-4 confirmation is asked, never an immediate termination.
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    assert "Lantus SoloStar?" in result.response_text

    verified = workflow.process_turn("yes")
    assert engine.case.product == "Lantus SoloStar"
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role"


def test_latest_malayalam_transcript_denial_and_out_of_scope_paths():
    engine, workflow = _workflow("q-malayalam-lantus-denied")
    workflow.process_turn("ഞാനെടുത്ത മരുന്നിൻ്റെ പേര് ആണ് ലാൻടസ് സോലോ സ്റ്റാർ")

    # A denial follows the existing correction/re-ask path...
    denied = workflow.process_turn("no")
    assert denied.next_question is not None
    assert denied.next_question[0] == "product"

    # ...and naming a genuinely different medicine keeps the existing
    # out-of-scope termination with the verbatim description preserved.
    unrelated = workflow.process_turn("paracetamol")
    assert "Lantus SoloStar only" in unrelated.response_text
    assert "cannot continue this report" in unrelated.response_text
    assert unrelated.next_question is None
    assert "ലാൻടസ് സോലോ സ്റ്റാർ" in engine.case.adverse_event


def test_unclear_malayalam_medication_reply_is_clarified_not_terminated():
    # The observed failure: an unclear Malayalam reply to the medication
    # question was captured as a raw product name and terminated the report
    # out-of-scope. Unclear speech must trigger the bounded clarification
    # instead — the product is never fabricated.
    engine, workflow = _workflow("q-ml-unclear-med")
    asked = workflow.process_turn("I have a headache")
    assert asked.next_question is not None
    assert asked.next_question[0] == "product"

    result = workflow.process_turn("എഴുമല്ലേ പിന്നെ അങ്ങനെ ലഭിക്കാന് കഴിയുന്നു")

    assert not engine.case.product
    assert result.next_question is not None
    assert result.next_question[0] == "product"
    assert "Lantus SoloStar only" not in result.response_text
    assert "cannot continue this report" not in result.response_text
    # The adverse-event description stays verbatim.
    assert engine.case.adverse_event == "I have a headache"


def test_repeated_unclear_malayalam_replies_park_for_review_not_terminate():
    engine, workflow = _workflow("q-ml-unclear-bounded")
    workflow.process_turn("I have a headache")
    unclear = "എഴുമല്ലേ പിന്നെ അങ്ങനെ ലഭിക്കാന് കഴിയുന്നു"
    workflow.process_turn(unclear)
    parked = workflow.process_turn(unclear)

    # The bounded safeguard parks the case for human review with the product
    # left empty; the out-of-scope termination never fires for unclear speech.
    assert not engine.case.product
    assert parked.human_review_required is True
    assert parked.completed is False
    assert "Lantus SoloStar only" not in parked.response_text
    assert engine.case.adverse_event == "I have a headache"


def test_malayalam_unclear_event_and_medication_parks_for_review_verbatim():
    # The audited browser sequence: a garbled event answer, a verbatim
    # Malayalam description, and two unclear medication replies. The bounded
    # product safeguard must park the case for human review — never complete
    # it, never fabricate a medicine, never infer consent — while the event
    # description stays byte-for-byte identical.
    engine, workflow = _workflow("q-ml-premature-report")

    garbled = workflow.process_turn("ഐ കെ")
    assert garbled.next_question is not None
    assert garbled.next_question[0] == "adverse_event"
    assert engine.case.adverse_event is None

    event = "ഇന്ന് രാവിലെ ഈ മരുന്ന് കഴിച്ചിട്ട് കഴിഞ്ഞിട്ട് ആദ്യം വൈകിട്ട് കഴിയും"
    described = workflow.process_turn(event)
    assert engine.case.adverse_event == event  # byte-for-byte verbatim
    assert described.next_question is not None
    assert described.next_question[0] == "product"

    clarified = workflow.process_turn("എന്തിന്റെ പേരാണ് ഞങ്ങള്")
    assert clarified.next_question is not None
    assert clarified.next_question[0] == "product"
    assert "Sorry, I didn't catch the product name" in clarified.response_text

    parked = workflow.process_turn("മരുന്നിന്റെ പേരും ലൈറ്റർ ഡിക്രോ സാധിക്കുന്നു")

    # The bounded product safeguard parks the case for human review instead
    # of completing it or fabricating an unrelated medicine.
    assert parked.current_state == "HUMAN_REVIEW"
    assert parked.human_review_required is True
    assert parked.completed is False
    assert engine.case.product is None
    assert engine.case.consent_to_contact is None
    assert engine.case.adverse_event == event


def test_malayalam_product_variants_reach_step4_confirmation():
    # The two observed Malayalam medication answers (transliteration variants
    # of Lantus SoloStar) must reach the Step-4 confirmation, never an
    # out-of-scope termination, with the description preserved verbatim.
    for answer in (
        "ഇവിടെ പേരാണ് ഡി ലാറ്റസ് മികവ് സ്റ്റാ",
        "ലാബ് സോളോസ്റ്റാ",
    ):
        engine, workflow = _workflow("q-ml-variants")
        described = workflow.process_turn("I have a headache")
        assert described.next_question is not None
        assert described.next_question[0] == "product"

        result = workflow.process_turn(answer)

        assert engine.case.product == "Lantus SoloStar", answer
        assert result.next_question is not None, answer
        assert result.next_question[0] == "product_verification", answer
        assert "Lantus SoloStar?" in result.response_text, answer
        assert "cannot continue this report" not in result.response_text, answer

        confirmed = workflow.process_turn("yes")
        assert engine.case.product == "Lantus SoloStar", answer
        assert confirmed.next_question is not None, answer
        assert confirmed.next_question[0] == "reporter_role", answer


def test_denied_malayalam_variant_keeps_out_of_scope_termination():
    engine, workflow = _workflow("q-ml-variant-denied")
    described = workflow.process_turn("I have a headache")
    assert described.next_question is not None
    assert described.next_question[0] == "product"

    result = workflow.process_turn("ലാബ് സോളോസ്റ്റാ")
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"

    denied = workflow.process_turn("no")
    assert denied.next_question is not None
    assert denied.next_question[0] == "product"

    unrelated = workflow.process_turn("paracetamol")
    # Naming a genuinely different medicine keeps the existing out-of-scope
    # termination; the description stays preserved verbatim.
    assert "Lantus SoloStar only" in unrelated.response_text
    assert "cannot continue this report" in unrelated.response_text
    assert unrelated.next_question is None
    assert engine.case.adverse_event == "I have a headache"


def test_denied_malayalam_product_follows_the_out_of_scope_termination():
    engine, workflow = _workflow("q-malayalam-denied")
    workflow.process_turn("ഞാൻ എടുത്ത മരുന്നാണ് ലാൻഡസ് കൊതുക")

    # Denying the verification follows the existing path: the medication
    # question is re-asked.
    denied = workflow.process_turn("no")
    assert denied.next_question is not None
    assert denied.next_question[0] == "product"

    # Naming a genuinely different medicine keeps the existing Lantus-only
    # termination: intake stops (no further questions) and the verbatim
    # Malayalam description stays preserved.
    unrelated = workflow.process_turn("paracetamol")
    assert "Lantus SoloStar only" in unrelated.response_text
    assert "cannot continue this report" in unrelated.response_text
    assert unrelated.next_question is None
    assert "ലാൻഡസ് കൊതുക" in engine.case.adverse_event


def test_hindi_lantus_stem_variant_recognizes_the_medication():
    # The observed failure: "लैंटर" at the medication question was not
    # recognized; the fuller "लैंडर सोलो स्टार" worked only via the sub-brand.
    engine, workflow = _workflow("q-hindi-lantus")
    workflow.process_turn("Hello")
    workflow.process_turn("I have a headache")

    result = workflow.process_turn("लैंटर")

    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    assert "Lantus SoloStar?" in result.response_text

    verified = workflow.process_turn("yes")
    assert engine.case.product == "Lantus SoloStar"
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role"


# ---- product-confirmation replies in all ten locales -------------------------

# (affirmative, negative) replies per locale. All are conservative standalone
# answers; the Malayalam forms are the observed browser-test phrases.
_CONFIRM_LANGUAGE_PHRASES = {
    "en-IN": ("yes", "no"),
    "hi-IN": ("हाँ", "नहीं"),
    "bn-IN": ("হ্যাঁ", "না"),
    "gu-IN": ("હા", "ના"),
    "kn-IN": ("ಹೌದು", "ಇಲ್ಲ"),
    "ml-IN": ("അതെ തന്നെ", "ഇല്ല"),
    "mr-IN": ("होय", "नाही"),
    "pa-IN": ("ਹਾਂ", "ਨਹੀਂ"),
    "ta-IN": ("ஆம்", "இல்லை"),
    "te-IN": ("అవును", "కాదు"),
}


def _at_medication_verification(case_id: str):
    engine, workflow = _workflow(case_id)
    result = workflow.process_turn("I took Lantus and have a headache")
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    return engine, workflow


@pytest.mark.parametrize("language", sorted(_CONFIRM_LANGUAGE_PHRASES))
def test_affirmative_reply_in_every_locale_confirms_the_medication(language):
    affirmative, _ = _CONFIRM_LANGUAGE_PHRASES[language]
    engine, workflow = _at_medication_verification(f"conf-yes-{language}")

    result = workflow.process_turn(affirmative)

    # Only a clearly affirmative reply advances the workflow as confirmed.
    assert engine.case.product == "Lantus SoloStar", language
    assert engine.case.pending_confirmation is None, language
    assert result.next_question is not None, language
    assert result.next_question[0] == "reporter_role", language
    assert result.safety_status == "ok", language


@pytest.mark.parametrize("language", sorted(_CONFIRM_LANGUAGE_PHRASES))
def test_negative_reply_in_every_locale_reasks_the_medication(language):
    _, negative = _CONFIRM_LANGUAGE_PHRASES[language]
    engine, workflow = _at_medication_verification(f"conf-no-{language}")

    result = workflow.process_turn(negative)

    # A clear denial follows the existing correction/re-ask path: the captured
    # product is never silently kept or confirmed.
    assert result.next_question is not None, language
    assert result.next_question[0] == "product", language
    assert engine.case.pending_confirmation is None, language


@pytest.mark.parametrize(
    "ambiguous",
    ["maybe", "I am not sure what you mean", "crocin"],
)
def test_ambiguous_or_unrelated_replies_never_confirm_the_medication(ambiguous):
    engine, workflow = _at_medication_verification("conf-ambiguous")

    result = workflow.process_turn(ambiguous)

    # Ambiguity is clarified, never silently confirmed; the bounded attempt
    # safeguard keeps the loop finite.
    assert engine.case.product == "Lantus SoloStar"
    assert result.safety_status != "urgent_life_threatening"
    assert result.next_question is not None
    assert result.next_question[0] in ("product", "product_verification")


def test_repeated_ambiguous_verification_is_bounded():
    engine, workflow = _at_medication_verification("conf-bounded")
    workflow.process_turn("maybe")
    workflow.process_turn("hmm")
    third = workflow.process_turn("hmm")

    # The bounded-attempt safeguard (three unclear replies) parks the case
    # for human review instead of looping the verification forever.
    assert third.human_review_required is True
    assert third.completed is False
    assert engine.case.adverse_event == "I took Lantus and have a headache"


def test_malayalam_short_affirmative_and_extended_affirmation_confirm():
    for phrase in ("അതെ", "അതെ, ശരി"):
        engine, workflow = _at_medication_verification(f"conf-ml-{phrase}")
        result = workflow.process_turn(phrase)
        assert engine.case.product == "Lantus SoloStar", phrase
        assert result.next_question is not None
        assert result.next_question[0] == "reporter_role", phrase


def test_extended_malayalam_affirmation_with_a_new_value_is_clarified():
    # "അതെ" followed by a possible new value behaves like "yes alpha beta":
    # clarified, never silently confirmed.
    engine, workflow = _at_medication_verification("conf-ml-value")
    result = workflow.process_turn("അതെ crocin")
    assert engine.case.product == "Lantus SoloStar"  # unchanged
    assert result.next_question is not None


def test_consent_is_not_recorded_by_an_ambiguous_reply():
    # Consent extraction stays gated on the explicit "consent" keyword: an
    # ambiguous reply records nothing, and a clear negative in a new locale
    # never silently grants consent.
    provider = AnswerLLMProvider()
    assert (
        "consent_to_contact" not in provider.interpret("maybe", {}).extracted_fields
    )
    assert (
        provider.interpret(
            "please note my consent ഇല്ല", {}
        ).extracted_fields.get("consent_to_contact")
        is False
    )


# ---- per-locale product confirmation + Marathi browser regression ------------


def _to_product_verification(case_id: str):
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")
    assert workflow._scope_verification_pending is True
    return engine, workflow


@pytest.mark.parametrize(
    ("locale", "affirmative"),
    [
        (locale, form)
        for locale, forms in _NATURAL_AFFIRMATIVES.items()
        for form in (forms if isinstance(forms, list) else [forms])
    ]
    + [("mr-IN", "हो बरोबर आहे"), ("mr-IN", "बरोबर आहे"), ("mr-IN", "हो")],
    ids=lambda value: str(value)[:24],
)
def test_product_confirmation_advances_in_every_locale(locale, affirmative):
    # The Step 4 medicine verification must accept each locale's supported
    # affirmative form and advance to the reporter-role question - exactly
    # the state transition the working Malayalam flow performs.
    engine, workflow = _to_product_verification(f"q-verify-{locale}-{affirmative[:6]}")

    verified = workflow.process_turn(affirmative)

    assert workflow._product_verified is True, (locale, affirmative)
    assert workflow._scope_verification_pending is False
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role", (locale, affirmative)
    assert verified.completed is False


def test_marathi_browser_regression_confirmation_and_language():
    # The reported browser sequence, pinned: the conversation is Marathi
    # (its sentences carry Marathi markers), the medicine is confirmed with
    # the Marathi compound affirmative, the flow advances to the reporter
    # role, and nothing switches the session away from Marathi. The event
    # description is captured deterministically first: live Marathi
    # description extraction is the documented STT boundary, not this test's
    # subject.
    from app.language import detect_language

    engine, workflow = _workflow("q-mr-browser-regression")
    report = "\u092e\u0932\u093e \u0915\u093e\u0932\u092a\u093e\u0938\u0942\u0928 \u0921\u094b\u0915\u0947 \u0926\u0941\u0916\u0924 \u0906\u0939\u0947"

    assert detect_language(report) == "mr-IN"
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")

    compound = "\u0939\u094b \u092c\u0930\u094b\u092c\u0930 \u0906\u0939\u0947"
    assert detect_language(compound) == "mr-IN"
    verified = workflow.process_turn(compound)

    assert workflow._product_verified is True
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role"
    assert verified.completed is False
    assert verified.human_review_required is False


# ---- Marathi Latin transliterations + confirmation distinctness ---------------


@pytest.mark.parametrize(
    "affirmative",
    [
        "हो",
        "होय",
        "बरोबर",
        "बरोबर आहे",
        "हो बरोबर आहे",
        "होय ते बरोबर आहे",
        "hoy",
        "ho",
        "barobar",
        "barobar aahe",
        "hoy barobar aahe",
        "hoy te barobar aahe",
    ],
)
def test_marathi_affirmative_variants_confirm_the_product(affirmative):
    # The reported live failure: the medicine confirmation was answered in
    # Marathi (native script and Latin transliteration) and the workflow
    # repeated the question. Every supported Marathi affirmative variant
    # must verify the product and advance to the reporter-role question.
    engine, workflow = _to_product_verification(f"q-mr-variants-{affirmative[:8]}")

    verified = workflow.process_turn(affirmative)

    assert workflow._product_verified is True, affirmative
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role", affirmative


def test_marathi_negation_and_medicine_corrections_stay_distinct():
    # Affirmative, negative and correction responses stay distinct: a
    # negated compound never confirms, and a genuine medicine-name
    # correction is still handled by the correction path (never read as an
    # affirmative).
    engine, workflow = _to_product_verification("q-mr-distinct")

    negated = workflow.process_turn("बरोबर नाही")
    assert workflow._product_verified is False
    assert negated.next_question is not None

    engine2, workflow2 = _to_product_verification("q-mr-correction-distinct")
    corrected = workflow2.process_turn("actually it was paracetamol")
    # A different medicine is the accepted out-of-scope path: the product is
    # never verified as Lantus and routine intake ends with the bounded
    # scope notice - it is never read as an affirmative confirmation.
    assert workflow2._product_verified is False
    assert corrected.next_question is None
    assert "prototype is configured for Lantus SoloStar only" in (
        corrected.response_text
    )


# ---- reported Marathi compounds at product and patient-detail confirmation ---


@pytest.mark.parametrize(
    ("affirmative", "field"),
    [
        ("होय बरोबर आहे", "product"),
        ("होय बरोबर आहे", "patient_identifier"),
        ("होय देतोय मी", "contact_email"),
    ],
    ids=["product", "patient-details", "email"],
)
def test_marathi_compound_affirmatives_across_confirmation_states(affirmative, field):
    # The reported live failure: the same Marathi compound behaved
    # differently across confirmation states. It must confirm consistently
    # at the product verification, the patient-detail read-back and the
    # email read-back.
    if field == "product":
        engine, workflow = _to_product_verification("q-mr-compound-product")
    elif field == "patient_identifier":
        engine = AEStateEngine.create_case("q-mr-compound-name")
        engine.update("product", "Lantus SoloStar")
        engine.update("adverse_event", "rash")
        engine.update("reporter_role", "self")
        engine.update("patient_age", "15-Mar-1995")
        workflow = ConversationalAEWorkflow(
            engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
        )
        workflow.process_turn("Alpha Jones")
        assert workflow.engine.case.pending_confirmation == "patient_identifier"
    else:
        from tests.test_ae_email_flow import _to_email_offer

        engine, workflow = _to_email_offer("q-mr-compound-email")
        workflow.process_turn("yes")
        workflow.process_turn("alpha@example.com")
        assert workflow.engine.case.pending_confirmation == "contact_email"

    confirmed = workflow.process_turn(affirmative)

    assert workflow.engine.case.pending_confirmation is None, (field, affirmative)
    if field == "product":
        assert workflow._product_verified is True
        assert confirmed.next_question is not None
        assert confirmed.next_question[0] == "reporter_role"
    elif field == "patient_identifier":
        assert workflow.engine.case.patient_identifier == "Alpha Jones"
        assert confirmed.next_question is not None
        assert confirmed.next_question[0] == "contact_phone"
    else:
        assert workflow.engine.case.contact_email == "alpha@example.com"
        assert confirmed.next_question is not None
        assert confirmed.next_question[0] == "seriousness"


def test_ambiguous_dob_loop_specific_clarification_bounded_recovery():
    # The reported multi-turn failure IN THE OBSERVED LIVE ORDER: the
    # combined answer's name is held clean, the FIRST ambiguous retry is
    # the transliterated form (ट्वेंटी में टू थाउज़ंड) and must get the
    # SPECIFIC month clarification (never the generic re-ask), the second
    # ambiguous retry stays bounded at the review park, and the report
    # never claims the DOB was captured, the name confirmed, or the
    # workflow completed.
    from app.workflow.ae_engine import WorkflowState

    engine, workflow = _workflow("q-dob-clar-recovery")
    workflow.process_turn("I have a headache")
    workflow.process_turn("Lantus")
    workflow.process_turn("yes")
    workflow.process_turn("myself")

    combined = workflow.process_turn(
        "राजू के बीस में दो हजार"
    )
    # The name candidate is clean (the failed date attempt stripped; the
    # initial के retained) and the workflow proceeds to the DOB question.
    assert engine.case.confirmation_candidate["patient_identifier"] == (
        "राजू के"
    )
    assert combined.next_question is not None
    assert combined.next_question[0] == "patient_age"
    assert engine.case.patient_age is None

    # First ambiguous retry (the transliterated form): the SPECIFIC
    # clarification - the month is named as the missing piece, never guessed.
    retry = workflow.process_turn(
        "ट्वेंटी में टू थाउज़ंट"
    )
    assert (
        "I could not hear the month clearly - please say the date of birth "
        "as day, month, and year." in retry.response_text
    )
    assert "Could you say the date of birth again?" not in retry.response_text
    assert engine.case.patient_age is None

    # Second ambiguous retry: the bounded review park - the held name and
    # the unknown DOB are preserved, nothing is falsely completed.
    parked = workflow.process_turn(
        "बीस में दो हजार"
    )
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.completed is False
    assert engine.case.patient_age == "unknown"
    snapshot = workflow._collected_snapshot()
    assert snapshot["patient_identifier"] is None  # unconfirmed: never claimed
    assert snapshot["patient_identifier_held"] == "राजू के"
    assert snapshot["patient_age"] == "unknown"
    assert parked.human_review_required is True



@pytest.mark.parametrize(
    "form",
    [
        "हाँ जी सही है",
        "हाँ यह सही है",
        "हाँ, सही है",
    ],
    ids=["haan-ji-sahi", "haan-yah-sahi", "haan-comma-sahi"],
)
def test_hindi_polite_confirmations_commit_the_read_back(form):
    # The reported DOB-confirmation failure: the polite/pointer particles
    # after the हाँ lead ("जी", "यह") must confirm the pending read-back
    # and commit EXACTLY the value read back - never a reconstructed value.
    engine = AEStateEngine.create_case("q-hi-confirm-" + str(abs(hash(form)) % 10000))
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_age", "20-Jun-2000")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    readback = workflow.process_turn("Alpha Jones")
    assert "Alpha Jones, 20-Jun-2000" in readback.next_question[1]

    confirmed = workflow.process_turn(form)

    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.case.patient_age == "20-Jun-2000"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "contact_phone"
