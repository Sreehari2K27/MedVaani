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
    question = "Could you tell me the name of the medication involved?"

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
        "alpha beta and my date of birth is twenty eighth January twenty twenty",
    )
    # The confirmation carries canonical values only.
    assert "I have your details as alpha beta, 28-Jan-2020." in readback.response_text
    assert "twenty eighth" not in readback.response_text

    workflow.process_turn("no")
    asked = workflow.process_turn("the date of birth was incorrect")
    assert asked.next_question is not None
    assert asked.next_question[0] == "patient_age"

    corrected = workflow.process_turn("twentieth January two thousand twenty")
    # Canonical values only: the old raw phrase and the old date are gone.
    assert "I have your details as alpha beta, 20-Jan-2020." in (
        corrected.response_text
    )
    assert "twenty eighth" not in corrected.response_text
    assert "twenty twenty" not in corrected.response_text
    assert engine.case.patient_age == "20-Jan-2020"
    # The name candidate is untouched by the date correction.
    assert engine.case.confirmation_candidate == {"patient_identifier": "alpha beta"}

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "alpha beta"
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
