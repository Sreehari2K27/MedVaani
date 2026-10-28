"""Regression tests for the reported Malayalam voice session defects.

Covers the three connected defects together, exactly as they appeared in the
reported browser transcript:

1. language stability: English number words / spelled letters inside a
   Malayalam session must never switch the assistant's response language,
2. phone confirmation: the Malayalam echo reply ("അത് ശരിയാണ്") and
   English words-only confirmation-style speech must commit once and
   advance, while genuine corrections still replace the candidate,
3. recoverable review: retry exhaustion parks the case with the unresolved
   field named and recovery invited, and a re-engaged field resumes the
   interview - never a dead end, never a false completion.

All tests are deterministic and offline: the AnswerLLMProvider is the
existing deterministic provider; the text endpoint bypasses STT/TTS entirely.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    CONFIRMATION_REVIEW_NOTICE,
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)

ATHU_SARIYANU = "അത് ശരിയാണ്"  # "that is correct" - the observed echo reply
ENGLISH_NUMBER_WORDS = "nine two two four four one nine one zero three"
PHONE_TAIL = "9103"
MALAYALAM_DIGIT_WORDS = "ഒൻപത് രണ്ട് രണ്ട് നാല് നാല് ഒന്ന് ഒൻപത് ഒന്ന് പൂജ്യം"
SPELLED_EMAIL = "S R E E H A R I at the rate gmail dot com"
MALAYALAM_DESCRIPTION = "എനിക്ക് തലവേദനയുണ്ട്"


def _phone_ready_workflow(case_id: str):
    """A workflow waiting at the contact-phone question (voice scope)."""
    engine = AEStateEngine.create_case(case_id)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju P K")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    assert engine.next_field()[0] == "contact_phone"
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    return engine, workflow


# ---- Issue 1: language stability (endpoint level, text turns) ----------------


def _client() -> TestClient:
    router = create_voice_conversation_router(
        stt=None,
        tts=None,
        session_store=SessionStore(),
        llm_provider=AnswerLLMProvider(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _turn(client: TestClient, session_id: str, message: str) -> dict:
    resp = client.post(
        "/text/conversation",
        data={"session_id": session_id, "message": message, "language": "ml-IN"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_english_number_words_keep_the_malayalam_session():
    # The reported defect: the phone number spoken as English number words
    # must never switch the assistant away from the selected Malayalam.
    client = _client()
    session_id = "ml-lang-numwords"

    first = _turn(client, session_id, MALAYALAM_DESCRIPTION)
    assert first["conversation_language"] == "ml-IN"

    second = _turn(client, session_id, ENGLISH_NUMBER_WORDS)
    assert second["conversation_language"] == "ml-IN"
    assert second["language_source"] == "session"
    # The spoken form is translated (all-or-nothing), never silently English.
    assert second["language_translated"] is True
    assert second["spoken_response_text"] != second["response_text"]

    # And the next question is still asked in Malayalam.
    third = _turn(client, session_id, ATHU_SARIYANU)
    assert third["conversation_language"] == "ml-IN"
    assert third["language_translated"] is True


def test_malayalam_number_words_keep_the_malayalam_session():
    client = _client()
    session_id = "ml-lang-mlnumwords"

    _turn(client, session_id, MALAYALAM_DESCRIPTION)
    data = _turn(client, session_id, MALAYALAM_DIGIT_WORDS)

    assert data["conversation_language"] == "ml-IN"
    assert data["language_translated"] is True


def test_spelled_letters_and_email_cues_keep_the_malayalam_session():
    # Letters and email-spelling cues are field content, not a language switch.
    client = _client()
    session_id = "ml-lang-spelling"

    _turn(client, session_id, MALAYALAM_DESCRIPTION)
    data = _turn(client, session_id, SPELLED_EMAIL)

    assert data["conversation_language"] == "ml-IN"
    assert data["language_translated"] is True


def test_genuine_english_sentence_still_switches_the_session():
    # The existing mirroring feature is preserved: a real English sentence
    # still switches (weak evidence never does).
    client = _client()
    session_id = "ml-lang-sentence"

    _turn(client, session_id, MALAYALAM_DESCRIPTION)
    data = _turn(client, session_id, "Now I am speaking in English.")

    assert data["conversation_language"] == "en-IN"


# ---- Issue 2: phone confirmation ----------------------------------------------


def test_malayalam_phone_confirmation_commits_once_and_advances():
    engine, workflow = _phone_ready_workflow("AE-regr-ml-confirm")

    readback = workflow.process_turn(ENGLISH_NUMBER_WORDS)
    assert engine.case.contact_phone is None  # held for read-back
    assert PHONE_TAIL in readback.response_text  # tail-only privacy wording

    confirmed = workflow.process_turn(ATHU_SARIYANU)

    assert engine.case.contact_phone == "9224419103"
    assert engine.case.pending_confirmation is None
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_malayalam_phone_confirmation_short_form_commits():
    engine, workflow = _phone_ready_workflow("AE-regr-ml-confirm-short")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    confirmed = workflow.process_turn("അത് ശരി")

    assert engine.case.contact_phone == "9224419103"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_english_words_only_confirmation_commits_within_any_session():
    # "the number is correct" carries an affirmative cue and no possible
    # number: it confirms instead of looping as ambiguous.
    engine, workflow = _phone_ready_workflow("AE-regr-en-words")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    confirmed = workflow.process_turn("the number is correct")

    assert engine.case.contact_phone == "9224419103"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_genuine_phone_correction_replaces_the_candidate():
    engine, workflow = _phone_ready_workflow("AE-regr-correction")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    workflow.process_turn("no")  # reject the read-back
    corrected = workflow.process_turn("nine eight eight six zero nine four three two one")

    # The NEW candidate is read back; nothing was committed.
    assert engine.case.contact_phone is None
    assert "4321" in corrected.response_text
    assert "9103" not in corrected.response_text

    confirmed = workflow.process_turn(ATHU_SARIYANU)

    assert engine.case.contact_phone == "9886094321"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_spoken_digit_reply_while_pending_is_a_correction_not_a_confirmation():
    # A reply that DOES carry a possible number must never be swallowed by
    # the affirmative cue: it re-opens the read-back with the new candidate.
    engine, workflow = _phone_ready_workflow("AE-regr-digit-correction")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    ambiguous = workflow.process_turn(
        "yes it is nine eight eight six zero nine four three two one"
    )

    assert engine.case.contact_phone is None  # never silently confirmed
    assert ambiguous.next_question is not None
    assert ambiguous.next_question[0] == "confirmation"
    assert "4321" in ambiguous.response_text


def test_short_or_invalid_number_is_never_accepted_or_committed():
    engine, workflow = _phone_ready_workflow("AE-regr-short")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    replied = workflow.process_turn("12345")

    # Too short to be a number: the pending candidate is untouched and the
    # read-back is re-asked (clarified), never confirmed.
    assert engine.case.contact_phone is None
    assert engine.case.confirmation_candidate["contact_phone"] == "9224419103"
    assert replied.next_question is not None
    assert replied.next_question[0] == "confirmation"


def test_mumble_never_replaces_or_commits_a_phone_candidate():
    engine, workflow = _phone_ready_workflow("AE-regr-mumble")

    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    workflow.process_turn("hmm")

    assert engine.case.confirmation_candidate["contact_phone"] == "9224419103"


# ---- Issue 3: recoverable review ----------------------------------------------


def _parked_phone(case_id: str):
    engine, workflow = _phone_ready_workflow(case_id)
    workflow.process_turn(ENGLISH_NUMBER_WORDS)
    for _ in range(3):
        workflow.process_turn("I'm not sure")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    return engine, workflow


def test_retry_exhaustion_parks_with_named_field_and_recovery_invitation():
    engine, workflow = _parked_phone("AE-regr-park")

    assert engine.case.contact_phone is None  # never falsely committed
    assert workflow._collected_snapshot()["contact_phone"] is None
    # The unconfirmed candidate stays reviewer-visible, clearly unconfirmed.
    assert workflow._collected_snapshot()["contact_phone_held"] == "9224419103"
    assert "contact_phone" in engine.missing_information()

    final = workflow.process_turn("hmm")
    text = final.response_text

    assert final.completed is False
    assert final.human_review_required is True
    assert final.priority == "normal"  # review, never an urgency escalation
    assert CONFIRMATION_REVIEW_NOTICE in text
    assert "We could not confirm the phone number." in text
    assert "You can say it again now, or our team will collect it during review." in text
    assert "please say yes" not in text


def test_parked_phone_resumes_after_restatement_and_confirmation():
    engine, workflow = _parked_phone("AE-regr-resume-restated")

    # A mumble does NOT resume the parked interview.
    mumble = workflow.process_turn("hmm")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert mumble.human_review_required is True

    # Restating the number re-engages the field: the interview resumes with
    # the fresh read-back, the confirmation commits, and questioning
    # continues from where it stopped.
    restated = workflow.process_turn(ENGLISH_NUMBER_WORDS)
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert restated.human_review_required is False
    assert restated.next_question is not None
    assert restated.next_question[0] == "confirmation"
    assert PHONE_TAIL in restated.response_text

    confirmed = workflow.process_turn(ATHU_SARIYANU)

    assert engine.case.contact_phone == "9224419103"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"
    # The review flag is cleared on resume; everything collected is kept.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.consent_to_contact is True


def test_parked_phone_resumes_after_a_late_confirmation():
    engine, workflow = _parked_phone("AE-regr-resume-late")

    resumed = workflow.process_turn(ATHU_SARIYANU)

    assert engine.case.contact_phone == "9224419103"
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert resumed.human_review_required is False
    assert resumed.next_question is not None
    assert resumed.next_question[0] == "seriousness"


def test_parked_name_holds_the_candidate_without_claiming_it():
    # Identity variant: the held name is reviewer-visible under the held key
    # while the flat value stays unclaimed, and the park invites recovery.
    engine = AEStateEngine.create_case("AE-regr-held-name")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn("Demo Patient")
    workflow.process_turn("25")
    assert engine.case.pending_confirmation == "patient_identifier"
    during_pending = workflow._collected_snapshot()
    assert during_pending["patient_identifier"] is None
    assert during_pending["patient_identifier_held"] is None

    for _ in range(3):
        workflow.process_turn("I'm not sure")

    parked = workflow.process_turn("hmm")
    snapshot = workflow._collected_snapshot()

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.patient_identifier is None
    assert snapshot["patient_identifier"] is None
    assert snapshot["patient_identifier_held"] == "Demo Patient"
    assert "We could not confirm the name." in parked.response_text
    assert "You can say it again now, or our team will collect it during review." in (
        parked.response_text
    )


def test_parked_name_resumes_after_the_name_is_said_again():
    engine = AEStateEngine.create_case("AE-regr-resume-name")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn("Demo Patient")
    workflow.process_turn("25")
    for _ in range(3):
        workflow.process_turn("I'm not sure")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW

    restated = workflow.process_turn("Beta Jones")

    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert engine.case.patient_identifier is None  # still a held candidate
    assert restated.next_question is not None
    assert restated.next_question[0] == "confirmation"

    confirmed = workflow.process_turn("yes")

    assert engine.case.patient_identifier == "Beta Jones"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "contact_phone"


# ---- repeat requests never park the interview ---------------------------------


def test_repeat_request_reasks_without_consuming_the_field_bound():
    engine = AEStateEngine.create_case("AE-regr-repeat")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I took Lantus and have a rash")
    workflow.process_turn("yes")

    workflow.process_turn("shayad")  # unclear reply: one attempt
    repeat = workflow.process_turn("dobara kahiye")  # repeat request: no bound
    answer = workflow.process_turn("myself")

    assert repeat.next_question is not None
    assert repeat.next_question[0] == "reporter_role"
    assert engine.case.reporter_role == "self"
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert answer.next_question is not None
    assert answer.next_question[0] == "patient_identifier"


def test_genuinely_unclear_replies_still_park_after_the_bound():
    # The bound itself is untouched: two genuinely unclear replies still park.
    engine = AEStateEngine.create_case("AE-regr-bound-intact")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I took Lantus and have a rash")
    workflow.process_turn("yes")

    workflow.process_turn("hmm")
    parked = workflow.process_turn("not sure")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.reporter_role is None
    assert parked.human_review_required is True
    assert parked.completed is False
