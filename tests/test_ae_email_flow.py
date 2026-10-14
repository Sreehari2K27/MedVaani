"""Focused tests for the AE email-capture, verification and consent integrity.

Follow-up consent, email availability and email verification are three
independent states: an email reply can never change consent, a rejected or
unprovided email never blocks the optional step, and an unconfirmed address
is never reported as verified. All data is synthetic; all providers are
offline stubs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    EMAIL_OFFER_QUESTION,
    EMAIL_REQUEST_QUESTION,
    SEVERITY_QUESTION,
    VOICE_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine,
        AnswerLLMProvider(),
        confirm_fields=VOICE_CONFIRMATION_FIELDS,
    )
    return engine, workflow


def _to_email_offer(case_id: str):
    """Workflow with consent granted and the Step 8 email offer pending."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")  # Step 4
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Jones, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    consented = workflow.process_turn("Yes, I consent")
    assert engine.case.consent_to_contact is True
    assert consented.next_question is not None
    assert consented.next_question[0] == "contact_email"
    return engine, workflow


def _email_to_readback(workflow):
    offered = workflow.process_turn("yes")
    assert offered.next_question is not None
    assert offered.next_question[1] == EMAIL_REQUEST_QUESTION
    return workflow.process_turn("alpha@example.com")


# ---- 1-4. confirmation, rejection and correction ----------------------------


def test_consent_yes_valid_email_and_confirmation_keeps_both():
    engine, workflow = _to_email_offer("em-confirm")

    readback = _email_to_readback(workflow)
    assert "I heard your email address as alpha@example.com." in (
        readback.response_text
    )
    # Unconfirmed: the candidate is never reported as the address.
    assert engine.case.contact_email is None

    confirmed = workflow.process_turn("yes")

    assert engine.case.consent_to_contact is True
    assert engine.case.contact_email == "alpha@example.com"
    assert engine.case.contact_email_declined is False
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_incorrect_email_phrase_rejects_the_candidate_and_asks_for_a_fix():
    engine, workflow = _to_email_offer("em-incorrect")
    _email_to_readback(workflow)

    rejected = workflow.process_turn("the email address is incorrect")

    # The rejected candidate is discarded, never kept as verified.
    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    # The corrected address is asked for.
    assert rejected.next_question is not None
    assert rejected.next_question[1] == EMAIL_REQUEST_QUESTION
    # Consent is untouched by an email correction.
    assert engine.case.consent_to_contact is True


def test_plain_no_to_the_email_readback_never_changes_consent():
    engine, workflow = _to_email_offer("em-no")
    _email_to_readback(workflow)

    rejected = workflow.process_turn("no")

    assert engine.case.contact_email is None
    assert rejected.next_question is not None
    assert rejected.next_question[1] == EMAIL_REQUEST_QUESTION
    assert engine.case.consent_to_contact is True


def test_corrected_email_updates_only_the_email_field():
    engine, workflow = _to_email_offer("em-corrected")
    _email_to_readback(workflow)
    workflow.process_turn("the email address is incorrect")

    fixed = workflow.process_turn("beta@example.com")
    assert "I heard your email address as beta@example.com." in fixed.response_text

    workflow.process_turn("yes")
    assert engine.case.contact_email == "beta@example.com"
    # Identity and contact details are untouched.
    assert engine.case.confirmation_candidate == {}
    assert engine.case.patient_age == "05-Jan-1990"
    assert engine.case.contact_phone == "9886094321"
    assert engine.case.consent_to_contact is True


# ---- 5. optional email skipped ----------------------------------------------


def test_no_email_address_is_recorded_as_unavailable_and_interview_continues():
    engine, workflow = _to_email_offer("em-unavailable")

    result = workflow.process_turn("I don't have an email address")

    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.consent_to_contact is True
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


def test_declining_at_the_request_turn_also_skips_safely():
    engine, workflow = _to_email_offer("em-decline-request")
    workflow.process_turn("yes")

    result = workflow.process_turn("no, I would rather not share it")

    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.consent_to_contact is True
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


# ---- 8. bounded ambiguity ---------------------------------------------------


def test_ambiguous_email_replies_hit_the_retry_limit_and_continue():
    engine, workflow = _to_email_offer("em-bound")
    workflow.process_turn("yes")  # email requested

    workflow.process_turn("hmm")
    workflow.process_turn("not sure")
    final = workflow.process_turn("maybe")

    # The optional email is left unprovided and the interview continues:
    # no loop, no human-review park for an optional field.
    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.pending_confirmation is None
    assert engine.current_state.value != "HUMAN_REVIEW"
    assert final.next_question is not None
    assert final.next_question[0] == "seriousness"
    assert engine.case.consent_to_contact is True


# ---- 9-10. consent integrity ------------------------------------------------


def test_explicit_consent_answers_set_the_field_only_at_the_consent_turn():
    engine, workflow = _workflow("em-consent-yes")
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 05-01-1990",
        "yes",
        "9886094321",
        "yes",
    ):
        workflow.process_turn(message)
    assert engine.case.consent_to_contact is None

    workflow.process_turn("Yes, I consent")
    assert engine.case.consent_to_contact is True


def test_consent_no_is_never_flipped_by_a_later_email_answer():
    engine, workflow = _workflow("em-consent-no")
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 05-01-1990",
        "yes",
        "9886094321",
        "yes",
    ):
        workflow.process_turn(message)
    workflow.process_turn("no thanks")

    assert engine.case.consent_to_contact is False

    # A later email-shaped answer cannot withdraw or flip the consent.
    workflow.process_turn("you can email me at alpha@example.com")
    assert engine.case.consent_to_contact is False


# ---- 12. report integrity ---------------------------------------------------


def test_final_report_distinguishes_consent_from_email_status():
    # Confirmed email: consent and the address are separate report fields.
    engine, workflow = _to_email_offer("em-report-final")
    _email_to_readback(workflow)
    workflow.process_turn("yes")
    contact = engine.to_dict()["contact"]
    assert contact["consent_to_contact"] is True
    assert contact["email"] == "alpha@example.com"
    assert contact["email_declined"] is False

    # Declined/unavailable email: consent stays, the email stays unprovided.
    engine3, workflow3 = _to_email_offer("em-report-declined")
    workflow3.process_turn("I don't have an email address")
    contact3 = engine3.to_dict()["contact"]
    assert contact3["consent_to_contact"] is True
    assert contact3["email"] is None
    assert contact3["email_declined"] is True


# ---- 6-7. endpoint-level: empty transcript and provider failure -------------


class _QueueSTT:
    def __init__(self, transcripts: list[str], fail_status: int | None = None):
        self._transcripts = list(transcripts)
        self._fail_status = fail_status

    def transcribe(self, audio_path, language_code="en-IN"):
        from app.services.prisma import PrismaAPIError

        if self._transcripts:
            return SimpleNamespace(text=self._transcripts.pop(0))
        raise PrismaAPIError(
            "Gnani STT returned an empty transcript.", status_code=self._fail_status or 200
        )


class _WavTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\x00" * 40)
        return path


def _voice_client(stt, store):
    router = create_voice_conversation_router(
        stt=stt, tts=_WavTTS(), session_store=store, llm_provider=AnswerLLMProvider()
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, session_id):
    return client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": "en-IN"},
    )


EMAIL_JOURNEY = [
    "I have a headache after taking Lantus",
    "yes",
    "myself",
    "Alpha Jones, 05-01-1990",
    "yes",
    "9886094321",
    "yes",
    "Yes, I consent",
    "yes",  # email requested; the next turn would carry the address
]


def test_empty_transcript_during_email_capture_retries_without_state_change():
    from app.config import get_gnani_api_key

    store = SessionStore()
    client = _voice_client(_QueueSTT(EMAIL_JOURNEY), store)
    session_id = "em-empty-stt"

    for _ in range(len(EMAIL_JOURNEY)):
        ok = _post(client, session_id)
        assert ok.status_code == 200

    session = store.get_or_create(session_id)
    before_state = session.ae_engine.current_state.value
    before_consent = session.ae_engine.case.consent_to_contact

    empty = _post(client, session_id)  # STT returns an empty transcript
    assert empty.status_code == 200
    body = empty.json()
    assert body["reason"] == "empty_transcript"
    assert "couldn't catch the email address" in body["response_text"]
    # Nothing advanced and nothing was treated as "no" or a rejection.
    case = store.get_or_create(session_id).ae_engine.case
    assert store.get_or_create(session_id).ae_engine.current_state.value == before_state
    assert case.consent_to_contact is before_consent
    assert case.contact_email_declined is False
    assert body["next_question"][0] == "contact_email"
    key = get_gnani_api_key()
    assert not key or key not in body["response_text"]


def test_provider_failure_preserves_the_captured_email_and_consent():
    from app.services.prisma import PrismaAPIError

    store = SessionStore()
    stt = _QueueSTT(EMAIL_JOURNEY + ["alpha@example.com", "yes"], fail_status=403)
    client = _voice_client(stt, store)
    session_id = "em-provider-fail"

    for _ in range(len(EMAIL_JOURNEY) + 2):
        ok = _post(client, session_id)
        assert ok.status_code == 200

    case = store.get_or_create(session_id).ae_engine.case
    assert case.contact_email == "alpha@example.com"
    assert case.consent_to_contact is True

    failed = _post(client, session_id)  # STT now fails with 403
    assert failed.status_code == 500

    after = store.get_or_create(session_id).ae_engine.case
    assert after.contact_email == "alpha@example.com"
    assert after.consent_to_contact is True
    assert after.contact_email_declined is False


# ---- 11. text and voice consistency -----------------------------------------


def test_text_endpoint_drives_the_same_email_flow():
    from app.main import create_app

    client = TestClient(create_app())
    session_id = "em-text"

    def turn(message: str) -> dict:
        resp = client.post(
            "/text/conversation",
            data={"session_id": session_id, "message": message},
        )
        assert resp.status_code == 200
        return resp.json()

    turn("I have a headache after taking Lantus")
    turn("yes")
    turn("myself")
    turn("Alpha Jones, 05-01-1990")
    turn("yes")
    turn("9886094321")
    turn("yes")
    consented = turn("Yes, I consent")
    assert consented["next_question"][0] == "contact_email"

    offered = turn("yes")
    assert offered["next_question"][1] == EMAIL_REQUEST_QUESTION

    readback = turn("alpha@example.com")
    assert "I heard your email address as alpha@example.com." in (
        readback["response_text"]
    )

    confirmed = turn("yes")
    collected = confirmed["ae"]["collected_information"]
    assert collected["consent_to_contact"] is True
    assert collected["contact_email"] == "alpha@example.com"
    assert collected["contact_email_declined"] is False
    assert confirmed["next_question"][0] == "seriousness"


# ---- spelled/dictated email normalization ----------------------------------


def _norm(text: str):
    from app.providers.answer import _normalize_spoken_email

    return _normalize_spoken_email(text)


def test_individually_spelled_letters_normalize():
    assert (
        _norm("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com")
        == "sreehari@gmail.com"
    )


def test_spoken_digits_and_mixed_alphanumeric_normalize():
    assert (
        _norm("A, B, C, one, two, three, at the rate, Gmail, dot, com")
        == "abc123@gmail.com"
    )


def test_repeated_letters_are_preserved():
    assert _norm("A, A, B, C, at the rate, Gmail, dot, com") == "aabc@gmail.com"
    # The repeated "E, E" in "sreehari" is kept, never collapsed.
    assert _norm("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com") == (
        "sreehari@gmail.com"
    )


def test_spoken_at_dot_underscore_and_hyphen_normalize():
    assert _norm("srihari at gmail dot com") == "srihari@gmail.com"
    assert (
        _norm("john dot doe underscore work at the rate gmail dot com")
        == "john.doe_work@gmail.com"
    )
    assert (
        _norm("my alias hyphen work at the rate gmail dot com")
        == "myalias-work@gmail.com"
    )


def test_ordinary_speech_never_becomes_an_email():
    # Evidence of an address (an @ cue and a dot cue ending in a real domain
    # label) is required; ordinary sentences are rejected.
    assert _norm("look at this dot on my arm") is None
    assert _norm("I am not sure") is None
    assert _norm("the email address is incorrect") is None


# ---- corrections and confirmation state -------------------------------------


def test_correction_phrase_replaces_the_candidate_with_the_new_address():
    engine, workflow = _to_email_offer("em-spell-correct")

    # Initial (misheard) candidate captured from a spoken address.
    first = workflow.process_turn("shrihari at gmail dot com")
    assert "I heard your email address as shrihari@gmail.com." in first.response_text

    # The spelled correction replaces it and is read back in normalized form.
    second = workflow.process_turn("it should be srihari at the rate gmail dot com")
    assert "I heard your email address as srihari@gmail.com." in second.response_text
    # The superseded candidate is never re-read.
    assert "shrihari@gmail.com" not in second.response_text

    workflow.process_turn("yes")
    # Only the latest candidate can be confirmed and included as verified.
    assert engine.case.contact_email == "srihari@gmail.com"


def test_no_the_email_should_be_enters_correction_handling():
    engine, workflow = _to_email_offer("em-should-be")
    _email_to_readback(workflow)

    corrected = workflow.process_turn(
        "No, the email should be alpha dot beta at the rate gmail dot com"
    )

    assert "I heard your email address as alpha.beta@gmail.com." in (
        corrected.response_text
    )
    assert engine.case.contact_email is None  # candidate only, unverified

    workflow.process_turn("yes")
    assert engine.case.contact_email == "alpha.beta@gmail.com"


def test_unclear_confirmation_never_verifies_the_candidate():
    engine, workflow = _to_email_offer("em-unclear")
    _email_to_readback(workflow)

    unclear = workflow.process_turn("hmm")

    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    assert unclear.next_question is not None
    assert unclear.next_question[0] == "confirmation"
    # The interview stays in the email workflow, never the severity question.
    assert "severe or life-threatening" not in unclear.response_text


def test_email_pending_states_never_ask_the_severity_question():
    engine, workflow = _to_email_offer("em-no-severity-leak")

    for reply in ("yes", "alpha beta at gmail dot com", "hmm", "no"):
        result = workflow.process_turn(reply)
        assert "severe or life-threatening" not in result.response_text
        assert result.next_question is not None
        assert result.next_question[0] in ("contact_email", "confirmation")
        # Follow-up consent is unchanged by every email turn.
        assert engine.case.consent_to_contact is True

    # Only after the email is declined/confirmed does severity follow.
    assert engine.case.contact_email_declined or engine.case.pending_confirmation


def test_no_duplicate_email_question_in_one_response():
    engine, workflow = _to_email_offer("em-single-prompt")

    offered = workflow.process_turn("yes")
    assert offered.response_text.count(EMAIL_REQUEST_QUESTION) == 1

    captured = workflow.process_turn("alpha beta at gmail dot com")
    assert captured.response_text.count(EMAIL_REQUEST_QUESTION) <= 1
    assert captured.response_text.count("spell out your email address") <= 1
