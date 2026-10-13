"""Integration tests for patient-name confirmation via the voice endpoint.

Each test drives the actual conversation entry point (``POST
/voice/conversation``) through ``create_voice_conversation_router`` with
injected offline STT/TTS stand-ins: no database, speech service, or hosted
model is contacted. Separate HTTP requests share one in-memory
``SessionStore``, so session state (including a pending confirmation)
persists across turns exactly as it does in production.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.main import FastAPI
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)

REPORT = (
    "I took Lantus yesterday and today I have severe headache and vomiting. "
    "I want to report this adverse event."
)
REPORTER_ROLE = "I am the patient"
PATIENT_NAME = "Alpha Jones"
CORRECTED_NAME = "Beta Jones"
READBACK = f"I heard {PATIENT_NAME}. Is that correct?"


class QueueSTT:
    """Offline STT stand-in replaying one scripted transcript per turn."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    def transcribe(self, audio_path, language_code="en-IN"):
        if not self._transcripts:
            raise AssertionError("Unexpected extra turn: transcript queue empty")
        return SimpleNamespace(text=self._transcripts.pop(0))


class FakeTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _build_client(transcripts: list[str]) -> TestClient:
    router = create_voice_conversation_router(
        stt=QueueSTT(transcripts),
        tts=FakeTTS(),
        session_store=SessionStore(),
        llm_provider=AnswerLLMProvider(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _turn(client: TestClient, session_id: str) -> dict:
    resp = client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": "en-IN"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _run_to_name_readback(
    session_id: str, replies: list[str]
) -> tuple[TestClient, dict]:
    client = _build_client([REPORT, REPORTER_ROLE, PATIENT_NAME, *replies])
    _turn(client, session_id)
    _turn(client, session_id)
    return client, _turn(client, session_id)


def test_name_capture_returns_readback_and_blocks_advancement():
    client, data = _run_to_name_readback("confirm-readback", [])

    assert data["workflow"] == "ae"
    assert READBACK in data["response_text"]
    assert data["next_question"] == ["confirmation", READBACK]
    # The candidate was extracted but not collected: no premature advance.
    assert data["current_state"] == "EVENT_IDENTIFIED"
    assert data["completed"] is False
    assert "patient_identifier" in data["missing_information"]
    assert data["ae"]["extracted_information"]["patient_identifier"] == PATIENT_NAME
    assert data["ae"]["collected_information"]["patient_identifier"] is None


def test_yes_commits_confirmed_name_and_advances_across_requests():
    client, _ = _run_to_name_readback("confirm-yes", ["yes"])

    # A separate HTTP request in the same session answers the read-back.
    data = _turn(client, "confirm-yes")

    assert data["ae"]["collected_information"]["patient_identifier"] == PATIENT_NAME
    assert "patient_identifier" not in data["missing_information"]
    assert data["current_state"] == "REPORTER_IDENTIFIED"
    assert data["next_question"] is not None
    assert data["next_question"][0] == "patient_age"
    assert data["completed"] is False


def test_no_then_correction_is_read_back_again_before_advancing():
    client, _ = _run_to_name_readback(
        "confirm-correct", ["no", CORRECTED_NAME, "correct"]
    )

    rejected = _turn(client, "confirm-correct")
    assert "May I have your first and last name?" in rejected["response_text"]
    assert rejected["next_question"] is not None
    assert rejected["next_question"][0] == "patient_identifier"
    assert rejected["current_state"] == "EVENT_IDENTIFIED"
    assert rejected["ae"]["collected_information"]["patient_identifier"] is None

    corrected = _turn(client, "confirm-correct")
    assert f"I heard {CORRECTED_NAME}. Is that correct?" in corrected["response_text"]
    assert corrected["next_question"] == [
        "confirmation",
        f"I heard {CORRECTED_NAME}. Is that correct?",
    ]
    assert corrected["ae"]["collected_information"]["patient_identifier"] is None

    confirmed = _turn(client, "confirm-correct")
    assert (
        confirmed["ae"]["collected_information"]["patient_identifier"]
        == CORRECTED_NAME
    )
    assert confirmed["current_state"] == "REPORTER_IDENTIFIED"
    assert confirmed["next_question"] is not None
    assert confirmed["next_question"][0] == "patient_age"


def test_ambiguous_reply_clarifies_without_advancing():
    client, _ = _run_to_name_readback("confirm-ambiguous", ["I'm not sure"])

    data = _turn(client, "confirm-ambiguous")

    assert "please say yes" in data["response_text"]
    # The read-back is re-asked so the turn cannot pass the name.
    assert READBACK in data["response_text"]
    assert data["next_question"] == ["confirmation", READBACK]
    assert data["current_state"] == "EVENT_IDENTIFIED"
    assert data["completed"] is False
    assert "patient_identifier" in data["missing_information"]
    assert data["ae"]["collected_information"]["patient_identifier"] is None


def test_only_integrated_fields_confirm_and_the_rest_capture_legacy_style():
    # Phone contact after the confirmed name must keep legacy capture: no
    # read-back, direct commit, workflow keeps its normal next question.
    transcripts = [
        REPORT,
        REPORTER_ROLE,
        PATIENT_NAME,
        "yes",
        "45 years old",
        "Yes, I consent",
        "You can reach me by phone",
        "9999990000",
    ]
    client = _build_client(transcripts)
    session_id = "confirm-scope"

    for _ in range(7):
        data = _turn(client, session_id)
    phone_turn = _turn(client, session_id)

    assert phone_turn["ae"]["collected_information"]["contact_phone"] == "9999990000"
    assert phone_turn["next_question"] is not None
    assert phone_turn["next_question"][0] == "onset_date"
    assert "I heard" not in phone_turn["response_text"]
    assert phone_turn["current_state"] == "CONTACT_CAPTURED"

    # The name integrated above was confirmed along the way.
    assert data["ae"]["collected_information"]["patient_identifier"] == PATIENT_NAME


def test_default_workflow_constructor_keeps_legacy_name_capture():
    engine = AEStateEngine.create_case("legacy-name-check")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn(REPORT)
    workflow.process_turn(REPORTER_ROLE)
    result = workflow.process_turn(PATIENT_NAME)

    assert engine.case.patient_identifier == PATIENT_NAME
    assert engine.case.pending_confirmation is None
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    assert "I heard" not in result.response_text


# ---- filler stripping in corrections ---------------------------------------


def test_filler_is_stripped_from_the_corrected_name():
    transcripts = [
        REPORT,
        REPORTER_ROLE,
        PATIENT_NAME,
        "no",
        "no, only alpha beta",
        "yes",
    ]
    client = _build_client(transcripts)
    session_id = "confirm-filler"

    for _ in range(4):  # report, role, name read-back, "no"
        _turn(client, session_id)
    correction = _turn(client, session_id)

    # "only" is filler: the read-back carries the clean value, not the filler.
    assert "I heard alpha beta. Is that correct?" in correction["response_text"]
    assert "only alpha beta" not in correction["response_text"]
    assert correction["next_question"] == [
        "confirmation",
        "I heard alpha beta. Is that correct?",
    ]
    # The corrected value is still only a candidate until confirmed.
    assert correction["ae"]["collected_information"]["patient_identifier"] is None

    confirmed = _turn(client, session_id)
    assert confirmed["ae"]["collected_information"]["patient_identifier"] == "alpha beta"
    assert confirmed["current_state"] == "REPORTER_IDENTIFIED"


def test_affirmative_with_a_new_value_clarifies_instead_of_silently_accepting():
    client, _ = _run_to_name_readback("confirm-affirm-value", ["yes alpha beta"])
    data = _turn(client, "confirm-affirm-value")

    assert "please say yes" in data["response_text"]
    # The original read-back is re-asked: the old value is neither silently
    # committed nor silently replaced.
    assert READBACK in data["response_text"]
    assert data["ae"]["collected_information"]["patient_identifier"] is None
    assert data["current_state"] == "EVENT_IDENTIFIED"
    assert data["completed"] is False


# ---- bounded confirmation failures -----------------------------------------


def test_repeated_failed_confirmations_park_the_report_for_human_review():
    transcripts = [
        REPORT,
        REPORTER_ROLE,
        PATIENT_NAME,
        "I'm not sure",  # attempt 1 -> clarification
        "maybe",  # attempt 2 -> clarification
        "not sure",  # attempt 3 -> human review
    ]
    client = _build_client(transcripts)
    session_id = "confirm-review"

    for _ in range(5):
        data = _turn(client, session_id)
    # Attempts 1 and 2 still clarify and keep asking.
    assert "please say yes" in data["response_text"]
    assert data["current_state"] == "EVENT_IDENTIFIED"
    assert data["human_review_required"] is False

    final = _turn(client, session_id)
    assert final["human_review_required"] is True
    assert final["current_state"] == "HUMAN_REVIEW"
    assert final["completed"] is False
    assert final["priority"] == "normal"  # review, not an urgency escalation
    assert final["next_question"] is None

    # Nothing was discarded, invented, or falsely completed.
    collected = final["ae"]["collected_information"]
    assert collected["patient_identifier"] is None
    assert "patient_identifier" in final["missing_information"]
    assert collected["adverse_event"]
    assert collected["product"]

    # The person is told plainly what happens next; no endless clarification.
    assert "reviewed by a member of our team" in final["response_text"]
    assert "please say yes" not in final["response_text"]


def test_contact_confirmation_uses_the_same_attempt_bound():
    engine = AEStateEngine.create_case("contact-confirm-bound-demo")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "45")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    assert engine.next_field()[0] == "contact_phone"

    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=CONFIRMATION_FIELDS
    )
    readback = workflow.process_turn("9999990000")
    assert "I heard your phone number as 9999990000" in readback.response_text
    assert engine.case.contact_phone is None  # candidate only, not committed

    first = workflow.process_turn("not sure")
    assert "please say yes" in first.response_text
    second = workflow.process_turn("maybe")
    assert "please say yes" in second.response_text
    third = workflow.process_turn("I don't know")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert third.human_review_required is True
    assert third.completed is False
    assert third.priority == "normal"
    assert "contact_phone" in third.missing_information
    # No contact detail was invented, consent was not weakened, and the
    # previously collected report data is preserved.
    assert engine.case.contact_phone is None
    assert engine.case.consent_to_contact is True
    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.case.adverse_event == "rash"
