"""End-to-end AE voice conversation tests through the real HTTP endpoint.

Each turn is a genuine ``POST /voice/conversation`` against the production
router (the one behind ``app.main:app``) with injected offline STT and
recording TTS stand-ins: no database, speech service or hosted model is ever
contacted, and every transcript is synthetic.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore

# ---- synthetic scripted journeys (fictional names and numbers only) ---------

SELF_SESSION_ID = "e2e-self-consent-yes"
REPORTER_SESSION_ID = "e2e-reporter-urgent"

# Self-reporter, consent given, event not severe, email collected.
SELF_SCRIPT = [
    "Hello",
    "I took Lantus yesterday and since this morning I have a severe headache",
    "yes",  # Step 4: medication verified
    "myself",  # Step 5
    "Alpha Jones, 05-01-1990",  # Step 6B.1 (name and date of birth)
    "yes",  # Step 6B.2: combined read-back confirmed
    "9886094321",  # Step 6B.3
    "yes",  # Step 6B.4: contact read-back confirmed
    "Yes, you can contact me",  # Step 7 consent
    "yes",  # Step 8: email offered
    "alpha@example.com",  # Step 8: email requested
    "yes",  # email read-back confirmed
    "No, it was not severe",  # Step 9
    "It started yesterday",  # Step 10B.1 -> Step 10B.2 asked
    "no",  # Step 10B.2 declined -> report and closing (no outcome question)
]

SELF_EXPECTED_QUESTIONS = [
    "adverse_event",  # Step 2
    "product_verification",  # Step 4
    "reporter_role",  # Step 5
    "patient_identifier",  # Step 6B.1
    "confirmation",  # Step 6B.2
    "contact_phone",  # Step 6B.3
    "confirmation",  # Step 6B.4
    "consent_to_contact",  # Step 7
    "contact_email",  # Step 8 offer
    "contact_email",  # Step 8 request
    "confirmation",  # email read-back
    "seriousness",  # Step 9
    "onset_date",  # Step 10B.1
    None,  # Step 10B.2 is asked in the response; next_question stays null
    None,  # report presented; no post-report question
]

# Reporter branch (someone else), consent declined, event severe.
REPORTER_SCRIPT = [
    "I want to report an adverse event: my mother had a rash after taking Lantus",
    "yes",  # Step 4: medication verified
    "for my mother",  # Step 5
    "Dr. Observer",  # Step 6A.1
    "yes",  # Step 6A.2: reporter name read-back confirmed
    "9999990001",  # Step 6A.3
    "yes",  # Step 6A.4: reporter contact read-back confirmed
    "Mary Jones, 12-03-1975",  # Step 6A.5
    "yes",  # Step 6A.6: combined patient read-back confirmed
    "9886094399",  # Step 6A.7
    "yes",  # Step 6A.8: patient contact read-back confirmed
    "no thanks",  # Step 7 consent declined (Step 8 skipped)
    "Yes, it was severe",  # Step 9 -> Step 10A urgent branch
]

REPORTER_EXPECTED_QUESTIONS = [
    "product_verification",  # Step 4
    "reporter_role",  # Step 5
    "reporter_name",  # Step 6A.1
    "confirmation",  # Step 6A.2
    "reporter_contact_phone",  # Step 6A.3
    "confirmation",  # Step 6A.4
    "patient_identifier",  # Step 6A.5
    "confirmation",  # Step 6A.6
    "contact_phone",  # Step 6A.7
    "confirmation",  # Step 6A.8
    "consent_to_contact",  # Step 7
    "seriousness",  # Step 8 skipped; Step 9 asked directly
    None,  # Step 10A urgent branch: no further questions
]

# A minimal but valid RIFF/WAVE header (16-bit mono PCM, no frames) so the
# audio assertions exercise the same shape a real adapter would return.
_FAKE_WAV = (
    b"RIFF"
    + (36).to_bytes(4, "little")
    + b"WAVE"
    + b"fmt "
    + (16).to_bytes(4, "little")
    + b"\x01\x00"  # PCM
    + b"\x01\x00"  # mono
    + (16000).to_bytes(4, "little")
    + (32000).to_bytes(4, "little")
    + b"\x02\x04"  # block align, bits per sample
    + b"data"
    + (0).to_bytes(4, "little")
)


class QueueSTT:
    """Offline STT stand-in replaying one synthetic transcript per turn."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    def transcribe(self, audio_path, language_code="en-IN"):
        if not self._transcripts:
            raise AssertionError("Unexpected extra turn: transcript queue empty")
        return SimpleNamespace(text=self._transcripts.pop(0))


class RecordingWavTTS:
    """Offline TTS stand-in that records every synthesis and writes a WAV."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        self.calls.append({"text": text, "language": language, "voice": voice})
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_FAKE_WAV)
        return path


def _client(transcripts: list[str], tts: RecordingWavTTS) -> TestClient:
    router = create_voice_conversation_router(
        stt=QueueSTT(transcripts),
        tts=tts,
        session_store=SessionStore(),
        llm_provider=AnswerLLMProvider(),
        ask_additional_information=True,
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


def _assert_valid_voice_audio(body: dict) -> None:
    """Every assistant voice turn must carry spoken text and a real WAV file."""
    assert body["audio_path"], "voice turns must return an audio_path"
    assert body["spoken_response_text"], "spoken text must not be empty"
    audio = Path(body["audio_path"])
    assert audio.is_file(), f"audio file missing: {audio}"
    data = audio.read_bytes()
    assert len(data) > 0
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _assert_no_submission_claim(text: str) -> None:
    lowered = text.lower()
    assert "submitted" not in lowered
    assert "sanofi" not in lowered
    assert "regulator" not in lowered


def _run_journey(
    session_id: str, script: list[str], expected_questions: list
) -> tuple[list[dict], RecordingWavTTS]:
    tts = RecordingWavTTS()
    client = _client(list(script), tts)

    bodies = []
    for transcript in script:
        body = _turn(client, session_id)
        bodies.append(body)
        # One session across successive HTTP requests.
        assert body["session_id"] == session_id
        assert body["workflow"] == "ae"
        assert body["case_id"] == bodies[0]["case_id"]
        # Language metadata and spoken audio on every turn.
        assert body["conversation_language"] == "en-IN"
        assert body["language_translated"] is False
        assert body["spoken_response_text"] == body["response_text"]
        _assert_valid_voice_audio(body)
        _assert_no_submission_claim(body["response_text"])

    assert [body["next_question"][0] if body["next_question"] else None
            for body in bodies] == expected_questions
    assert len(tts.calls) == len(script)
    return bodies, tts


# ---- journey 1: self-reporter, consent yes ----------------------------------


def test_self_reporter_consent_yes_completes_the_full_journey():
    bodies, tts = _run_journey(
        SELF_SESSION_ID, SELF_SCRIPT, SELF_EXPECTED_QUESTIONS
    )
    final = bodies[-1]

    # The report is completed and the closing message reflects the consent.
    assert final["completed"] is True
    assert final["current_state"] == "CASE_COMPLETED"
    assert final["human_review_required"] is False
    assert final["missing_information"] == []
    assert final["response_text"].endswith(
        "Thank you for sharing this information with us. "
        "We've prepared the report for review. "
        "If you provided consent for follow-up contact, the appropriate team "
        "may contact you if needed."
    )

    collected = final["ae"]["collected_information"]
    assert collected["product"] == "Lantus SoloStar"
    assert collected["adverse_event"] == (
        "I took Lantus yesterday and since this morning I have a severe headache"
    )
    assert collected["reporter_role"] == "self"
    assert collected["patient_identifier"] == "Alpha Jones"
    assert collected["patient_age"] == "05-Jan-1990"
    assert collected["contact_phone"] == "9886094321"
    assert collected["contact_email"] == "alpha@example.com"
    assert collected["contact_email_declined"] is False
    assert collected["consent_to_contact"] is True
    assert collected["seriousness"] == "no"
    assert collected["onset_date"] == "It started yesterday"
    # No outcome was volunteered, so the field stays unset (the outcome is no
    # longer an interview question).
    assert collected["outcome"] is None
    assert collected["additional_information_status"] == "declined"

    # Confirmation wording appeared exactly once per confirmed value.
    spoken = "\n".join(body["response_text"] for body in bodies)
    assert spoken.count("I have your details as Alpha Jones, 05-Jan-1990.") == 1
    assert spoken.count("I have a 10-digit number ending in 4321.") == 1
    assert spoken.count("I heard your email address as alpha@example.com.") == 1
    # The patient contact number is never asked again after it is confirmed.
    assert spoken.count("best contact number for you") == 1

    # Every synthesis used the English defaults.
    assert all(call["language"] == "en-IN" for call in tts.calls)


# ---- journey 2: reporter branch, consent no, urgent severity ----------------


def test_reporter_branch_with_declined_consent_and_urgent_severity():
    bodies, _ = _run_journey(
        REPORTER_SESSION_ID, REPORTER_SCRIPT, REPORTER_EXPECTED_QUESTIONS
    )
    final = bodies[-1]

    # Step 10A: urgent escalation outranks routine intake and needs review.
    assert final["safety_status"] == "urgent_life_threatening"
    assert final["human_review_required"] is True
    assert final["priority"] == "high"
    assert final["current_state"] == "HUMAN_REVIEW"
    assert final["completed"] is False
    assert final["next_question"] is None
    assert "immediate medical attention" in final["response_text"].lower()

    # Declined consent: no email question anywhere, and the closing never
    # implies that follow-up contact was consented to.
    after_consent = "\n".join(body["response_text"] for body in bodies[11:])
    assert "email address" not in after_consent
    assert (
        "If you provided consent for follow-up contact"
        not in final["response_text"]
    )
    assert final["response_text"].endswith(
        "Thank you for sharing this information with us. "
        "We've prepared the report for review."
    )

    # Urgent handling skipped the incident date and additional information.
    for body in bodies:
        assert body["next_question"] is None or body["next_question"][0] != "onset_date"
    collected = final["ae"]["collected_information"]
    assert collected["onset_date"] is None
    assert collected["additional_information"] is None
    assert collected["additional_information_status"] is None

    # Reporter and patient identities and contact numbers stay distinct.
    assert collected["reporter_role"] == "someone_else"
    assert collected["reporter_name"] == "Dr. Observer"
    assert collected["reporter_contact_phone"] == "9999990001"
    assert collected["patient_identifier"] == "Mary Jones"
    assert collected["patient_age"] == "12-Mar-1975"
    assert collected["contact_phone"] == "9886094399"
    assert collected["consent_to_contact"] is False
    assert collected["seriousness"] == "yes"
    assert collected["product"] == "Lantus SoloStar"

    # Branch-specific confirmation wording appeared exactly once each.
    spoken = "\n".join(body["response_text"] for body in bodies)
    assert spoken.count("I have your name as Dr. Observer.") == 1
    assert spoken.count("I have a 10-digit number ending in 0001.") == 1
    assert spoken.count(
        "Let me read that back to make sure I've recorded it correctly: "
        "Mary Jones, 12-Mar-1975."
    ) == 1
    assert spoken.count("I have a 10-digit number ending in 4399.") == 1
    # Consent was asked once; the patient contact number was asked once.
    assert spoken.count("Would it be okay for the appropriate team") == 1
    assert spoken.count("patient's contact number") == 1
