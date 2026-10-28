"""Regression tests for the latest AE voice transcript (Hindi name extraction,
name-vs-DOB correction, medication confirmations, speech-service failure).

Observed defects:
- ``मेरा नाम है राजू रस्तोगी और मेरा जन्मदिन है पच्चीस मे दो हजार चार``
  stored the name as ``है राजू रस्तोगी`` (the ``मेरा नाम है`` lead was not
  stripped),
- at the name-or-DOB choice ``नाम गलत था`` was stored AS the name
  (``गलत था``) instead of selecting the name field, and the corrected name
  ``मेरा सही नाम राजू रस्तोगी है`` was classed ambiguous,
- at medication verification ``हाँ वही है`` was rejected as ambiguous and
  the question looped.

Pinned here: Hindi name-intro leads (``मेरा नाम है``/``मेरा नाम``/``नाम है``/
glued ``नामहै``) with DOB extracted independently; the Hindi field-choice
phrases never becoming name values (DOB untouched); the corrected name
replacing only the name field; Hindi medication affirmatives advancing while
negative/ambiguous replies never confirm; a Prisma 403 "Access denied"
surfacing as the clear HTTP 500 while the session's captured state survives
and later turns continue; and the English name/DOB/correction/consent/urgent
regressions staying green.

Synthetic data and mocked services only — no live Gnani/Prisma/Timbre calls.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.answer import AnswerLLMProvider, _parse_answer, _parse_for_pending_field
from app.services.prisma import PrismaAPIError
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)

DESCRIPTION = "I took Lantus and have a headache."
FULL_HINDI_NAME_DOB = (
    "मेरा नाम है राजू रस्तोगी और मेरा जन्मदिन है पच्चीस मे दो हजार चार"
)
CORRECTED_NAME_REPLY = "मेरा सही नाम राजेश रस्तोगी है"
ACCESS_DENIED_HINT = "Access denied. Check the API key and the account's available credits."


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    return engine, workflow


def _at_readback(case_id: str):
    """Drive to the combined name/DOB read-back (candidate held)."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn(DESCRIPTION)
    workflow.process_turn("हाँ वही है")   # medication confirmation
    workflow.process_turn("मैं हूँ")        # identity: self branch
    result = workflow.process_turn(FULL_HINDI_NAME_DOB)
    assert engine.case.pending_confirmation == "patient_identifier"
    assert engine.case.patient_age == "25-May-2004"
    return engine, workflow, result


# ---- 1. Hindi patient-name extraction ----------------------------------------


def test_hindi_name_introduction_phrases_are_stripped():
    for text in (
        "मेरा नाम है राजू रस्तोगी",
        "मेरा नाम राजू रस्तोगी है",
        "नाम है राजू रस्तोगी",
        "मेरा नामहै राजू रस्तोगी",   # glued STT variant
        "जी मेरा नाम राजू रस्तोगी",
    ):
        pending = _parse_for_pending_field("patient_identifier", text)
        assert pending.get("patient_identifier") == "राजू रस्तोगी", text


def test_bare_nam_is_never_stored_as_a_name():
    assert _parse_for_pending_field("patient_identifier", "नाम") == {}


def test_full_hindi_sentence_extracts_name_and_dob_independently():
    name = _parse_for_pending_field("patient_identifier", FULL_HINDI_NAME_DOB)
    dob = _parse_answer(FULL_HINDI_NAME_DOB, allow_bare_dob=True)
    assert name.get("patient_identifier") == "राजू रस्तोगी"
    assert dob.get("patient_age") == "25-May-2004"
    assert "जन्मदिन" not in name.get("patient_identifier", "")


# ---- 2. name-vs-DOB correction flow ------------------------------------------


def test_hindi_name_choice_phrases_select_the_name_field_only():
    for case_id, phrase in (
        ("AE-ch-1", "नाम गलत था"),
        ("AE-ch-2", "नाम"),
        ("AE-ch-3", "नाम गलत है"),
        ("AE-ch-4", "नाम गलत था मेरा"),
    ):
        engine, workflow, _ = _at_readback(case_id)
        rejected = workflow.process_turn("no")
        assert rejected.next_question is not None
        assert "name or the date of birth" in rejected.next_question[1].lower()

        choice = workflow.process_turn(phrase)

        # The phrase names the FIELD: it is never stored as a value, the old
        # candidate survives, the DOB is untouched, and the corrected name is
        # requested next.
        assert engine.case.patient_identifier is None, phrase
        assert engine.case.confirmation_candidate == {
            "patient_identifier": "राजू रस्तोगी"
        }, phrase
        assert engine.case.patient_age == "25-May-2004", phrase
        assert choice.next_question is not None, phrase
        assert choice.next_question[0] == "patient_identifier", phrase
        assert "first and last name" in choice.next_question[1].lower(), phrase
        assert "गलत था" not in str(engine.case.confirmation_candidate), phrase


def test_corrected_name_replaces_only_the_name_field():
    engine, workflow, _ = _at_readback("AE-ch-replace")
    workflow.process_turn("no")
    workflow.process_turn("नाम गलत था")           # choose the name field

    corrected = workflow.process_turn(CORRECTED_NAME_REPLY)

    # Only the name candidate changes; the date stays exactly as captured.
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "राजेश रस्तोगी"
    }
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.patient_identifier is None  # still uncommitted
    assert "राजेश रस्तोगी, 25-May-2004" in corrected.response_text

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "राजेश रस्तोगी"
    assert engine.case.patient_age == "25-May-2004"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_ambiguous_name_correction_clarifies_without_overwriting():
    engine, workflow, _ = _at_readback("AE-ch-ambiguous")
    workflow.process_turn("no")
    workflow.process_turn("नाम गलत था")

    result = workflow.process_turn("अच्छा")   # single-word mumbler

    # Clarified instead of overwriting: candidate and DOB are untouched.
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "राजू रस्तोगी"
    }
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.patient_identifier is None
    assert result.next_question is not None
    assert "name or the date of birth" in result.response_text.lower()


def test_english_name_choice_and_dob_correction_stay_intact():
    # English analogue: rejection -> field choice -> corrected value.
    engine = AEStateEngine.create_case("AE-english-choice")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn(DESCRIPTION)
    workflow.process_turn("yes")
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Jones 25/05/2004")
    workflow.process_turn("no")
    chosen = workflow.process_turn("the date")     # DOB is the wrong field
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Jones"}

    workflow.process_turn("26/05/2004")            # corrected date
    assert engine.case.patient_age == "26-May-2004"
    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.case.patient_age == "26-May-2004"
    assert chosen.next_question is not None


# ---- 3. medication confirmation ----------------------------------------------


def test_hindi_medication_affirmatives_advance():
    for case_id, phrase in (
        ("AE-med-vahi", "हाँ वही है"),
        ("AE-med-sahi", "हाँ सही है"),
        ("AE-med-jihaan", "जी हाँ"),
        ("AE-med-bilkul", "बिल्कुल"),
    ):
        engine, workflow = _workflow(case_id)
        workflow.process_turn(DESCRIPTION)
        result = workflow.process_turn(phrase)
        assert result.next_question is not None, phrase
        assert result.next_question[0] == "reporter_role", phrase
        assert "Just to make sure" not in result.response_text, phrase
        assert engine.case.product == "Lantus SoloStar", phrase


def test_negative_and_contradictory_medication_replies_never_confirm():
    for case_id, phrase in (
        ("AE-med-nahi", "नहीं"),
        ("AE-med-nahi-galat", "नही, यह गलत है"),
        ("AE-med-contradict", "नहीं वही नहीं है"),
    ):
        engine, workflow = _workflow(case_id)
        workflow.process_turn(DESCRIPTION)
        result = workflow.process_turn(phrase)
        assert result.next_question is not None, phrase
        assert result.next_question[0] == "product", phrase
        assert engine.current_state.value == "PRODUCT_IDENTIFIED", phrase


def test_ambiguous_medication_reply_reasks_verification_only():
    engine, workflow = _workflow("AE-med-shayad")
    workflow.process_turn(DESCRIPTION)
    result = workflow.process_turn("शायद वही है")
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    assert engine.case.product == "Lantus SoloStar"
    assert result.completed is False


# ---- 4. speech-service failure preserves workflow state ----------------------


class FailingSTT:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def transcribe(self, audio_path, language_code="en-IN"):
        raise self._exc


class FakeSTT:
    def transcribe(self, audio_path, language_code="en-IN"):
        return SimpleNamespace(
            text="I took Lantus and have a headache.",
            detected_language=None,
            timestamp=None,
        )


class FailingTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", **kwargs):
        raise PrismaAPIError(ACCESS_DENIED_HINT, status_code=403)


def _create_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


def _voice_turn(client: TestClient, session_id: str):
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "turn.wav"
        _create_wav(wav)
        with open(wav, "rb") as handle:
            return client.post(
                "/voice/conversation",
                files={"audio_file": ("turn.wav", handle, "audio/wav")},
                data={"session_id": session_id},
            )


def test_stt_access_denied_surfaces_clear_error_and_preserves_state():
    app = create_app(
        prisma=FailingSTT(PrismaAPIError("Invalid API key.", status_code=403)),
        timbre=object(),
    )
    client = TestClient(app)
    session = "sess-stt-denied"

    # Captured before the failure: a typed turn (no STT involved).
    first = client.post(
        "/text/conversation",
        data={"session_id": session, "message": DESCRIPTION},
    )
    assert first.status_code == 200
    assert first.json()["ae"]["collected_information"]["adverse_event"] == DESCRIPTION

    # The denied voice turn: clear account/configuration error, no state change.
    denied = _voice_turn(client, session)
    assert denied.status_code == 500
    assert ACCESS_DENIED_HINT in denied.json()["detail"]

    # Captured information survives and the conversation continues normally.
    after = client.post(
        "/text/conversation",
        data={"session_id": session, "message": "yes"},
    )
    assert after.status_code == 200
    body = after.json()
    assert body["ae"]["collected_information"]["adverse_event"] == DESCRIPTION
    assert body["ae"]["collected_information"]["product"] == "Lantus SoloStar"
    assert body["safety_status"] == "ok"
    assert body["completed"] is False
    assert body["current_state"] != "HUMAN_REVIEW"


def test_tts_access_denied_keeps_the_applied_turn_and_captured_data():
    app = create_app(prisma=FakeSTT(), timbre=FailingTTS())
    client = TestClient(app)
    session = "sess-tts-denied"

    failed = _voice_turn(client, session)
    assert failed.status_code == 502
    assert "Gnani service failure" in failed.json()["detail"]

    # The transcribed turn was applied (state not corrupted, nothing
    # discarded): a later text turn sees the captured event and continues.
    follow = client.post(
        "/text/conversation",
        data={"session_id": session, "message": "yes"},
    )
    assert follow.status_code == 200
    body = follow.json()
    assert body["ae"]["collected_information"]["adverse_event"] == DESCRIPTION
    assert body["safety_status"] == "ok"
    assert body["completed"] is False
