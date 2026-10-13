"""Focused tests for the shared-turn text conversation endpoint (AE Text).

POST /text/conversation must follow the same authoritative workflow as the
voice endpoint: same transcript routing, ConversationalAEWorkflow state,
safety envelope, name read-back confirmation and consent behaviour — with
STT/TTS bypassed (``audio_path`` is always null).
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from app.main import create_app

    return TestClient(create_app())


def test_text_conversation_validation():
    client = _client()
    missing_message = client.post("/text/conversation", data={"session_id": "t-val"})
    assert missing_message.status_code == 422

    missing_session = client.post("/text/conversation", data={"message": "hello"})
    assert missing_session.status_code == 422

    empty_message = client.post(
        "/text/conversation", data={"session_id": "t-val", "message": "   "}
    )
    assert empty_message.status_code == 400


def test_text_turn_runs_backend_ae_workflow_without_audio():
    client = _client()
    resp = client.post(
        "/text/conversation",
        data={"session_id": "t-ae-1", "message": "I took Lantus and have a headache."},
    )
    assert resp.status_code == 200
    body = resp.json()

    # Same envelope as the voice endpoint, minus audio.
    for key in (
        "session_id",
        "transcript",
        "intent",
        "workflow",
        "response_text",
        "audio_path",
        "current_state",
        "case_id",
        "priority",
        "human_review_required",
        "completed",
        "missing_information",
        "next_question",
        "safety_status",
        "mi",
        "ae",
    ):
        assert key in body, key

    assert body["workflow"] == "ae"
    assert body["audio_path"] is None
    assert body["safety_status"] == "ok"
    assert body["human_review_required"] is False

    collected = body["ae"]["collected_information"]
    assert collected["product"] == "Lantus SoloStar"
    assert "headache" in (collected["adverse_event"] or "")
    assert "reporter_role" in (body["missing_information"] or [])
    response = body["response_text"].lower()
    assert "physician" in response or "healthcare professional" in response


def test_text_turns_share_session_state_like_voice():
    client = _client()
    first = client.post(
        "/text/conversation",
        data={"session_id": "t-ae-2", "message": "I took Lantus and have a headache."},
    ).json()
    second = client.post(
        "/text/conversation",
        data={"session_id": "t-ae-2", "message": "myself"},
    ).json()

    assert second["case_id"] == first["case_id"]
    assert second["ae"]["collected_information"]["reporter_role"] == "self"
    assert "reporter_role" not in (second["missing_information"] or [])
    assert "patient_identifier" in (second["missing_information"] or [])
    assert second["next_question"][0] == "patient_identifier"


def test_text_consent_flow_uses_same_confirmation_and_consent_rules():
    client = _client()
    session_id = "t-consent"

    def turn(message: str) -> dict:
        resp = client.post(
            "/text/conversation",
            data={"session_id": session_id, "message": message},
        )
        assert resp.status_code == 200
        return resp.json()

    turn("I took Lantus and have a headache.")  # product + event captured
    turn("myself")  # reporter is the patient

    named = turn("Demo Patient")  # name is read back before it is committed
    assert "I heard Demo Patient" in named["response_text"]

    confirmed = turn("yes")
    assert (
        confirmed["ae"]["collected_information"]["patient_identifier"]
        == "Demo Patient"
    )

    turn("45")  # age
    consented = turn("yes")  # consent to contact
    assert consented["ae"]["collected_information"]["consent_to_contact"] is True


def test_text_urgent_turn_escalates_and_preserves_description():
    client = _client()
    session_id = "t-urgent"
    client.post(
        "/text/conversation",
        data={"session_id": session_id, "message": "I took Lantus and have a headache."},
    )
    resp = client.post(
        "/text/conversation",
        data={
            "session_id": session_id,
            "message": "Now I am having a severe heart attack",
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["safety_status"] == "urgent_life_threatening"
    assert body["human_review_required"] is True
    assert body["priority"] == "high"
    assert body["current_state"] == "HUMAN_REVIEW"

    collected = body["ae"]["collected_information"]
    assert "headache" in (collected["adverse_event"] or "")
    assert "severe heart attack" in (collected["adverse_event"] or "")


# ---- language mirroring for text turns ---------------------------------------


def test_hindi_text_turn_returns_translated_metadata():
    client = _client()
    resp = client.post(
        "/text/conversation",
        data={"session_id": "t-hi-text", "message": "मुझे कल से सिर में दर्द है।"},
    )
    assert resp.status_code == 200
    body = resp.json()

    # Text turns carry the same translated form as voice turns (no audio).
    assert body["conversation_language"] == "hi-IN"
    assert body["language_translated"] is True
    assert body["spoken_response_text"] != body["response_text"]
    assert body["audio_path"] is None
    assert body["voice"] is None


def test_english_text_turn_stays_english():
    client = _client()
    body = client.post(
        "/text/conversation",
        data={
            "session_id": "t-en-text",
            "message": "I took Lantus and have a headache.",
        },
    ).json()

    assert body["conversation_language"] == "en-IN"
    assert body["language_translated"] is False
    assert body["spoken_response_text"] == body["response_text"]


def test_urgent_hindi_text_reaches_safety_and_preserves_report_data():
    client = _client()
    first = client.post(
        "/text/conversation",
        data={
            "session_id": "t-hi-urgent",
            "message": "I took Lantus and have a headache.",
        },
    ).json()
    original_event = first["ae"]["collected_information"]["adverse_event"]
    assert original_event

    second = client.post(
        "/text/conversation",
        data={"session_id": "t-hi-urgent", "message": "मुझे दिल का दौरा हुआ"},
    ).json()

    # The Hindi urgent message reaches the existing safety layer unchanged.
    assert second["reason"] == "safety_urgent"
    assert second["safety_status"] == "urgent_life_threatening"
    assert second["human_review_required"] is True
    assert second["priority"] == "high"
    assert second["current_state"] == "HUMAN_REVIEW"
    # Collected report data is preserved; the urgent wording is appended.
    collected = second["ae"]["collected_information"]
    assert original_event in collected["adverse_event"]
    assert "दिल का दौरा" in collected["adverse_event"]
