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
    # Step 4: the recorded medication is verified before Step 5.
    verified = client.post(
        "/text/conversation",
        data={"session_id": "t-ae-2", "message": "yes"},
    ).json()
    second = client.post(
        "/text/conversation",
        data={"session_id": "t-ae-2", "message": "myself"},
    ).json()

    assert verified["next_question"][0] == "reporter_role"
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
    turn("yes")  # Step 4: medication verified
    turn("myself")  # reporter is the patient

    named = turn("Demo Patient")  # name held for the combined read-back
    assert named["ae"]["collected_information"]["patient_identifier"] is None

    aged = turn("27/05/2000")  # date of birth completes the details
    assert "I have your details as Demo Patient, 27-May-2000." in (
        aged["response_text"]
    )

    confirmed = turn("yes")  # Step 6B.2: patient details confirmed
    assert (
        confirmed["ae"]["collected_information"]["patient_identifier"]
        == "Demo Patient"
    )

    turn("9999990000")  # patient contact number (before consent)
    turn("yes")  # contact number confirmed
    consented = turn("yes")  # Step 7 consent to contact
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


# ---- complete text AE journey (no provider calls) ---------------------------


class _ForbiddenSTT:
    """STT spy: the text journey must never transcribe audio."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path, language_code="en-IN"):
        self.calls += 1
        raise AssertionError("STT must not be called on the text journey")


class _ForbiddenTTS:
    """TTS spy: the text journey must never synthesize speech."""

    def __init__(self) -> None:
        self.calls = 0

    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        self.calls += 1
        raise AssertionError("TTS must not be called on the text journey")


TEXT_JOURNEY = [
    "Hello",
    "I took Lantus yesterday and since this morning I have a severe headache",
    "yes",  # Step 4: medication verified
    "myself",  # Step 5
    "Alpha Jones, 05-01-1990",  # Step 6B.1 (name + date of birth)
    "yes",  # Step 6B.2
    "9886094321",  # Step 6B.3
    "yes",  # Step 6B.4
    "Yes, you can contact me",  # Step 7 consent
    "yes",  # Step 8: email offered
    "alpha@example.com",  # Step 8: email requested
    "yes",  # email read-back confirmed
    "No, it was not severe",  # Step 9
    "It started yesterday",  # Step 10B.1
    "no",  # Step 10B.2 declined -> report and closing
]

TEXT_JOURNEY_QUESTIONS = [
    "adverse_event",
    "product_verification",
    "reporter_role",
    "patient_identifier",
    "confirmation",
    "contact_phone",
    "confirmation",
    "consent_to_contact",
    "contact_email",
    "contact_email",
    "confirmation",
    "seriousness",
    "onset_date",
    None,  # Step 10B.2 is asked in the response; next_question stays null
    None,  # report prepared and closing message
]


def test_complete_text_ae_journey_runs_without_any_provider_call():
    from app.main import create_app

    stt = _ForbiddenSTT()
    tts = _ForbiddenTTS()
    client = TestClient(create_app(prisma=stt, timbre=tts))
    session_id = "t-full-journey"

    bodies = []
    for message in TEXT_JOURNEY:
        resp = client.post(
            "/text/conversation",
            data={"session_id": session_id, "message": message},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        bodies.append(body)
        # Text turns never produce audio.
        assert body["audio_path"] is None
        assert body["workflow"] == "ae"
        assert body["session_id"] == session_id
        # No submission or external-filing claim anywhere in the journey.
        assert "submitted" not in body["response_text"].lower()

    assert [
        b["next_question"][0] if b["next_question"] else None for b in bodies
    ] == TEXT_JOURNEY_QUESTIONS

    # The journey completes with the report prepared for human PV review.
    final = bodies[-1]
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

    # The structured report carries every collected field.
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

    # No external speech provider was contacted at any point.
    assert stt.calls == 0
    assert tts.calls == 0


# ---- approved text AE question sequence -------------------------------------


def _turn(client: TestClient, session_id: str, message: str, **form) -> dict:
    data = {"session_id": session_id, "message": message, **form}
    resp = client.post("/text/conversation", data=data)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_medication_question_follows_the_event_description_with_empathy():
    client = _client()
    session_id = "t-med-empathy"

    _turn(client, session_id, "Hello")
    described = _turn(client, session_id, "I have a headache")

    question = (
        "To help us record your report correctly, could you tell me the name "
        "of the medication involved?"
    )
    assert described["next_question"][0] == "product"
    assert question in described["response_text"]
    # The compassionate transition immediately precedes the question.
    assert described["response_text"].index(
        "I'm sorry to hear that you've experienced this."
    ) < described["response_text"].index(question)


def test_medication_verification_precedes_the_reporter_question():
    client = _client()
    session_id = "t-med-verify"

    _turn(client, session_id, "I have a headache")
    named = _turn(client, session_id, "Lantus")
    assert named["next_question"][0] == "product_verification"

    verified = _turn(client, session_id, "yes")
    assert verified["next_question"][0] == "reporter_role"


def test_out_of_scope_medication_terminates_routine_intake():
    client = _client()
    session_id = "t-out-of-scope"

    result = _turn(client, session_id, "I have a rash due to paracetamol")

    assert "configured for Lantus SoloStar only" in result["response_text"]
    assert "cannot continue this report" in result["response_text"]
    assert result["next_question"] is None
    assert result["completed"] is False


def test_reporter_branch_collects_required_fields_in_approved_order():
    client = _client()
    session_id = "t-reporter-branch"

    script = [
        "I want to report an adverse event: my mother had a rash after taking Lantus",
        "yes",  # Step 4
        "for my mother",  # Step 5
        "Dr. Observer",  # Step 6A.1
        "yes",  # Step 6A.2
        "9999990001",  # Step 6A.3
        "yes",  # Step 6A.4
        "Mary Jones, 12-03-1975",  # Step 6A.5
        "yes",  # Step 6A.6
        "9886094399",  # Step 6A.7
        "yes",  # Step 6A.8
    ]
    keys = []
    for message in script:
        body = _turn(client, session_id, message)
        keys.append(body["next_question"][0] if body["next_question"] else None)

    assert keys == [
        "product_verification",
        "reporter_role",
        "reporter_name",
        "confirmation",
        "reporter_contact_phone",
        "confirmation",
        "patient_identifier",
        "confirmation",
        "contact_phone",
        "confirmation",
        "consent_to_contact",
    ]
    # No unapproved question (e.g. medical care) appears anywhere.
    assert all(
        "medical care" not in body["response_text"].lower()
        for body in [_turn(client, session_id, "no thanks")]  # consent no
    )
    final = _turn(client, session_id, "Yes, it was severe")
    assert final["safety_status"] == "urgent_life_threatening"
    assert final["human_review_required"] is True
    assert "medical care" not in final["response_text"].lower()


def test_dob_question_requests_a_date_of_birth_not_an_age():
    client = _client()
    session_id = "t-dob-question"

    _turn(client, session_id, "I have a headache after taking Lantus")
    _turn(client, session_id, "yes")
    patient = _turn(client, session_id, "myself")

    question = patient["next_question"][1]
    assert "date of birth" in question
    assert "How old" not in question

    named = _turn(client, session_id, "Alpha Beta and my date of birth is 27/05/2000")
    assert named["next_question"][0] == "confirmation"
    assert (
        "I have your details as Alpha Beta, 27-May-2000. Is that correct?"
        in named["response_text"]
    )
    confirmed = _turn(client, session_id, "yes")
    assert (
        confirmed["ae"]["collected_information"]["patient_identifier"] == "Alpha Beta"
    )


def test_consent_yes_offers_email_and_consent_no_goes_straight_to_severity():
    # Consent Yes -> the email question.
    yes_client = _client()
    yes_id = "t-consent-yes"
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 27/05/2000",
        "yes",
        "9886094321",
        "yes",
    ):
        _turn(yes_client, yes_id, message)
    consented = _turn(yes_client, yes_id, "Yes, I consent")
    assert consented["next_question"][0] == "contact_email"

    # Consent No -> straight to severity, no email question.
    no_client = _client()
    no_id = "t-consent-no"
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 27/05/2000",
        "yes",
        "9886094321",
        "yes",
    ):
        _turn(no_client, no_id, message)
    declined = _turn(no_client, no_id, "no thanks")
    assert declined["next_question"][0] == "seriousness"
    assert "email" not in declined["response_text"].lower()
    assert declined["ae"]["collected_information"]["consent_to_contact"] is False


def test_non_severe_branch_collects_date_then_additional_information():
    client = _client()
    session_id = "t-non-severe"

    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 27/05/2000",
        "yes",
        "9886094321",
        "yes",
        "no thanks",
    ):
        _turn(client, session_id, message)

    severity = _turn(client, session_id, "No, it was not severe")
    assert severity["next_question"][0] == "onset_date"

    onset = _turn(client, session_id, "It started yesterday")
    # Step 10B.2 is asked in the response (next_question stays null).
    assert onset["next_question"] is None
    assert "anything else you'd like to share" in onset["response_text"]

    final = _turn(client, session_id, "no")
    assert final["completed"] is True
    assert final["current_state"] == "CASE_COMPLETED"
    assert final["ae"]["collected_information"]["onset_date"] == (
        "It started yesterday"
    )


# ---- relative onset dates and channel-appropriate wording -------------------


def test_yesterday_resolves_to_a_calendar_date_in_text_reports():
    client = _client()
    session_id = "t-yesterday"

    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 27/05/2000",
        "yes",
        "9886094321",
        "yes",
        "no thanks",
        "No, it was not severe",
    ):
        _turn(client, session_id, message)

    onset = _turn(
        client,
        session_id,
        "Yesterday",
        conversation_timestamp="2026-10-09T08:30:00+05:30",
    )

    collected = onset["ae"]["collected_information"]
    # The original wording is preserved and the calendar date is resolved
    # against the conversation timestamp (9 October 2026 -> yesterday).
    assert collected["onset_date"] == "Yesterday"
    assert collected["onset_date_normalized"] == "2026-10-08"
    assert collected["onset_date_display"] == "08-Oct-2026"
    assert collected["onset_date_source"] == "normalized"
    assert "date unknown" not in onset["response_text"]


def test_text_email_confirmation_says_understood_not_heard():
    client = _client()
    session_id = "t-understood"

    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 27/05/2000",
        "yes",
        "9886094321",
        "yes",
        "Yes, I consent",
        "yes",
    ):
        _turn(client, session_id, message)

    readback = _turn(client, session_id, "alpha beta at gmail dot com")

    # Text conversations never say "heard"; the value and confirmation
    # behaviour are unchanged.
    assert (
        "I understood your email address as alphabeta@gmail.com."
        in readback["response_text"]
    )
    assert "I heard" not in readback["response_text"]
    assert readback["next_question"][0] == "confirmation"
