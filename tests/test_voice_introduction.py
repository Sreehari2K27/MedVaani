"""Focused tests for the voice-first AE greeting and introduction de-duplication.

Offline: Timbre and Prisma are replaced with fakes, and turns are driven
through the shared text/voice endpoints. The contract under test is that the
approved introduction is delivered exactly once per fresh voice session —
spoken and marked by the greeting endpoint before the first utterance — so
that first utterance is answered by the ordinary AE workflow without a
repeated introduction, with question order, event-description capture and
urgent escalation untouched.
"""
from __future__ import annotations

import io
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.mock import MockLLMProvider
from app.services.timbre import TimbreAPIError
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    EVENT_DESCRIPTION_QUESTION,
    ConversationalAEWorkflow,
    INTRODUCTION_MESSAGE,
)

INTRO_PREFIX = "Thank you for taking the time to speak with us."


class FakeTTS:
    """Writes a real 0.1 s PCM WAV, exactly the artefact Timbre produces."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize_to_file(self, text, output_path, language="en-IN", **kwargs):
        self.calls.append(
            {"text": text, "output": str(output_path), "language": language, **kwargs}
        )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        Path(output_path).write_bytes(buffer.getvalue())
        return str(output_path)


class FailingTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", **kwargs):
        raise TimbreAPIError("synthesis failed")


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_path, language_code="en-IN"):
        return SimpleNamespace(text=self.text, detected_language=None, timestamp=None)


def _client(tts=None, stt=None):
    app = create_app(prisma=stt if stt is not None else object(), timbre=tts or FakeTTS())
    return TestClient(app)


def _greeting(client: TestClient, session_id: str):
    return client.post("/voice/introduction", data={"session_id": session_id})


def _text_turn(client: TestClient, session_id: str, message: str):
    return client.post(
        "/text/conversation", data={"session_id": session_id, "message": message}
    )


def _create_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


# ---- greeting delivery ------------------------------------------------------


def test_greeting_returns_the_approved_introduction_and_audio():
    tts = FakeTTS()
    client = _client(tts=tts)
    resp = _greeting(client, "greet-intro")
    assert resp.status_code == 200
    body = resp.json()
    assert body["response_text"] == INTRODUCTION_MESSAGE
    assert body["audio_path"].replace("\\", "/").endswith("conv_intro_greet-intro.wav")
    assert body["conversation_language"] == "en-IN"
    assert body["language_translated"] is False
    assert body["spoken_response_text"] == INTRODUCTION_MESSAGE
    assert tts.calls[0]["text"] == INTRODUCTION_MESSAGE
    assert tts.calls[0]["language"] == "en-IN"
    # No explicit voice override: the default Timbre identity stays authoritative.
    assert "voice" not in tts.calls[0]


def test_greeting_audio_is_served_as_valid_wav_bytes():
    """The produced WAV must be fetchable as audio — the silent-greeting regression.

    The JSON carries a filesystem path (`output\\conv_intro_*.wav`), which is
    NOT a route: the proven failure was the client requesting it raw (404).
    Clients must map it to the mounted /audio/ route (frontend audioUrlFor),
    which serves the full bytes under an audio content type.
    """
    client = _client()
    body = _greeting(client, "greet-served").json()
    assert body["audio_path"].replace("\\", "/").startswith("output/")

    raw = client.get("/" + body["audio_path"].replace("\\", "/"))
    assert raw.status_code == 404  # the failure the browser hit before the fix

    name = body["audio_path"].replace("\\", "/").split("/")[-1]
    served = client.get(f"/audio/{name}?_=1")
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("audio/")
    assert len(served.content) > 44  # WAV header + PCM frames
    assert served.content[:4] == b"RIFF"
    assert served.headers["content-length"] == str(len(served.content))


def test_greeting_is_delivery_only_not_a_workflow_turn():
    client = _client()
    body = _greeting(client, "greet-shape").json()
    # The greeting carries no engine/workflow turn envelope: no case state,
    # no completion flags, no questions consumed.
    for key in ("current_state", "completed", "next_question", "transcript"):
        assert key not in body


def test_greeting_requires_a_session_id():
    client = _client()
    assert client.post("/voice/introduction", data={}).status_code == 422
    assert (
        client.post("/voice/introduction", data={"session_id": "   "}).status_code
        == 400
    )


# ---- first utterance never repeats the introduction -------------------------


def test_first_text_utterance_after_greeting_does_not_repeat_the_introduction():
    client = _client()
    assert _greeting(client, "greet-first").status_code == 200
    body = _text_turn(client, "greet-first", "I am having severe").json()

    assert INTRO_PREFIX not in body["response_text"]
    assert "Thank you for taking the time" not in body["response_text"]
    # The utterance itself is preserved and still drives the ordinary flow.
    assert body["transcript"] == "I am having severe"
    assert body["next_question"][0] == "adverse_event"
    assert "Please tell me what happened" in body["response_text"]
    assert body["safety_status"] == "ok"


def test_first_voice_utterance_after_greeting_does_not_repeat_the_introduction():
    client = _client(stt=FakeSTT("I am having severe"))
    assert _greeting(client, "greet-voice").status_code == 200
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "turn.wav"
        _create_wav(wav_path)
        with open(wav_path, "rb") as handle:
            resp = client.post(
                "/voice/conversation",
                files={"audio_file": ("turn.wav", handle, "audio/wav")},
                data={"session_id": "greet-voice"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert INTRO_PREFIX not in body["response_text"]
    assert body["transcript"] == "I am having severe"
    assert body["next_question"][0] == "adverse_event"
    assert body["safety_status"] == "ok"


def test_utterance_processed_identically_apart_from_the_introduction():
    greeting_client = _client()
    baseline_client = _client()

    assert _greeting(greeting_client, "greet-parity").status_code == 200
    first = _text_turn(greeting_client, "greet-parity", "I am having severe").json()
    second = _text_turn(
        greeting_client, "greet-parity", "I have a headache after taking Lantus"
    ).json()

    base_first = _text_turn(baseline_client, "base-parity", "I am having severe").json()
    base_second = _text_turn(
        baseline_client, "base-parity", "I have a headache after taking Lantus"
    ).json()

    # Without the greeting the endpoint still emits the introduction, whose
    # own tail already carries the event invitation (the workflow never asks
    # it twice on the same turn).
    assert base_first["response_text"] == INTRODUCTION_MESSAGE
    assert INTRODUCTION_MESSAGE.endswith(
        "Please tell me what happened, in your own words."
    )
    # With the greeting the introduction is gone and the same invitation
    # arrives as the ordinary pending question — same words, same ask.
    assert first["response_text"] == EVENT_DESCRIPTION_QUESTION
    assert first["next_question"] == base_first["next_question"]
    assert first["next_question"][0] == "adverse_event"
    assert first["current_state"] == base_first["current_state"]
    assert first["safety_status"] == base_first["safety_status"]

    # The following turn is byte-for-byte the same on both paths: question
    # order, acknowledgements and warnings are untouched by the greeting.
    assert second["response_text"] == base_second["response_text"]
    assert second["next_question"] == base_second["next_question"]
    assert second["current_state"] == base_second["current_state"]
    assert second["safety_status"] == base_second["safety_status"]
    assert not second["response_text"].startswith(INTRODUCTION_MESSAGE)


def test_urgent_escalation_unchanged_after_greeting():
    client = _client()
    assert _greeting(client, "greet-urgent").status_code == 200
    body = _text_turn(
        client, "greet-urgent", "I am having severe difficulty breathing"
    ).json()

    assert body["safety_status"] == "urgent_life_threatening"
    assert body["human_review_required"] is True
    assert INTRO_PREFIX not in body["response_text"]
    response = body["response_text"].lower()
    assert "emergency" in response or "immediate" in response


def test_tts_failure_still_marks_the_introduction_shown():
    client = _client(tts=FailingTTS())
    assert _greeting(client, "greet-tts-fail").status_code == 502

    # The text greeting was already delivered to the UI, so even a failed
    # synthesis must not let the first utterance repeat it.
    body = _text_turn(client, "greet-tts-fail", "hello").json()
    assert INTRO_PREFIX not in body["response_text"]
    assert body["next_question"][0] == "adverse_event"


def test_without_greeting_the_first_turn_still_emits_the_introduction():
    client = _client()
    body = _text_turn(client, "no-greeting", "hello").json()
    assert body["response_text"].startswith(INTRODUCTION_MESSAGE)


# ---- the public marking API -------------------------------------------------


def test_mark_introduction_shown_is_public_and_sufficient():
    assert callable(getattr(ConversationalAEWorkflow, "mark_introduction_shown"))

    engine = AEStateEngine.create_case("intro-marked")
    marked = ConversationalAEWorkflow(engine, MockLLMProvider())
    marked.mark_introduction_shown()
    result = marked.process_turn("hello")
    assert INTRO_PREFIX not in result.response_text
    assert result.next_question[0] == "adverse_event"

    fresh_engine = AEStateEngine.create_case("intro-unmarked")
    fresh = ConversationalAEWorkflow(fresh_engine, MockLLMProvider())
    baseline = fresh.process_turn("hello")
    assert baseline.response_text.startswith(INTRODUCTION_MESSAGE)


# ---- localized introduction (explicitly selected language) --------------------


def _greeting_in(client: TestClient, session_id: str, language: str):
    return client.post(
        "/voice/introduction",
        data={"session_id": session_id, "language": language},
    )


def test_greeting_localizes_for_all_nine_catalogued_languages():
    for code, voice in (
        ("hi-IN", "Nalini"),
        ("bn-IN", "Kirra"),
        ("gu-IN", "Falak"),
        ("kn-IN", "Saanvi"),
        ("ml-IN", "Reshma"),
        ("mr-IN", "Zahira"),
        ("pa-IN", "Mehuli"),
        ("ta-IN", "Asmita"),
        ("te-IN", "Suhana"),
    ):
        tts = FakeTTS()
        client = _client(tts=tts)
        resp = _greeting_in(client, f"greet-{code}", code)
        assert resp.status_code == 200, code
        body = resp.json()
        # The approved English wording stays the UI contract; the spoken form
        # is the localized one, synthesized with that language's voice.
        assert body["response_text"] == INTRODUCTION_MESSAGE, code
        assert body["conversation_language"] == code, code
        assert body["language_translated"] is True, code
        assert body["spoken_response_text"] != INTRODUCTION_MESSAGE, code
        assert body["voice"] == voice, code
        assert tts.calls[0]["language"] == code, code
        assert tts.calls[0]["voice"] == voice, code
        assert tts.calls[0]["text"] == body["spoken_response_text"], code


def test_greeting_with_an_unsupported_language_stays_english():
    tts = FakeTTS()
    client = _client(tts=tts)
    body = _greeting_in(client, "greet-xx", "xx-XX").json()
    assert body["response_text"] == INTRODUCTION_MESSAGE
    assert body["conversation_language"] == "en-IN"
    assert body["language_translated"] is False
    assert body["spoken_response_text"] == INTRODUCTION_MESSAGE
    assert tts.calls[0]["language"] == "en-IN"
    assert "voice" not in tts.calls[0]


def test_greeting_without_a_language_field_keeps_the_existing_contract():
    tts = FakeTTS()
    client = _client(tts=tts)
    body = _greeting(client, "greet-default").json()
    assert body["language_translated"] is False
    assert body["spoken_response_text"] == INTRODUCTION_MESSAGE
    assert tts.calls[0]["language"] == "en-IN"
    assert "voice" not in tts.calls[0]


def test_localized_synthesis_failure_falls_back_to_english_honestly():
    class LocalizedOnlyFailingTTS(FakeTTS):
        def synthesize_to_file(self, text, output_path, language="en-IN", **kwargs):
            if language != "en-IN":
                self.calls.append(
                    {
                        "text": text,
                        "output": str(output_path),
                        "language": language,
                        **kwargs,
                    }
                )
                raise TimbreAPIError("localized synthesis failed")
            return super().synthesize_to_file(
                text, output_path, language=language, **kwargs
            )

    tts = LocalizedOnlyFailingTTS()
    client = _client(tts=tts)
    body = _greeting_in(client, "greet-fallback", "ta-IN").json()
    # The spoken introduction is the approved English one and is reported as
    # untranslated: never a silent English fallback behind a localized claim.
    assert body["language_translated"] is False
    assert body["spoken_response_text"] == INTRODUCTION_MESSAGE
    assert body["voice"] is None
    assert body["conversation_language"] == "en-IN"
    assert [call["language"] for call in tts.calls] == ["ta-IN", "en-IN"]


def test_failing_synthesis_still_returns_502_for_localized_greeting():
    client = _client(tts=FailingTTS())
    assert _greeting_in(client, "greet-fail-502", "ta-IN").status_code == 502
