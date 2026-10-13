"""Offline end-to-end tests for the conversational voice endpoint MI/AE routing."""
from __future__ import annotations

from pathlib import Path
import tempfile
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import MagicMock

import requests

from app.api.voice_conversation import create_voice_conversation_router
from app.services import timbre as timbre_module
from app.services.prisma import PrismaAPIError, PrismaValidationError
from app.services.timbre import TimbreAPIError, TimbreTTS

OUT = Path("output")


def create_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


class FakeSTT:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    def transcribe(self, audio_path, language_code="en-IN"):
        if not self._transcripts:
            return SimpleNamespace(text="")
        return SimpleNamespace(text=self._transcripts.pop(0))


class FakeTTS:
    def __init__(self, raise_error=None):
        self._error = raise_error
        self.written = []

    def synthesize_to_file(self, text, output_path, language="en-IN"):
        if self._error is not None:
            raise self._error
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        self.written.append(Path(output_path))
        return Path(output_path)


def _client(stt, tts=None):
    app = FastAPI()
    app.include_router(create_voice_conversation_router(stt=stt, tts=tts or FakeTTS()))
    return TestClient(app)


def _post(client, session_id, tmp):
    wav_path = Path(tmp) / "audio.wav"
    create_test_wav(wav_path)
    with open(wav_path, "rb") as f:
        return client.post(
            "/voice/conversation",
            files={"audio_file": ("audio.wav", f, "audio/wav")},
            data={"session_id": session_id},
        )


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _cleanup(session_ids):
    for sid in session_ids:
        for p in OUT.glob(f"conv_*_{sid}.wav"):
            try:
                p.unlink()
            except OSError:
                pass


def test_mi_turn_returns_grounded_envelope_and_does_not_touch_ae_engine(tmp):
    client = _client(FakeSTT(["What are the side effects of Lantus?"]))
    state_before = None

    sid = "mi-1"
    r1 = _post(client, sid, tmp)
    assert r1.status_code == 200
    data = r1.json()
    assert data["workflow"] == "mi"
    assert data["intent"] == "product_info"
    assert "response_text" in data and data["response_text"]
    assert isinstance(data["audio_path"], str)
    assert isinstance(data["current_state"], str)
    assert data["case_id"]
    assert data["priority"]
    assert data["mi"] is not None
    assert data["mi"]["status"] == "allowed"
    assert data["mi"]["grounded"] is True
    assert data["mi"]["supported"] is True
    assert data["mi"]["source_page_numbers"]
    assert data["ae"] is None
    state_before = data["current_state"]
    assert "package insert" in data["response_text"]

    # Second MI turn in the same session must not advance the AE engine.
    r2 = _post(client, sid, tmp)
    assert r2.status_code == 200
    assert r2.json()["current_state"] == state_before
    _cleanup([sid])


def test_same_session_mi_then_ae_and_envelope_keys_match(tmp):
    sid = "mix-1"
    stt = FakeSTT([
        "What is Lantus used for?",
        "I took Lantus yesterday and had a severe headache and vomiting",
    ])
    client = _client(stt)
    mi = _post(client, sid, tmp)
    assert mi.status_code == 200
    mi_keys = set(mi.json().keys())

    ae = _post(client, sid, tmp)
    assert ae.status_code == 200
    data = ae.json()
    assert data["workflow"] == "ae"
    assert data["mi"] is None
    assert data["ae"] is not None
    assert set(data.keys()) == mi_keys
    assert data["response_text"]
    _cleanup([sid])


def test_urgent_utterance_escalates_via_ae_workflow(tmp):
    sid = "urg-1"
    client = _client(FakeSTT(["I am having severe difficulty breathing, what is Lantus used for?"]))
    resp = _post(client, sid, tmp)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "ae"
    assert data["intent"] == "urgent_medical"
    assert data["safety_status"] == "urgent_life_threatening"
    assert data["human_review_required"] is True
    assert data["priority"] == "high"
    _cleanup([sid])


def test_personalized_dosing_question_gets_bounded_refusal(tmp):
    sid = "block-1"
    client = _client(FakeSTT(["How many units should I take tonight?"]))
    resp = _post(client, sid, tmp)
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "mi"
    assert data["intent"] == "treatment_query"
    assert data["mi"]["status"] == "blocked"
    assert data["mi"]["grounded"] is False
    assert data["mi"]["supported"] is False
    assert data["mi"]["source_page_numbers"] == []
    assert "dosing" in data["response_text"].lower()
    _cleanup([sid])


def test_prisma_api_error_maps_to_502(tmp):
    class ErrorSTT(FakeSTT):
        def transcribe(self, audio_path, language_code="en-IN"):
            raise PrismaAPIError("upstream failed")

    sid = "err-502"
    client = _client(ErrorSTT([]))
    resp = _post(client, sid, tmp)
    assert resp.status_code == 502
    _cleanup([sid])


def test_prisma_validation_error_maps_to_400(tmp):
    class ErrorSTT(FakeSTT):
        def transcribe(self, audio_path, language_code="en-IN"):
            raise PrismaValidationError("invalid input")

    sid = "err-400"
    client = _client(ErrorSTT([]))
    resp = _post(client, sid, tmp)
    assert resp.status_code == 400
    _cleanup([sid])


def test_timbre_api_error_maps_to_502(tmp):
    sid = "err-tts"
    tts = FakeTTS(raise_error=TimbreAPIError("synthesis failed"))
    client = _client(FakeSTT(["What is Lantus used for?"]), tts=tts)
    resp = _post(client, sid, tmp)
    assert resp.status_code == 502
    _cleanup([sid])


def test_timbre_retry_transparent_to_endpoint(tmp, monkeypatch):
    fake_audio = b"RIFF0000WAVEfake"

    def response_factory(status_code):
        response = MagicMock(spec=requests.Response)
        response.status_code = status_code
        response.content = fake_audio
        response.text = ""
        response.headers = {
            "Content-Type": "audio/wav" if status_code == 200 else "application/json"
        }
        response.json.return_value = (
            {"success": False, "error": {"type": "RATE_LIMIT", "message": "slow down"}}
            if status_code != 200
            else {}
        )
        return response

    monkeypatch.setattr(timbre_module.time, "sleep", lambda _: None)
    session = MagicMock()
    session.post.side_effect = [response_factory(429), response_factory(200)]
    tts = TimbreTTS(api_key="test-key", session=session, max_attempts=3)
    sid = "retry-429"
    client = _client(FakeSTT(["What is Lantus used for?"]), tts=tts)

    resp = _post(client, sid, tmp)

    assert resp.status_code == 200
    assert session.post.call_count == 2
    _cleanup([sid])


def test_audio_written_to_unique_session_path(tmp):
    sid = "uniq-sid"
    tts = FakeTTS()
    client = _client(FakeSTT(["What is Lantus used for?"]), tts=tts)
    resp = _post(client, sid, tmp)
    assert resp.status_code == 200
    assert tts.written
    assert str(tts.written[0]).endswith(f"conv_mi_{sid}.wav")
    assert tts.written[0].exists()
    _cleanup([sid])