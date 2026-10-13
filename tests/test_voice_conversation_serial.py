"""Regression test for Path serialization in voice conversation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import wave
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.main import FastAPI


def create_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


class FakeSTT:
    def transcribe(self, audio_path, language_code="en-IN"):
        return SimpleNamespace(text="What is Lantus used for?")


class FakeTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def test_voice_conversation_response_serializable():
    app = FastAPI()
    app.include_router(create_voice_conversation_router(stt=FakeSTT(), tts=FakeTTS()))
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "test.wav"
        create_test_wav(wav_path)
        with open(wav_path, "rb") as f:
            resp = client.post(
                "/voice/conversation",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={"session_id": "test-serial"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "audio_path" in data
    assert isinstance(data["audio_path"], str)
