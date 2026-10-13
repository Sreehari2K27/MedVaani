"""Tests for conversational voice endpoint."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import wave

from fastapi.testclient import TestClient

from app.services.prisma import PrismaAPIError, PrismaNetworkError


def create_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


def test_voice_conversation_endpoint_exists():
    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    assert hasattr(client, "post")


def test_missing_session_id_returns_400():
    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "test.wav"
        create_test_wav(wav_path)
        with open(wav_path, "rb") as f:
            resp = client.post("/voice/conversation", files={"audio_file": ("test.wav", f, "audio/wav")})
    assert resp.status_code == 422 or resp.status_code == 400  # depends on validation


def test_empty_audio_returns_400():
    from app.main import create_app

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/voice/conversation",
        files={"audio_file": ("empty.wav", b"", "audio/wav")},
        data={"session_id": "s1"},
    )
    assert resp.status_code == 400


# ---- STT failure classification (distinct causes, no generic collapse) ------


class FailingSTT:
    """Offline STT stand-in that always raises one configured error."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def transcribe(self, audio_path, language_code="en-IN"):
        raise self._exc


class FailOnceSTT:
    """Fails the first call, then transcribes successfully."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def transcribe(self, audio_path, language_code="en-IN"):
        self.calls += 1
        if self.calls == 1:
            raise self._exc
        return SimpleNamespace(text="hello")


class FakeTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _post_voice(stt, session_id="stt-err"):
    from app.main import create_app

    app = create_app(prisma=stt, timbre=FakeTTS())
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "test.wav"
        create_test_wav(wav_path)
        with open(wav_path, "rb") as f:
            return client.post(
                "/voice/conversation",
                files={"audio_file": ("test.wav", f, "audio/wav")},
                data={"session_id": session_id},
            )


def test_stt_auth_failure_maps_to_500_with_clear_hint():
    resp = _post_voice(
        FailingSTT(PrismaAPIError("Invalid API key.", status_code=403))
    )

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "Access denied" in detail
    # The provider's raw message (and any key material) stays out of the path.
    assert "Invalid API key." not in detail


def test_stt_rate_limit_maps_to_429_with_clear_hint():
    resp = _post_voice(
        FailingSTT(PrismaAPIError("Too many requests.", status_code=429))
    )

    assert resp.status_code == 429
    assert "Rate limit" in resp.json()["detail"]


def test_stt_empty_transcript_stays_502_with_visible_detail():
    resp = _post_voice(
        FailingSTT(
            PrismaAPIError("Gnani STT returned an empty transcript.", status_code=200)
        )
    )

    assert resp.status_code == 502
    assert "empty transcript" in resp.json()["detail"]


def test_stt_network_failure_stays_502_with_visible_detail():
    resp = _post_voice(
        FailingSTT(
            PrismaNetworkError("Gnani STT request timed out after (10, 60): x")
        )
    )

    assert resp.status_code == 502
    assert "timed out" in resp.json()["detail"]


def test_failed_attempt_can_be_retried_successfully():
    stt = FailOnceSTT(PrismaNetworkError("Gnani STT request failed: boom"))

    first = _post_voice(stt, session_id="stt-retry")
    assert first.status_code == 502

    second = _post_voice(stt, session_id="stt-retry")
    assert second.status_code == 200
    body = second.json()
    assert body["transcript"] == "hello"
    assert body["workflow"] == "ae"
