"""Offline tests for the FastAPI voice round-trip endpoint.

The Prisma and Timbre clients are injected as mocked instances, so no test
contacts the network or needs a real API key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import OUTPUT_AUDIO_PATH, ROUND_TRIP_RESPONSE_TEXT, create_app
from app.services.prisma import (
    PrismaAPIError,
    PrismaConfigurationError,
    PrismaFileError,
    PrismaNetworkError,
    PrismaValidationError,
    TranscriptionResult,
)
from app.services.timbre import (
    TimbreAPIError,
    TimbreConfigurationError,
    TimbreNetworkError,
    TimbreValidationError,
)

FAKE_AUDIO = b"RIFF0000WAVEfake"

TEST_FILENAME = "patient-voice.wav"


def make_mock_prisma(transcript: str = "I am having a headache after taking Lantus.") -> MagicMock:
    prisma = MagicMock()
    prisma.transcribe.return_value = TranscriptionResult(
        text=transcript,
        request_id="req_smoke",
        model="gnani-prisma-v2.5",
    )
    return prisma


def make_mock_timbre() -> MagicMock:
    timbre = MagicMock()

    def fake_synthesize_to_file(text: str, output_path: object, **kwargs: object) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FAKE_AUDIO)
        return path

    timbre.synthesize_to_file.side_effect = fake_synthesize_to_file
    return timbre


def make_client(prisma: MagicMock, timbre: MagicMock) -> TestClient:
    return TestClient(create_app(prisma=prisma, timbre=timbre))


def upload(test_client: TestClient) -> TestClient:
    return test_client.post(
        "/voice/round-trip",
        files={"audio_file": (TEST_FILENAME, FAKE_AUDIO, "audio/wav")},
        data={"language": "en-IN"},
    )


def test_round_trip_success() -> None:
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    response = upload(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "I am having a headache after taking Lantus."
    assert payload["response_text"] == ROUND_TRIP_RESPONSE_TEXT
    assert payload["language"] == "en-IN"
    assert payload["audio_path"].endswith(OUTPUT_AUDIO_PATH.name)


def test_round_trip_response_text_is_deterministic() -> None:
    assert ROUND_TRIP_RESPONSE_TEXT == (
        "Thank you. I received your message through MedVaani."
    )


def test_round_trip_flows_audio_through_prisma_then_timbre(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    response = upload(client)

    assert response.status_code == 200
    prisma.transcribe.assert_called_once()
    tmp_arg = prisma.transcribe.call_args.args[0]
    assert prisma.transcribe.call_args.kwargs["language_code"] == "en-IN"

    timbre.synthesize_to_file.assert_called_once()
    args, kwargs = timbre.synthesize_to_file.call_args
    assert args[0] == ROUND_TRIP_RESPONSE_TEXT
    assert Path(kwargs.get("output_path", args[1])).name == OUTPUT_AUDIO_PATH.name

    assert not Path(tmp_arg).exists()


def test_round_trip_writes_audio_file() -> None:
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    audio_path = Path(OUTPUT_AUDIO_PATH)
    _ = upload(client)

    assert audio_path.read_bytes() == FAKE_AUDIO
    audio_path.unlink(missing_ok=True)


def test_missing_audio_file_returns_422() -> None:
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    response = client.post("/voice/round-trip")

    assert response.status_code == 422
    prisma.transcribe.assert_not_called()
    timbre.synthesize_to_file.assert_not_called()


def test_empty_audio_file_returns_400() -> None:
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    response = client.post(
        "/voice/round-trip",
        files={"audio_file": (TEST_FILENAME, b"", "audio/wav")},
    )

    assert response.status_code == 400
    prisma.transcribe.assert_not_called()


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (PrismaFileError("bad file"), 400),
        (PrismaValidationError("bad language"), 400),
        (TimbreValidationError("bad params"), 400),
        (PrismaConfigurationError("no key"), 500),
        (TimbreConfigurationError("no key"), 500),
        (PrismaNetworkError("unreachable"), 502),
        (PrismaAPIError("upstream refused", status_code=403), 502),
        (TimbreNetworkError("unreachable"), 502),
        (TimbreAPIError("upstream refused", status_code=503), 502),
    ],
)
def test_round_trip_maps_errors_to_status(error: Exception, expected_status: int) -> None:
    prisma = make_mock_prisma()
    prisma.transcribe.side_effect = error
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    response = upload(client)

    assert response.status_code == expected_status
    timbre.synthesize_to_file.assert_not_called()


def test_timbre_error_after_transcription() -> None:
    prisma = make_mock_prisma()
    timbre = make_mock_timbre()
    timbre.synthesize_to_file.side_effect = TimbreAPIError(
        "TTS failed", status_code=500
    )
    client = make_client(prisma, timbre)

    response = upload(client)

    assert response.status_code == 502
    prisma.transcribe.assert_called_once()


def test_temp_upload_is_cleaned_up_after_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    prisma = make_mock_prisma()
    prisma.transcribe.side_effect = PrismaAPIError("boom", status_code=500)
    timbre = make_mock_timbre()
    client = make_client(prisma, timbre)

    created_before = set(tmp_path.iterdir())
    response = upload(client)
    created_after = set(tmp_path.iterdir())

    assert response.status_code == 502
    assert created_before == created_after


def test_default_app_is_importable_and_exposes_route() -> None:
    from app.main import app

    assert app is not None
    found = any(route.path == "/voice/round-trip" for route in app.routes)
    assert found