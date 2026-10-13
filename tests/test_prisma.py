"""Offline tests for the Gnani Prisma v2.5 adapter and its CLI.

No test here contacts the network or needs a real API key: the HTTP session
is always replaced with a mock.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from app.config import get_gnani_api_key
from app.services import prisma
from app.services.prisma import (
    DEFAULT_ENDPOINT,
    PrismaAPIError,
    PrismaConfigurationError,
    PrismaFileError,
    PrismaNetworkError,
    PrismaSTT,
    PrismaValidationError,
    TranscriptionResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_response(
    status_code: int = 200,
    payload: object | None = None,
    text: str = "",
    invalid_json: bool = False,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    if invalid_json:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = payload
    return response


def make_client(
    response: MagicMock | None = None,
    side_effect: Exception | None = None,
    api_key: str = "test-key",
) -> tuple[PrismaSTT, MagicMock]:
    session = MagicMock()
    if side_effect is not None:
        session.post.side_effect = side_effect
    else:
        session.post.return_value = response
    return PrismaSTT(api_key=api_key, session=session), session


def make_audio(tmp_path: Path, name: str = "sample.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"RIFF0000WAVEfake-audio")
    return path


def test_success_returns_transcript(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(
        payload={
            "success": True,
            "request_id": "req_abc123",
            "timestamp": "20251226_143052.123",
            "transcript": "I took Lantus yesterday.",
            "model": "gnani-prisma-v2.5",
        }
    )
    client, session = make_client(response=response)

    result = client.transcribe(audio, language_code="en-IN")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "I took Lantus yesterday."
    assert result.request_id == "req_abc123"
    assert result.model == "gnani-prisma-v2.5"
    assert result.timestamp == "20251226_143052.123"


def test_success_sends_documented_request_shape(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(payload={"success": True, "transcript": "hello"})
    client, session = make_client(response=response)

    client.transcribe(audio, language_code="hi-IN", output_format="transcribe")

    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args == (DEFAULT_ENDPOINT,)
    assert kwargs["headers"] == {"X-API-Key-ID": "test-key"}
    assert kwargs["data"] == {"language_code": "hi-IN", "format": "transcribe"}
    assert "audio_file" in kwargs["files"]
    assert kwargs["files"]["audio_file"][0] == "sample.wav"
    assert "timeout" in kwargs


def test_missing_api_key_raises_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = make_audio(tmp_path)
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(prisma, "get_gnani_api_key", lambda: None)
    client, session = make_client(api_key=None)

    with pytest.raises(PrismaConfigurationError):
        client.transcribe(audio)

    session.post.assert_not_called()


def test_missing_audio_file_raises_file_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(PrismaFileError):
        client.transcribe(PROJECT_ROOT / "does-not-exist.wav")

    session.post.assert_not_called()


def test_unsupported_audio_type_raises_file_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path, name="notes.txt")
    client, session = make_client(response=make_response())

    with pytest.raises(PrismaFileError):
        client.transcribe(audio)

    session.post.assert_not_called()


def test_unsupported_language_code_raises_validation_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    client, session = make_client(response=make_response())

    with pytest.raises(PrismaValidationError):
        client.transcribe(audio, language_code="xx-XX")

    session.post.assert_not_called()


def test_unsupported_output_format_raises_validation_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    client, session = make_client(response=make_response())

    with pytest.raises(PrismaValidationError):
        client.transcribe(audio, output_format="json")

    session.post.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 403, 429, 500, 503])
def test_http_error_raises_api_error_with_status(tmp_path: Path, status_code: int) -> None:
    audio = make_audio(tmp_path)
    response = make_response(
        status_code=status_code,
        payload={
            "success": False,
            "error": {"type": "SOME_ERROR", "message": "Something went wrong."},
        },
    )
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio)

    assert excinfo.value.status_code == status_code
    assert "Something went wrong." in str(excinfo.value)
    assert excinfo.value.hint is not None


def test_forbidden_hint_mentions_api_key(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(
        status_code=403,
        payload={"success": False, "error": {"type": "FORBIDDEN", "message": "Nope."}},
    )
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio)

    assert "API key" in excinfo.value.hint


def test_success_false_with_http_200_raises_api_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(
        status_code=200,
        payload={"success": False, "error": {"type": "API_ERROR", "message": "boom"}},
    )
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio)

    assert excinfo.value.status_code == 200


def test_connection_error_raises_network_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    client, _ = make_client(side_effect=requests.exceptions.ConnectionError("refused"))

    with pytest.raises(PrismaNetworkError):
        client.transcribe(audio)


def test_timeout_raises_network_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    client, _ = make_client(side_effect=requests.exceptions.Timeout("too slow"))

    with pytest.raises(PrismaNetworkError):
        client.transcribe(audio)


def test_non_json_body_raises_api_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(status_code=502, text="<html>bad gateway</html>", invalid_json=True)
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio)

    assert "non-JSON" in str(excinfo.value)


def test_empty_transcript_raises_api_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(payload={"success": True, "transcript": "   "})
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio)

    assert "empty transcript" in str(excinfo.value)


def test_missing_transcript_field_raises_api_error(tmp_path: Path) -> None:
    audio = make_audio(tmp_path)
    response = make_response(payload={"success": True})
    client, _ = make_client(response=response)

    with pytest.raises(PrismaAPIError):
        client.transcribe(audio)


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "medvaani_transcribe_cli", PROJECT_ROOT / "scripts" / "transcribe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_reports_missing_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: None)
    audio = make_audio(tmp_path)

    exit_code = module.main([str(audio)])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_CONFIG
    assert "[FAIL] API key" in output
    assert "GNANI_API_KEY" in output


def test_cli_reports_missing_audio_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")

    exit_code = module.main([str(tmp_path / "nope.wav")])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_FILE
    assert "[FAIL] Audio" in output


def test_cli_reports_api_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")
    audio = make_audio(tmp_path)

    def failing_transcribe(*args: object, **kwargs: object) -> object:
        raise PrismaAPIError(
            "You do not have permission to access this resource.",
            status_code=403,
            error_type="FORBIDDEN",
        )

    monkeypatch.setattr(module.PrismaSTT, "transcribe", failing_transcribe)

    exit_code = module.main([str(audio)])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_API
    assert "[FAIL] API" in output
    assert "HTTP 403" in output
    assert "Hint:" in output


def test_cli_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")
    audio = make_audio(tmp_path)

    def fake_transcribe(*args: object, **kwargs: object) -> TranscriptionResult:
        return TranscriptionResult(
            text="I am having a headache after taking Lantus.",
            request_id="req_123",
            model="gnani-prisma-v2.5",
        )

    monkeypatch.setattr(module.PrismaSTT, "transcribe", fake_transcribe)

    exit_code = module.main([str(audio), "--language", "en-IN"])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_OK
    assert "[ OK ] Request" in output
    assert "TRANSCRIPT" in output
    assert "headache after taking Lantus." in output


def test_config_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    assert get_gnani_api_key() is None
