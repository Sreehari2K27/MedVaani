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
    headers: dict | None = None,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    # ``headers`` is an instance attribute of requests.Response, so it is
    # absent from the MagicMock(spec=...) surface: provide a default dict.
    response.headers = dict(headers) if headers is not None else {}
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


# ---- bounded retry behaviour (mirrors the Timbre TTS retry conventions) -----


def make_retriable_response(
    status_code: int, *, text: str = "", invalid_json: bool = False,
    headers: dict | None = None,
) -> MagicMock:
    return make_response(
        status_code=status_code, text=text, invalid_json=invalid_json,
        headers=headers,
    )


@pytest.fixture
def mocked_retry_timing(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No real sleeping and no jitter randomness; records every slept delay."""
    sleeps: list[float] = []
    monkeypatch.setattr(prisma.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(prisma.random, "uniform", lambda low, high: 0)
    return sleeps


_OK_RESPONSE_PAYLOAD = {"success": True, "transcript": "I took Lantus yesterday."}


def test_504_html_response_is_retried_then_successfully_transcribed(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    retry_response = make_retriable_response(
        504, text="<html>gateway timeout</html>", invalid_json=True
    )
    ok_response = make_response(payload=_OK_RESPONSE_PAYLOAD)
    client, session = make_client()
    session.post.side_effect = [retry_response, ok_response]

    result = client.transcribe(audio, language_code="ml-IN")

    assert result.text == "I took Lantus yesterday."
    assert session.post.call_count == 2
    assert mocked_retry_timing == [0.5]          # base delay after attempt 1


def test_repeated_504_exhausts_retries_with_identical_requests(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    retry_response = make_retriable_response(
        504, text="<html>gateway timeout</html>", invalid_json=True
    )
    client, session = make_client()
    session.post.side_effect = [retry_response, retry_response, retry_response]

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio, language_code="ml-IN")

    assert excinfo.value.status_code == 504
    assert session.post.call_count == 3
    first_kwargs = session.post.call_args_list[0].kwargs
    second_kwargs = session.post.call_args_list[1].kwargs
    third_kwargs = session.post.call_args_list[2].kwargs
    for kwargs in (second_kwargs, third_kwargs):
        assert kwargs["headers"] == first_kwargs["headers"]
        assert kwargs["data"] == first_kwargs["data"]
    assert mocked_retry_timing == [0.5, 1.0]


@pytest.mark.parametrize("status_code", [400, 403])
def test_permanent_statuses_fail_immediately_without_retry(
    tmp_path: Path, mocked_retry_timing: list[float], status_code: int
) -> None:
    audio = make_audio(tmp_path)
    response = make_response(
        status_code=status_code,
        payload={
            "success": False,
            "error": {"type": "REJECTED", "message": "Bad audio."},
        },
    )
    client, session = make_client(response=response)

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio, language_code="ml-IN")

    assert excinfo.value.status_code == status_code
    assert session.post.call_count == 1
    assert mocked_retry_timing == []


def test_403_html_body_is_not_retried_and_public_message_stays_html_free(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    response = make_retriable_response(
        403, text="<html>access denied</html>", invalid_json=True
    )
    client, session = make_client()
    session.post.side_effect = [response, response]

    with pytest.raises(PrismaAPIError) as excinfo:
        client.transcribe(audio, language_code="ml-IN")

    assert excinfo.value.status_code == 403
    assert session.post.call_count == 1
    assert "<html>" not in str(excinfo.value)
    assert excinfo.value.body_head == "<html>access denied</html>"
    assert mocked_retry_timing == []


def test_read_timeout_is_never_retried(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    ok_response = make_response(payload=_OK_RESPONSE_PAYLOAD)
    client, session = make_client()
    session.post.side_effect = [
        requests.exceptions.Timeout("too slow"),
        ok_response,
    ]

    with pytest.raises(PrismaNetworkError):
        client.transcribe(audio, language_code="ml-IN")

    assert session.post.call_count == 1
    assert mocked_retry_timing == []


def test_connection_error_is_retried_then_successfully_transcribed(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    ok_response = make_response(payload=_OK_RESPONSE_PAYLOAD)
    client, session = make_client()
    session.post.side_effect = [
        requests.exceptions.ConnectionError("refused"),
        ok_response,
    ]

    result = client.transcribe(audio, language_code="ml-IN")

    assert result.text == "I took Lantus yesterday."
    assert session.post.call_count == 2
    assert len(mocked_retry_timing) == 1


def test_retry_after_is_capped_and_backoff_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(prisma.time, "sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr(prisma.random, "uniform", lambda low, high: 0)
    client, _ = make_client()

    # A provider Retry-After of 999s is honored only up to MAX_RETRY_AFTER_SECONDS.
    client._sleep_before_retry(1, PrismaAPIError("busy", status_code=429, retry_after=999.0))
    assert sleeps == [prisma.MAX_RETRY_AFTER_SECONDS]
    sleeps.clear()

    # A Retry-After within the set is honored exactly.
    client._sleep_before_retry(1, PrismaAPIError("busy", status_code=429, retry_after=17.5))
    assert sleeps == [prisma.MAX_RETRY_AFTER_SECONDS]
    sleeps.clear()

    # Without a usable Retry-After, exponential backoff applies per attempt
    # and is capped at MAX_BACKOFF_SECONDS.
    client._sleep_before_retry(1, PrismaAPIError("busy", status_code=429))
    client._sleep_before_retry(5, PrismaAPIError("busy", status_code=429))
    assert sleeps == [0.5, prisma.MAX_BACKOFF_SECONDS]
    sleeps.clear()

    # Backoff doubles per attempt and never exceeds MAX_BACKOFF_SECONDS.
    client._sleep_before_retry(1, None)
    client._sleep_before_retry(2, None)
    client._sleep_before_retry(99, None)
    assert sleeps == [0.5, 1.0, prisma.MAX_BACKOFF_SECONDS]


def test_non_json_504_message_has_no_body_and_retains_retry_after(
    tmp_path: Path, mocked_retry_timing: list[float]
) -> None:
    audio = make_audio(tmp_path)
    retry_response = make_retriable_response(
        504,
        text="<html>gateway timeout</html>",
        invalid_json=True,
        headers={"Retry-After": "0.2"},
    )
    ok_response = make_response(payload=_OK_RESPONSE_PAYLOAD)
    client, session = make_client()
    session.post.side_effect = [retry_response, ok_response]

    result = client.transcribe(audio, language_code="ml-IN")

    assert result.text == "I took Lantus yesterday."
    assert mocked_retry_timing == [0.2]


def test_zero_max_attempts_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PrismaSTT(api_key="test-key", session=MagicMock(), max_attempts=0)


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
