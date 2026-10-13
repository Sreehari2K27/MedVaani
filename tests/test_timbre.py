"""Offline tests for the Gnani Timbre v2.5 adapter and its CLI.

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

from app.services import timbre
from app.services.timbre import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    ENGLISH_VOICES,
    TimbreAPIError,
    TimbreConfigurationError,
    TimbreNetworkError,
    TimbreTTS,
    TimbreValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_AUDIO = b"RIFF0000WAVEfake-tts-audio"

SLEEP_CALLS: list[float] = []


@pytest.fixture(autouse=True)
def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite offline and fast; record each retry delay for assertions."""
    SLEEP_CALLS.clear()
    monkeypatch.setattr(timbre.time, "sleep", SLEEP_CALLS.append)


def make_response(
    status_code: int = 200,
    content: bytes = FAKE_AUDIO,
    content_type: str = "audio/wav",
    payload: object | None = None,
    invalid_json: bool = False,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.content = content
    response.headers = {"Content-Type": content_type}
    response.text = content.decode("utf-8", errors="replace")
    if invalid_json:
        response.json.side_effect = ValueError("not json")
    elif payload is not None:
        response.json.return_value = payload
    else:
        response.json.return_value = {}
    return response


def make_client(
    response: MagicMock | None = None,
    side_effect: Exception | list[Exception] | None = None,
    api_key: str = "test-key",
    **kwargs: object,
) -> tuple[TimbreTTS, MagicMock]:
    session = MagicMock()
    if side_effect is not None:
        session.post.side_effect = side_effect
    else:
        session.post.return_value = response
    return TimbreTTS(api_key=api_key, session=session, **kwargs), session


def test_success_returns_audio_bytes() -> None:
    client, session = make_client(response=make_response())

    audio = client.synthesize("I am having a headache after taking Lantus.")

    assert audio == FAKE_AUDIO
    session.post.assert_called_once()


def test_success_sends_documented_request_shape() -> None:
    client, session = make_client(response=make_response())

    client.synthesize("Hello", voice="Kaveri", language="en-IN", speed=1.0)

    args, kwargs = session.post.call_args
    assert args == (DEFAULT_ENDPOINT,)
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert set(kwargs["headers"]) == {"Content-Type", "X-API-Key-ID"}
    assert kwargs["headers"]["X-API-Key-ID"] == "test-key"
    assert "json" in kwargs

    body = kwargs["json"]
    assert body["text"] == "Hello"
    assert body["voice"] == "Kaveri"
    assert body["model"] == DEFAULT_MODEL
    assert body["language"] == "en-IN"
    assert body["speed"] == 1.0
    assert body["audio_config"] == DEFAULT_AUDIO_CONFIG


def test_default_audio_config_matches_step_spec() -> None:
    assert DEFAULT_AUDIO_CONFIG["sample_rate"] == 48000
    assert DEFAULT_AUDIO_CONFIG["num_channels"] == 1
    assert DEFAULT_AUDIO_CONFIG["sample_width"] == 2
    assert DEFAULT_AUDIO_CONFIG["encoding"] == "linear_pcm"
    assert DEFAULT_AUDIO_CONFIG["container"] == "wav"


def test_default_voice_is_documented_english_voice() -> None:
    assert timbre.DEFAULT_VOICE in ENGLISH_VOICES
    assert timbre.DEFAULT_LANGUAGE == "en-IN"
    assert timbre.DEFAULT_MODEL == "timbre-v2.5"


def test_missing_api_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(timbre, "get_gnani_api_key", lambda: None)
    client, session = make_client(api_key=None)

    with pytest.raises(TimbreConfigurationError):
        client.synthesize("Hello")

    session.post.assert_not_called()


def test_empty_text_raises_validation_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize("   ")

    session.post.assert_not_called()


def test_empty_voice_raises_validation_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize("Hello", voice="")

    session.post.assert_not_called()


def test_unsupported_language_raises_validation_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize("Hello", language="xx-XX")

    session.post.assert_not_called()


@pytest.mark.parametrize("speed", [0.5, 2.0, "turbo", True])
def test_unsupported_speed_raises_validation_error(speed: object) -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize("Hello", speed=speed)  # type: ignore[arg-type]

    session.post.assert_not_called()


@pytest.mark.parametrize("speed", [0.85, 1.0, 1.15, "slow", "medium", "fast"])
def test_supported_speed_values_are_accepted(speed: object) -> None:
    client, session = make_client(response=make_response())

    client.synthesize("Hello", speed=speed)  # type: ignore[arg-type]

    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"]["speed"] == speed


def test_unsupported_sample_rate_raises_validation_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize(
            "Hello",
            audio_config={**DEFAULT_AUDIO_CONFIG, "sample_rate": 12345},
        )

    session.post.assert_not_called()


def test_unsupported_container_raises_validation_error() -> None:
    client, session = make_client(response=make_response())

    with pytest.raises(TimbreValidationError):
        client.synthesize(
            "Hello",
            audio_config={**DEFAULT_AUDIO_CONFIG, "container": "flac"},
        )

    session.post.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 403, 429, 500, 503])
def test_http_error_raises_api_error_with_status(status_code: int) -> None:
    response = make_response(
        status_code=status_code,
        content_type="application/json",
        payload={
            "success": False,
            "error": {"type": "SOME_ERROR", "message": "Something went wrong."},
        },
    )
    client, _ = make_client(response=response)

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert excinfo.value.status_code == status_code
    assert "Something went wrong." in str(excinfo.value)
    assert excinfo.value.hint is not None


def test_forbidden_hint_mentions_api_key() -> None:
    response = make_response(
        status_code=403,
        content_type="application/json",
        payload={"success": False, "error": {"type": "FORBIDDEN", "message": "Nope."}},
    )
    client, _ = make_client(response=response)

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert "API key" in excinfo.value.hint


def test_json_body_with_http_200_raises_api_error() -> None:
    response = make_response(
        status_code=200,
        content_type="application/json",
        payload={
            "success": False,
            "error": {"type": "API_ERROR", "message": "boom"},
        },
    )
    client, _ = make_client(response=response)

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert "boom" in str(excinfo.value)


def test_empty_audio_body_raises_api_error() -> None:
    response = make_response(status_code=200, content=b"", content_type="audio/wav")
    client, _ = make_client(response=response)

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert "empty audio" in str(excinfo.value)


def test_non_json_error_body_raises_api_error() -> None:
    response = make_response(
        status_code=502,
        content=b"<html>bad gateway</html>",
        content_type="text/html",
        invalid_json=True,
    )
    client, _ = make_client(response=response)

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert "non-JSON" in str(excinfo.value)
    assert excinfo.value.status_code == 502


def test_connection_error_raises_network_error() -> None:
    client, _ = make_client(side_effect=requests.exceptions.ConnectionError("refused"))

    with pytest.raises(TimbreNetworkError):
        client.synthesize("Hello")


def test_timeout_raises_network_error() -> None:
    client, session = make_client(side_effect=requests.exceptions.Timeout("too slow"))

    pytest.raises(TimbreNetworkError, client.synthesize, "Hello")

    assert session.post.call_count == 1
    assert SLEEP_CALLS == []


def _rate_limit_response(status_code: int = 429) -> MagicMock:
    return make_response(
        status_code=status_code,
        content_type="application/json",
        payload={
            "success": False,
            "error": {"type": "RATE_LIMIT", "message": "slow down"},
        },
    )


def test_429_retried_then_succeeds() -> None:
    client, session = make_client(
        side_effect=[_rate_limit_response(), make_response()]
    )

    audio = client.synthesize("Hello")

    assert audio == FAKE_AUDIO
    assert session.post.call_count == 2
    assert len(SLEEP_CALLS) == 1


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_transient_5xx_retried_then_succeeds(status_code: int) -> None:
    client, session = make_client(
        side_effect=[_rate_limit_response(status_code), make_response()]
    )

    audio = client.synthesize("Hello")

    assert audio == FAKE_AUDIO
    assert session.post.call_count == 2


def test_retry_exhaustion_raises_last_api_error() -> None:
    client, session = make_client(
        side_effect=[_rate_limit_response(), _rate_limit_response(), _rate_limit_response()]
    )

    with pytest.raises(TimbreAPIError) as excinfo:
        client.synthesize("Hello")

    assert excinfo.value.status_code == 429
    assert session.post.call_count == 3
    assert len(SLEEP_CALLS) == 2


def test_non_retryable_status_is_single_attempt() -> None:
    response = make_response(
        status_code=400,
        content_type="application/json",
        payload={"success": False, "error": {"type": "BAD_REQUEST", "message": "bad"}},
    )
    client, session = make_client(side_effect=[response])

    with pytest.raises(TimbreAPIError):
        client.synthesize("Hello")

    assert session.post.call_count == 1
    assert SLEEP_CALLS == []


def test_connection_error_retried_then_succeeds() -> None:
    client, session = make_client(
        side_effect=[requests.exceptions.ConnectionError("refused"), make_response()]
    )

    audio = client.synthesize("Hello")

    assert audio == FAKE_AUDIO
    assert session.post.call_count == 2
    assert len(SLEEP_CALLS) == 1


def test_retry_honors_numeric_retry_after_header() -> None:
    response = _rate_limit_response()
    response.headers["Retry-After"] = "1"
    client, session = make_client(side_effect=[response, make_response()])

    audio = client.synthesize("Hello")

    assert audio == FAKE_AUDIO
    assert SLEEP_CALLS == [1.0]


def test_retry_after_header_is_capped() -> None:
    response = _rate_limit_response()
    response.headers["Retry-After"] = "300"
    client, session = make_client(side_effect=[response, make_response()])

    client.synthesize("Hello")

    assert SLEEP_CALLS == [10.0]


def test_invalid_retry_after_falls_back_to_backoff() -> None:
    response = _rate_limit_response()
    response.headers["Retry-After"] = "Thu, 01 Jan 2040 00:00:00 GMT"
    client, session = make_client(side_effect=[response, make_response()])

    client.synthesize("Hello")

    assert len(SLEEP_CALLS) == 1
    assert 0.5 <= SLEEP_CALLS[0] <= 0.75


def test_max_attempts_one_disables_retry() -> None:
    client, session = make_client(
        side_effect=[_rate_limit_response()], max_attempts=1
    )

    with pytest.raises(TimbreAPIError):
        client.synthesize("Hello")

    assert session.post.call_count == 1
    assert SLEEP_CALLS == []


def test_invalid_max_attempts_rejected() -> None:
    pytest.raises(ValueError, TimbreTTS, api_key="test-key", max_attempts=0)


def test_synthesize_to_file_writes_audio(tmp_path: Path) -> None:
    client, _ = make_client(response=make_response())
    target = tmp_path / "reply.wav"

    result = client.synthesize_to_file("Hello", target)

    assert result == target
    assert target.read_bytes() == FAKE_AUDIO


def test_synthesize_to_file_creates_missing_parent_dirs(tmp_path: Path) -> None:
    client, _ = make_client(response=make_response())
    target = tmp_path / "nested" / "deep" / "reply.wav"

    client.synthesize_to_file("Hello", target)

    assert target.read_bytes() == FAKE_AUDIO


def test_synthesize_to_file_appends_default_suffix(tmp_path: Path) -> None:
    client, _ = make_client(response=make_response())

    result = client.synthesize_to_file("Hello", tmp_path / "reply")

    assert result.name == "reply.wav"


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "medvaani_synthesize_cli", PROJECT_ROOT / "scripts" / "synthesize.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: None)

    exit_code = module.main(["Hello"])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_CONFIG
    assert "[FAIL] API key" in output
    assert "GNANI_API_KEY" in output


def test_cli_reports_empty_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")

    exit_code = module.main(["   "])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_REQUEST
    assert "[FAIL] Text" in output


def test_cli_reports_api_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")

    def failing_synthesize(*args: object, **kwargs: object) -> object:
        raise TimbreAPIError(
            "You do not have permission to access this resource.",
            status_code=403,
            error_type="FORBIDDEN",
        )

    monkeypatch.setattr(module.TimbreTTS, "synthesize_to_file", failing_synthesize)

    exit_code = module.main(["Hello"])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_API
    assert "[FAIL] API" in output
    assert "HTTP 403" in output
    assert "Hint:" in output


def test_cli_writes_audio_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_cli_module()
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: "test-key")
    target = tmp_path / "cli.wav"

    def fake_synthesize_to_file(self: object, text: str, output_path: object, **kwargs: object) -> Path:
        path = Path(output_path)
        path.write_bytes(FAKE_AUDIO)
        return path

    monkeypatch.setattr(module.TimbreTTS, "synthesize_to_file", fake_synthesize_to_file)

    exit_code = module.main(["Hello there", "--output", str(target)])

    output = capsys.readouterr().out
    assert exit_code == module.EXIT_OK
    assert "[ OK ] Request" in output
    assert "[ OK ] Saved" in output
    assert target.read_bytes() == FAKE_AUDIO


def test_cli_does_not_print_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = load_cli_module()
    secret = "super-secret-api-key-value"
    monkeypatch.setattr(module, "get_gnani_api_key", lambda: secret)

    def fake_synthesize_to_file(self: object, text: str, output_path: object, **kwargs: object) -> Path:
        path = Path(output_path)
        path.write_bytes(FAKE_AUDIO)
        return path

    monkeypatch.setattr(module.TimbreTTS, "synthesize_to_file", fake_synthesize_to_file)

    module.main(["Hello", "--output", str(tmp_path / "secret_check.wav")])

    output = capsys.readouterr().out
    assert secret not in output
