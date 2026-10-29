"""Gnani Prisma v2.5 speech-to-text adapter.

This module is the single boundary between MedVaani and the Gnani STT API.
Nothing outside this module should depend on the endpoint URL, header names,
form field names or response shape.

Source of truth for every constant and branch below:
https://docs.gnani.ai/api/STT/speech-to-text

    POST https://api.vachana.ai/stt/v3          (multipart/form-data)
    Header  X-API-Key-ID: <api key>
    Fields  audio_file (file), language_code (BCP-47) [, format]
    200     {"success": true, "request_id": ..., "timestamp": ..., "transcript": ...}
    error   {"success": false, "error": {"type": ..., "message": ...}}
            with HTTP 400, 403, 429, 500 or 503
"""

from __future__ import annotations

import mimetypes
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from app.config import GNANI_API_KEY_ENV_VAR, get_gnani_api_key

DEFAULT_ENDPOINT = "https://api.vachana.ai/stt/v3"
DEFAULT_LANGUAGE_CODE = "en-IN"
DEFAULT_OUTPUT_FORMAT = "verbatim"

REQUEST_TIMEOUT_SECONDS: tuple[int, int] = (10, 60)

# Transient STT failure conventions mirror the Timbre TTS adapter
# (app/services/timbre.py): the same bounded retry/backoff constants and the
# same provider gateway, adapted to the STT request lifecycle below.
RETRYABLE_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 5.0
JITTER_MAX_SECONDS = 0.25
MAX_RETRY_AFTER_SECONDS = 10.0

SUPPORTED_LANGUAGE_CODES: tuple[str, ...] = (
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
)

SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = ("verbatim", "transcribe")

SUPPORTED_AUDIO_SUFFIXES: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".aac",
    ".m4a",
)

STATUS_HINTS: dict[int, str] = {
    400: "The request was rejected. Check the audio format and duration (maximum 60 seconds).",
    403: "Access denied. Check the API key and the account's available credits.",
    429: "Rate limit exceeded. Wait a moment and retry.",
    500: "Gnani server error. Retry shortly.",
    503: "Gnani speech service is temporarily unavailable. Retry shortly.",
    504: "The speech gateway timed out. Retry shortly.",
}


class PrismaError(Exception):
    """Base class for every failure raised by the Prisma adapter."""


class PrismaConfigurationError(PrismaError):
    """Raised when the API key is missing from the environment."""


class PrismaFileError(PrismaError):
    """Raised when the audio file is missing or has an unsupported type."""


class PrismaValidationError(PrismaError):
    """Raised when a request parameter is outside the documented values."""


class PrismaNetworkError(PrismaError):
    """Raised when the HTTP request could not reach the Gnani API."""


class PrismaAPIError(PrismaError):
    """Raised when Gnani answers with an error status or an unusable body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        retry_after: float | None = None,
        body_head: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retry_after = retry_after
        # Optional truncated provider-body head for operator-side diagnostics
        # only. Never part of the public message, so no gateway HTML or
        # provider body text reaches users.
        self.body_head = body_head

    @property
    def hint(self) -> str | None:
        if self.status_code is None:
            return None
        return STATUS_HINTS.get(self.status_code)


@dataclass(frozen=True)
class TranscriptionResult:
    """Provider-neutral transcription outcome returned to callers."""

    text: str
    request_id: str | None = None
    model: str | None = None
    timestamp: str | None = None
    # Optional detected-language metadata. The documented REST /stt/v3 response
    # does not currently expose one, so this stays ``None`` unless a future
    # provider body includes it; callers must treat it as best-effort only.
    detected_language: str | None = None


def _parse_retry_after(headers: Any) -> float | None:
    """Parse a numeric ``Retry-After`` header into seconds (``None`` if unusable)."""
    try:
        raw = (headers or {}).get("Retry-After")
    except (AttributeError, TypeError):
        return None
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds < 0 or seconds != seconds:  # negative or NaN never guides a retry
        return None
    return seconds


def _extract_error(payload: Any, status_code: int) -> tuple[str, str | None]:
    message: str | None = None
    error_type: str | None = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            if isinstance(error.get("message"), str):
                message = error["message"]
            if isinstance(error.get("type"), str):
                error_type = error["type"]
        elif isinstance(error, str):
            message = error
        if message is None and isinstance(payload.get("message"), str):
            message = payload["message"]

    if message is None:
        message = f"Gnani STT returned HTTP {status_code}."
    return message, error_type


# Response keys a provider might use for detected-language metadata. The
# documented /stt/v3 body does not include any of them today, so this is a
# forward-compatible best-effort read rather than a contract.
_DETECTED_LANGUAGE_KEYS = ("detected_language", "language", "lang_code", "language_code")


def _extract_detected_language(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in _DETECTED_LANGUAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class PrismaSTT:
    """Transcribes audio files with Gnani Prisma v2.5 over the REST endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        session: requests.Session | None = None,
        timeout: float | tuple[float, float] = REQUEST_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self._api_key = api_key
        self._endpoint = endpoint
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._backoff_base_seconds = backoff_base_seconds

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def transcribe(
        self,
        audio_path: str | Path,
        language_code: str = DEFAULT_LANGUAGE_CODE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> TranscriptionResult:
        """Send one audio file to Gnani Prisma v2.5 and return its transcript.

        Transient upstream failures (HTTP 429/500/502/503/504 and connection
        errors) are retried with bounded backoff. Read timeouts are never
        retried because the outcome of the first request is unknown: the
        provider may already have processed and billed the transcription.
        Permanently rejected requests (400/401/403) and malformed bodies with
        a non-transient status raise immediately.
        """
        api_key = self._api_key if self._api_key is not None else get_gnani_api_key()
        if not api_key:
            raise PrismaConfigurationError(
                f"{GNANI_API_KEY_ENV_VAR} is not set. "
                f"Add {GNANI_API_KEY_ENV_VAR}=<your key> to the .env file at the repository root."
            )

        path = Path(audio_path)
        if not path.is_file():
            raise PrismaFileError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            supported = ", ".join(SUPPORTED_AUDIO_SUFFIXES)
            raise PrismaFileError(
                f"Unsupported audio type '{path.suffix}'. Supported types: {supported}."
            )
        if language_code not in SUPPORTED_LANGUAGE_CODES:
            raise PrismaValidationError(
                f"Unsupported language_code '{language_code}'. "
                f"Supported values: {', '.join(SUPPORTED_LANGUAGE_CODES)}."
            )
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise PrismaValidationError(
                f"Unsupported output_format '{output_format}'. "
                f"Supported values: {', '.join(SUPPORTED_OUTPUT_FORMATS)}."
            )

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        last_error: PrismaError | None = None
        for attempt in range(self._max_attempts):
            if attempt > 0:
                self._sleep_before_retry(attempt, last_error)

            try:
                # Reopened per attempt: a handle is exhausted after it is
                # passed once to POST, so each retry must replay the same
                # bytes through a fresh handle.
                with path.open("rb") as handle:
                    response = self._session.post(
                        self._endpoint,
                        headers={"X-API-Key-ID": api_key},
                        data={"language_code": language_code, "format": output_format},
                        files={"audio_file": (path.name, handle, content_type)},
                        timeout=self._timeout,
                    )
            except requests.exceptions.Timeout as exc:
                raise PrismaNetworkError(
                    f"Gnani STT request timed out after {self._timeout}: {exc}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                # The request was never delivered, so nothing was billed.
                last_error = PrismaNetworkError(f"Gnani STT request failed: {exc}")
                continue
            except requests.exceptions.RequestException as exc:
                raise PrismaNetworkError(f"Gnani STT request failed: {exc}") from exc

            try:
                return self._parse_response(response)
            except PrismaAPIError as exc:
                if exc.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                last_error = exc
                continue

        assert last_error is not None
        raise last_error

    def _sleep_before_retry(
        self,
        attempt: int,
        last_error: PrismaError | None,
    ) -> None:
        """Bounded exponential backoff with jitter, honoring Retry-After."""
        retry_after = getattr(last_error, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after >= 0:
            delay = min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
        else:
            delay = min(
                self._backoff_base_seconds * (2.0 ** (attempt - 1)),
                MAX_BACKOFF_SECONDS,
            )
            delay += random.uniform(0, JITTER_MAX_SECONDS)
        time.sleep(max(delay, 0.0))

    def _parse_response(self, response: requests.Response) -> TranscriptionResult:
        retry_after = _parse_retry_after(response.headers)
        try:
            payload = response.json()
        except ValueError:
            # Diagnostic head for operator logs only: the public message stays
            # concise and never echoes provider/gateway body content.
            body_head = (response.text or "").strip().replace("\n", " ")[:200]
            raise PrismaAPIError(
                f"Gnani STT returned a non-JSON response (HTTP {response.status_code}).",
                status_code=response.status_code,
                retry_after=retry_after,
                body_head=body_head,
            ) from None

        if response.status_code != 200 or not isinstance(payload, dict):
            message, error_type = _extract_error(payload, response.status_code)
            raise PrismaAPIError(
                message, status_code=response.status_code, error_type=error_type,
                retry_after=retry_after,
            )

        if payload.get("success") is False:
            message, error_type = _extract_error(payload, response.status_code)
            raise PrismaAPIError(
                message, status_code=response.status_code, error_type=error_type,
                retry_after=retry_after,
            )

        transcript = payload.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise PrismaAPIError(
                "Gnani STT returned an empty transcript.",
                status_code=response.status_code,
            )

        request_id = payload.get("request_id")
        model = payload.get("model")
        timestamp = payload.get("timestamp")
        detected_language = _extract_detected_language(payload)

        return TranscriptionResult(
            text=transcript.strip(),
            request_id=request_id if isinstance(request_id, str) else None,
            model=model if isinstance(model, str) else None,
            timestamp=timestamp if isinstance(timestamp, str) else None,
            detected_language=detected_language,
        )
