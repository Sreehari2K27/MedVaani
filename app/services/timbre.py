"""Gnani Timbre v2.5 text-to-speech adapter.

This module is the single boundary between MedVaani and the Gnani TTS API.
Nothing outside this module should depend on the endpoint URL, header names,
request body fields or response shape.

Source of truth for every constant and branch below:
https://docs.gnani.ai/api/TTS/tts-inference
https://docs.gnani.ai/api/TTS/available-voices

    POST https://api.vachana.ai/api/v1/tts/inference     (application/json)
    Header  X-API-Key-ID: <api key>
    Body    {"text", "voice", "model", "language", "speed", "audio_config"}
    200     raw binary audio matching audio_config.container
    error   {"success": false, "error": {"type": ..., "message": ...}}
            with HTTP 400, 403, 429, 500 or 503
"""

from __future__ import annotations

import math
import random
import struct
import time
from pathlib import Path
from typing import Any

import requests

from app.config import GNANI_API_KEY_ENV_VAR, get_gnani_api_key

DEFAULT_ENDPOINT = "https://api.vachana.ai/api/v1/tts/inference"
DEFAULT_MODEL = "timbre-v2.5"
DEFAULT_VOICE = "Kaveri"
DEFAULT_LANGUAGE = "en-IN"
DEFAULT_SPEED = 1.0

DEFAULT_AUDIO_CONFIG: dict[str, Any] = {
    "sample_rate": 48000,
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",
    "container": "wav",
}

DEFAULT_OUTPUT_SUFFIX = ".wav"
DEFAULT_OUTPUT_PATH = Path("output") / f"synthesized{DEFAULT_OUTPUT_SUFFIX}"

REQUEST_TIMEOUT_SECONDS: tuple[int, int] = (10, 120)

RETRYABLE_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 5.0
JITTER_MAX_SECONDS = 0.25
MAX_RETRY_AFTER_SECONDS = 10.0

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "auto",
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-en",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
)

ENGLISH_VOICES: tuple[str, ...] = (
    "Kaveri",
    "Trupti",
    "Devika",
    "Pranav",
    "Shlok",
    "Girish",
)

SPEED_SHORTCUTS: tuple[str, ...] = ("slow", "medium", "fast")
MIN_SPEED = 0.85
MAX_SPEED = 1.15

SUPPORTED_SAMPLE_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000, 44100, 48000)
SUPPORTED_ENCODINGS: tuple[str, ...] = (
    "linear_pcm",
    "pcm_s16le",
    "pcm_mulaw",
    "pcm_alaw",
    "oggopus",
)
SUPPORTED_CONTAINERS: tuple[str, ...] = ("raw", "mp3", "wav", "mulaw", "alaw", "ogg")

STATUS_HINTS: dict[int, str] = {
    400: "The request was rejected. Check the text, voice name and audio configuration.",
    403: "Access denied. Check the API key and the account's available credits.",
    429: "Rate limit exceeded. Wait a moment and retry.",
    500: "Gnani server error. Retry shortly.",
    503: "Gnani speech service is temporarily unavailable. Retry shortly.",
}


class TimbreError(Exception):
    """Base class for every failure raised by the Timbre adapter."""


class TimbreConfigurationError(TimbreError):
    """Raised when the API key is missing from the environment."""


class TimbreValidationError(TimbreError):
    """Raised when a request parameter is outside the documented values."""


class TimbreNetworkError(TimbreError):
    """Raised when the HTTP request could not reach the Gnani API."""


class TimbreAPIError(TimbreError):
    """Raised when Gnani answers with an error status or an unusable body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.retry_after = retry_after

    @property
    def hint(self) -> str | None:
        if self.status_code is None:
            return None
        return STATUS_HINTS.get(self.status_code)


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
        message = f"Gnani TTS returned HTTP {status_code}."
    return message, error_type


def _parse_retry_after(headers: Any) -> float | None:
    """Parse a numeric ``Retry-After`` header into seconds (``None`` if unusable)."""
    raw = (headers or {}).get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _validate_audio_config(audio_config: dict[str, Any]) -> None:
    sample_rate = audio_config.get("sample_rate")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise TimbreValidationError(
            f"Unsupported sample_rate '{sample_rate}'. "
            f"Supported values: {', '.join(str(r) for r in SUPPORTED_SAMPLE_RATES)}."
        )

    num_channels = audio_config.get("num_channels")
    if not isinstance(num_channels, int) or num_channels < 1:
        raise TimbreValidationError(
            f"Unsupported num_channels '{num_channels}'. Use 1 for mono."
        )

    sample_width = audio_config.get("sample_width")
    if not isinstance(sample_width, int) or not 1 <= sample_width <= 4:
        raise TimbreValidationError(
            f"Unsupported sample_width '{sample_width}'. Use 2 for 16-bit audio."
        )

    encoding = audio_config.get("encoding")
    if encoding not in SUPPORTED_ENCODINGS:
        raise TimbreValidationError(
            f"Unsupported encoding '{encoding}'. "
            f"Supported values: {', '.join(SUPPORTED_ENCODINGS)}."
        )

    container = audio_config.get("container")
    if container not in SUPPORTED_CONTAINERS:
        raise TimbreValidationError(
            f"Unsupported container '{container}'. "
            f"Supported values: {', '.join(SUPPORTED_CONTAINERS)}."
        )


def _validate_speed(speed: object) -> None:
    if isinstance(speed, str):
        if speed not in SPEED_SHORTCUTS:
            raise TimbreValidationError(
                f"Unsupported speed '{speed}'. "
                f"Use a number between {MIN_SPEED} and {MAX_SPEED}, "
                f"or one of: {', '.join(SPEED_SHORTCUTS)}."
            )
        return

    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise TimbreValidationError(
            f"Unsupported speed '{speed}'. "
            f"Use a number between {MIN_SPEED} and {MAX_SPEED}, "
            f"or one of: {', '.join(SPEED_SHORTCUTS)}."
        )

    if not MIN_SPEED <= float(speed) <= MAX_SPEED:
        raise TimbreValidationError(
            f"Unsupported speed '{speed}'. Allowed range is {MIN_SPEED} to {MAX_SPEED}."
        )


# ---- peak-safe loudness normalization ---------------------------------------
#
# The provider's speech audio measures hot peaks (76-95% of full scale) with
# a quiet body (RMS 12-17%), so it sounds barely audible even at maximum
# playback volume while any linear gain above ~1.3x would clip the peaks.
# The fix raises the RMS toward a conservative target with a soft limiter
# that hard-caps the peaks below full scale - no clipping, no distortion,
# no amplification of already-loud audio. Pure-Python and deterministic; the
# WAV structure (header, chunk layout, sample format) is preserved verbatim.
_NORMALIZE_TARGET_RMS = 0.20  # of full scale
_NORMALIZE_PEAK_CAP = 0.95  # of full scale
_NORMALIZE_LIMITER_KNEE = 0.70  # of the peak cap
_MIN_APPLIED_GAIN = 1.02  # below this the change is inaudible: passthrough
_MAX_APPLIED_GAIN = 4.0  # sanity bound for near-silent payloads


def _soft_limit(value: float, knee: float, peak: float) -> float:
    """Smoothly compress values between the knee and the peak cap."""
    excess = abs(value) - knee
    return math.copysign(knee + (peak - knee) * math.tanh(excess / (peak - knee)), value)


def _normalize_wav_loudness(audio: bytes) -> bytes:
    """Raise quiet 16-bit PCM WAV speech to the loudness target without
    clipping. Non-WAV, non-PCM, malformed or already-loud payloads pass
    through byte-for-byte; the WAV structure is never rebuilt, only the
    sample bytes inside the data chunk change. The gain targets the RMS and
    the soft limiter (not the peak cap) bounds the hot speech peaks."""
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return audio
    fmt = None
    data = None
    offset = 12
    while offset + 8 <= len(audio):
        chunk_id = audio[offset : offset + 4]
        chunk_size = struct.unpack("<I", audio[offset + 4 : offset + 8])[0]
        body_start = offset + 8
        body_end = min(body_start + chunk_size, len(audio))
        if chunk_id == b"fmt " and body_end - body_start >= 16:
            fmt = struct.unpack("<HHIIHH", audio[body_start : body_start + 16])
        elif chunk_id == b"data":
            data = (body_start, body_end)
        offset = body_start + chunk_size + (chunk_size & 1)
    if fmt is None or data is None:
        return audio
    audio_format, channels, _rate, _byterate, _block_align, bits = fmt
    if audio_format != 1 or bits != 16 or channels not in (1, 2):
        return audio
    start, end = data
    sample_bytes = audio[start:end]
    if not sample_bytes:
        return audio
    count = len(sample_bytes) // 2
    samples = list(struct.unpack(f"<{count}h", sample_bytes[: count * 2]))
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return audio
    rms = math.sqrt(sum(s * s for s in samples) / count)
    cap_abs = _NORMALIZE_PEAK_CAP * 32767.0
    rms_frac = rms / 32767.0
    if rms_frac >= _NORMALIZE_TARGET_RMS:
        return audio  # already at or above the target loudness
    gain = min(_NORMALIZE_TARGET_RMS / rms_frac, _MAX_APPLIED_GAIN)
    if gain <= _MIN_APPLIED_GAIN:
        return audio  # the change would be inaudible: leave it untouched
    knee = _NORMALIZE_LIMITER_KNEE * cap_abs
    limited = []
    for sample in samples:
        amplified = sample * gain
        if abs(amplified) > knee:
            amplified = _soft_limit(amplified, knee, cap_abs)
        limited.append(int(amplified))
    normalized = struct.pack(f"<{count}h", *limited)
    return audio[:start] + normalized + audio[end:]


class TimbreTTS:
    """Synthesizes speech with Gnani Timbre v2.5 over the REST endpoint."""

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

    def synthesize(
        self,
        text: str,
        *,
        voice: str = DEFAULT_VOICE,
        language: str = DEFAULT_LANGUAGE,
        speed: float | str = DEFAULT_SPEED,
        audio_config: dict[str, Any] | None = None,
    ) -> bytes:
        """Convert text to speech and return the raw audio bytes.

        Transient upstream failures (HTTP 429/500/502/503/504 and connection
        errors) are retried with bounded exponential backoff. Read timeouts are
        never retried because the outcome of the first request is unknown.
        """
        api_key = self._api_key if self._api_key is not None else get_gnani_api_key()
        if not api_key:
            raise TimbreConfigurationError(
                f"{GNANI_API_KEY_ENV_VAR} is not set. "
                f"Add {GNANI_API_KEY_ENV_VAR}=<your key> to the .env file at the repository root."
            )

        if not isinstance(text, str) or not text.strip():
            raise TimbreValidationError("Text to synthesize must not be empty.")
        if not isinstance(voice, str) or not voice.strip():
            raise TimbreValidationError(
                f"Voice must not be empty. English voices: {', '.join(ENGLISH_VOICES)}."
            )
        if language not in SUPPORTED_LANGUAGES:
            raise TimbreValidationError(
                f"Unsupported language '{language}'. "
                f"Supported values: {', '.join(SUPPORTED_LANGUAGES)}."
            )
        _validate_speed(speed)

        resolved_config = dict(DEFAULT_AUDIO_CONFIG if audio_config is None else audio_config)
        _validate_audio_config(resolved_config)

        payload = {
            "text": text,
            "voice": voice,
            "model": DEFAULT_MODEL,
            "language": language,
            "speed": speed,
            "audio_config": resolved_config,
        }

        last_error: TimbreError | None = None
        for attempt in range(self._max_attempts):
            if attempt > 0:
                self._sleep_before_retry(attempt, last_error)

            try:
                response = self._session.post(
                    self._endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key-ID": api_key,
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise TimbreNetworkError(
                    f"Gnani TTS request timed out after {self._timeout}: {exc}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = TimbreNetworkError(f"Gnani TTS request failed: {exc}")
                continue
            except requests.exceptions.RequestException as exc:
                raise TimbreNetworkError(f"Gnani TTS request failed: {exc}") from exc

            try:
                return self._parse_response(response)
            except TimbreAPIError as exc:
                if exc.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                last_error = exc
                continue

        assert last_error is not None
        raise last_error

    def _sleep_before_retry(
        self,
        attempt: int,
        last_error: TimbreError | None,
    ) -> None:
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

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        """Convert text to speech, write the audio to disk and return the path."""
        audio = _normalize_wav_loudness(self.synthesize(text, **kwargs))

        path = Path(output_path)
        if path.is_dir():
            path = path / f"synthesized{DEFAULT_OUTPUT_SUFFIX}"
        if not path.suffix:
            path = path.with_suffix(DEFAULT_OUTPUT_SUFFIX)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
        except OSError as exc:
            raise TimbreValidationError(
                f"Could not write audio to {path}: {exc}"
            ) from exc

        return path

    def _parse_response(self, response: requests.Response) -> bytes:
        content_type = (response.headers.get("Content-Type") or "").lower()

        if response.status_code != 200:
            try:
                payload = response.json()
            except ValueError:
                body = (response.text or "").strip().replace("\n", " ")[:200]
                raise TimbreAPIError(
                    f"Gnani TTS returned a non-JSON error body "
                    f"(HTTP {response.status_code}): {body}",
                    status_code=response.status_code,
                    retry_after=_parse_retry_after(response.headers),
                ) from None
            message, error_type = _extract_error(payload, response.status_code)
            raise TimbreAPIError(
                message,
                status_code=response.status_code,
                error_type=error_type,
                retry_after=_parse_retry_after(response.headers),
            )

        if "application/json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            message, error_type = _extract_error(payload, response.status_code)
            raise TimbreAPIError(
                message, status_code=response.status_code, error_type=error_type
            )

        audio = response.content
        if not audio:
            raise TimbreAPIError(
                "Gnani TTS returned an empty audio response.",
                status_code=response.status_code,
            )
        return audio
