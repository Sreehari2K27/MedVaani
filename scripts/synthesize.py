"""Minimal CLI check for the Gnani Timbre v2.5 text-to-speech integration.

Usage:
    python scripts/synthesize.py "Your text here"
    python scripts/synthesize.py "Your text here" --voice Pranav --output output\reply.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import GNANI_API_KEY_ENV_VAR, get_gnani_api_key
from app.services.timbre import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_ENDPOINT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    ENGLISH_VOICES,
    TimbreAPIError,
    TimbreConfigurationError,
    TimbreError,
    TimbreNetworkError,
    TimbreTTS,
    TimbreValidationError,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_OUTPUT = 3
EXIT_REQUEST = 4
EXIT_API = 5
EXIT_NETWORK = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/synthesize.py",
        description="Convert text to speech with Gnani Timbre v2.5 and write the audio to a file.",
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path for the generated audio file (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Timbre voice name (default: {DEFAULT_VOICE}). English voices: {', '.join(ENGLISH_VOICES)}",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"BCP-47 language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--speed",
        default=str(DEFAULT_SPEED),
        help=f"Playback speed 0.85 to 1.15, or slow/medium/fast (default: {DEFAULT_SPEED})",
    )
    return parser


def _coerce_speed(raw: str) -> float | str:
    try:
        return float(raw)
    except ValueError:
        return raw


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    speed = _coerce_speed(str(args.speed))

    config = DEFAULT_AUDIO_CONFIG
    print("MedVaani - Gnani Timbre v2.5 text-to-speech check")
    print(f"  Endpoint  : {DEFAULT_ENDPOINT}")
    print(f"  Model     : {DEFAULT_MODEL}")
    print(f"  Voice     : {args.voice}")
    print(f"  Language  : {args.language}")
    print(f"  Speed     : {speed}")
    print(f"  Audio     : {config['sample_rate']} Hz, "
          f"{config['num_channels']} ch, "
          f"{config['sample_width'] * 8}-bit, "
          f"{config['encoding']}, {config['container']}")
    print(f"  Output    : {args.output}")
    print(f"  Characters: {len(args.text)}")
    print()

    if not get_gnani_api_key():
        print(f"[FAIL] API key : {GNANI_API_KEY_ENV_VAR} is not set.")
        print(f"             Add {GNANI_API_KEY_ENV_VAR}=<your key> to the .env file at the repository root.")
        return EXIT_CONFIG
    print(f"[ OK ] API key : {GNANI_API_KEY_ENV_VAR} loaded from the environment.")

    if not args.text.strip():
        print("[FAIL] Text    : nothing to synthesize. Pass some text as the first argument.")
        return EXIT_REQUEST
    print(f"[ OK ] Text    : {len(args.text)} characters accepted.")

    print(f"[....] Request : sending text to {DEFAULT_ENDPOINT}")
    client = TimbreTTS()

    try:
        path = client.synthesize_to_file(
            args.text,
            args.output,
            voice=args.voice,
            language=args.language,
            speed=speed,
        )
    except TimbreConfigurationError as exc:
        print(f"[FAIL] API key : {exc}")
        return EXIT_CONFIG
    except TimbreValidationError as exc:
        print(f"[FAIL] Request : {exc}")
        return EXIT_REQUEST
    except TimbreNetworkError as exc:
        print(f"[FAIL] Network : {exc}")
        print("             Could not reach the Gnani API. Check your connection and retry.")
        return EXIT_NETWORK
    except TimbreAPIError as exc:
        status = f"HTTP {exc.status_code}" if exc.status_code else "API error"
        print(f"[FAIL] API     : {status} - {exc}")
        if exc.hint:
            print(f"             Hint: {exc.hint}")
        return EXIT_API
    except TimbreError as exc:
        print(f"[FAIL] Other   : {exc}")
        return EXIT_API

    size = path.stat().st_size
    print(f"[ OK ] Request : synthesis succeeded ({size:,} bytes of audio)")
    print(f"[ OK ] Saved   : {path.resolve()}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
