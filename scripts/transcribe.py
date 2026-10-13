"""Minimal CLI check for the Gnani Prisma v2.5 speech-to-text integration.

Usage:
    python scripts/transcribe.py path/to/audio.wav --language hi-IN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import GNANI_API_KEY_ENV_VAR, get_gnani_api_key
from app.services.prisma import (
    DEFAULT_ENDPOINT,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_OUTPUT_FORMAT,
    SUPPORTED_OUTPUT_FORMATS,
    PrismaAPIError,
    PrismaConfigurationError,
    PrismaError,
    PrismaFileError,
    PrismaNetworkError,
    PrismaSTT,
    PrismaValidationError,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_FILE = 3
EXIT_REQUEST = 4
EXIT_API = 5
EXIT_NETWORK = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/transcribe.py",
        description="Send a local audio file to Gnani Prisma v2.5 and print the transcript.",
    )
    parser.add_argument("audio_file", help="Path to a WAV, MP3, OGG, FLAC, AAC or M4A file (max 60 seconds)")
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE_CODE,
        help=f"BCP-47 language code (default: {DEFAULT_LANGUAGE_CODE})",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=SUPPORTED_OUTPUT_FORMATS,
        help=f"Transcript format (default: {DEFAULT_OUTPUT_FORMAT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("MedVaani - Gnani Prisma v2.5 speech-to-text check")
    print(f"  Endpoint : {DEFAULT_ENDPOINT}")
    print(f"  Language : {args.language}")
    print(f"  Format   : {args.output_format}")
    print(f"  Audio    : {args.audio_file}")
    print()

    if not get_gnani_api_key():
        print(f"[FAIL] API key : {GNANI_API_KEY_ENV_VAR} is not set.")
        print(f"             Add {GNANI_API_KEY_ENV_VAR}=<your key> to the .env file at the repository root.")
        return EXIT_CONFIG
    print(f"[ OK ] API key : {GNANI_API_KEY_ENV_VAR} loaded from the environment.")

    path = Path(args.audio_file)
    if not path.is_file():
        print(f"[FAIL] Audio   : file not found -> {path}")
        return EXIT_FILE
    print(f"[ OK ] Audio   : {path.name} ({path.stat().st_size:,} bytes)")

    print(f"[....] Request : sending audio to {DEFAULT_ENDPOINT}")
    client = PrismaSTT()

    try:
        result = client.transcribe(
            path,
            language_code=args.language,
            output_format=args.output_format,
        )
    except PrismaConfigurationError as exc:
        print(f"[FAIL] API key : {exc}")
        return EXIT_CONFIG
    except PrismaFileError as exc:
        print(f"[FAIL] Audio   : {exc}")
        return EXIT_FILE
    except PrismaValidationError as exc:
        print(f"[FAIL] Request : {exc}")
        return EXIT_REQUEST
    except PrismaNetworkError as exc:
        print(f"[FAIL] Network : {exc}")
        print("             Could not reach the Gnani API. Check your connection and retry.")
        return EXIT_NETWORK
    except PrismaAPIError as exc:
        status = f"HTTP {exc.status_code}" if exc.status_code else "API error"
        print(f"[FAIL] API     : {status} - {exc}")
        if exc.hint:
            print(f"             Hint: {exc.hint}")
        return EXIT_API
    except PrismaError as exc:
        print(f"[FAIL] Other   : {exc}")
        return EXIT_API

    details = [f"request_id={result.request_id}"] if result.request_id else []
    if result.model:
        details.append(f"model={result.model}")
    suffix = f" ({', '.join(details)})" if details else ""
    print(f"[ OK ] Request : transcription succeeded{suffix}")
    print()
    print("TRANSCRIPT")
    print("----------")
    print(result.text)
    print("----------")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
