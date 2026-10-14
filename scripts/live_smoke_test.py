"""Manual live smoke test for the Gnani Prisma STT and Timbre TTS adapters.

Run this by hand only - never as part of pytest (the offline suite must stay
network-free):

    python scripts/live_smoke_test.py

Exactly one fixed phrase is synthesized with the documented Timbre defaults
(voice Kaveri, model timbre-v2.5, language en-IN), written under the
git-ignored output/ directory, validated (non-empty RIFF/WAVE file) and then
transcribed once with Prisma (en-IN). That is one TTS call and one STT call;
any retry behaviour is whatever the adapters already implement. The personal
recordings under tests/ are never used.

The API key is loaded through the existing configuration mechanism and is
never printed - not even in part. Every printed error string is filtered
through a masking helper as an extra guard.

Exit codes: 0 = both services produced usable output; 1 = a service failed;
2 = GNANI_API_KEY is not configured (no request is made in that case).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import GNANI_API_KEY_ENV_VAR, get_gnani_api_key
from app.services.prisma import DEFAULT_LANGUAGE_CODE as STT_LANGUAGE_CODE
from app.services.prisma import PrismaSTT
from app.services.timbre import DEFAULT_VOICE, TimbreTTS

PHRASE = "This is a MedVaani smoke test."
OUTPUT_PATH = Path("output") / "live_smoke_test.wav"


def _safe(text: str, secret: str | None) -> str:
    """Return ``text`` with any accidental occurrence of the key masked."""
    if secret:
        text = text.replace(secret, "***")
    return text


def _has_wave_header(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def main() -> int:
    print("MedVaani live smoke test (Gnani Timbre TTS + Gnani Prisma STT)")
    print(f"  phrase : {PHRASE}")
    print(f"  output : {OUTPUT_PATH}")

    secret = get_gnani_api_key()
    if not secret:
        print(f"FAIL config: {GNANI_API_KEY_ENV_VAR} is not set - no request was made.")
        print(
            f"  Copy .env.example to .env at the repository root and set "
            f"{GNANI_API_KEY_ENV_VAR}=<your key>, or export {GNANI_API_KEY_ENV_VAR} "
            "in the environment. The .env file is git-ignored; never commit it."
        )
        return 2
    print("  api key: configured (value never printed)")

    # ---- Step 1: one TTS call with the documented defaults -----------------
    print("[1/3] Timbre TTS: synthesizing one phrase ...")
    try:
        tts = TimbreTTS()
        audio_path = tts.synthesize_to_file(
            PHRASE,
            OUTPUT_PATH,
            voice=DEFAULT_VOICE,
            language="en-IN",
        )
        audio = audio_path.read_bytes()
    except Exception as exc:  # TimbreError and anything unexpected
        detail = str(exc).strip() or type(exc).__name__
        print(f"FAIL tts: {_safe(f'{type(exc).__name__}: {detail}', secret)}")
        return 1

    if not audio:
        print(f"FAIL tts: {audio_path} is empty (0 bytes).")
        return 1
    if not _has_wave_header(audio):
        print(
            f"FAIL tts: {audio_path} does not start with a RIFF/WAVE header "
            f"({len(audio)} bytes)."
        )
        return 1
    print(f"      PASS: wrote {len(audio)} bytes of WAV audio to {audio_path}")

    # ---- Step 2: one STT call on that same file ----------------------------
    print(f"[2/3] Prisma STT: transcribing {audio_path} ({STT_LANGUAGE_CODE}) ...")
    try:
        stt = PrismaSTT()
        result = stt.transcribe(audio_path, language_code=STT_LANGUAGE_CODE)
    except Exception as exc:  # PrismaError and anything unexpected
        detail = str(exc).strip() or type(exc).__name__
        print(f"FAIL stt: {_safe(f'{type(exc).__name__}: {detail}', secret)}")
        return 1

    transcript = (result.text or "").strip()
    if not transcript:
        print("FAIL stt: the transcript came back empty.")
        return 1
    print(f"      PASS: transcript = {_safe(transcript, secret)!r}")

    # ---- Step 3: summary ---------------------------------------------------
    print("[3/3] Result")
    print("PASS: Timbre TTS produced valid WAV audio and Prisma STT")
    print("      returned a non-empty transcript for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
