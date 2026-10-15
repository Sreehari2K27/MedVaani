"""MedVaani minimal FastAPI voice round-trip integration.

Audio -> Gnani Prisma v2.5 (transcription) -> deterministic text
-> Gnani Timbre v2.5 (synthesis) -> saved audio file.

This is a smoke-test integration only. There is deliberately no LLM, Evon,
Mistral, safety engine or workflow logic in this module.

The Prisma and Timbre adapters are reused unchanged; this module only
orchestrates them and maps their errors to HTTP status codes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.services.prisma import (
    DEFAULT_LANGUAGE_CODE,
    PrismaSTT,
)
from app.services.timbre import (
    TimbreTTS,
)
from app.api.errors import (
    CONFIGURATION_ERRORS,
    UPSTREAM_ERRORS,
    VALIDATION_ERRORS,
)
from app.api.voice_conversation import create_voice_conversation_router
from app.api.mi_chat import create_mi_chat_router

ROUND_TRIP_RESPONSE_TEXT = "Thank you. I received your message through MedVaani."

OUTPUT_AUDIO_PATH = Path("output") / "round_trip.wav"


def _save_upload(audio_file: UploadFile) -> Path:
    suffix = Path(audio_file.filename or "").suffix
    upload = audio_file.file.read()
    if not upload:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(upload)
        return Path(handle.name)


def create_app(
    prisma: PrismaSTT | None = None,
    timbre: TimbreTTS | None = None,
) -> FastAPI:
    """Build the FastAPI app, optionally with injected service clients."""
    application = FastAPI(title="MedVaani Voice Round-Trip")
    prisma_service = prisma if prisma is not None else PrismaSTT()
    timbre_service = timbre if timbre is not None else TimbreTTS()

    @application.post("/voice/round-trip")
    def voice_round_trip(
        audio_file: UploadFile = File(...),
        language: str = Form(DEFAULT_LANGUAGE_CODE),
    ) -> dict[str, str]:
        tmp_path = _save_upload(audio_file)
        try:
            transcription = prisma_service.transcribe(
                tmp_path,
                language_code=language,
            )

            audio_path = timbre_service.synthesize_to_file(
                ROUND_TRIP_RESPONSE_TEXT,
                OUTPUT_AUDIO_PATH,
                language=language,
            )
        except VALIDATION_ERRORS as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CONFIGURATION_ERRORS as exc:
            raise HTTPException(
                status_code=500, detail=f"MedVaani services are not configured: {exc}"
            ) from exc
        except UPSTREAM_ERRORS as exc:
            raise HTTPException(
                status_code=502, detail=f"Gnani service failure: {exc}"
            ) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        return {
            "transcript": transcription.text,
            "response_text": ROUND_TRIP_RESPONSE_TEXT,
            "audio_path": str(audio_path),
            "language": language,
        }

    application.include_router(
        create_voice_conversation_router(
            stt=prisma_service,
            tts=timbre_service,
            ask_additional_information=True,
        )
    )

    # Dedicated MI chat endpoints (text-only answers, Prisma STT for the
    # microphone; no TTS and no AE session state).
    application.include_router(create_mi_chat_router(stt=prisma_service))

    # Serve generated TTS audio (output/) so clients can play response audio.
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    application.mount("/audio", StaticFiles(directory=output_dir), name="audio")

    return application


app = create_app()
