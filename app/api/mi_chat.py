"""Dedicated Medical Information chat endpoints (text-only answers).

Two narrowly scoped routes for the MI chat interface:

- ``POST /mi/chat``: urgent-safety check first (existing ``SafetyLayer``),
  then the existing ``MIGroundedService`` (guardrails -> retrieval -> Mistral
  when ``MISTRAL_MI_ENABLED`` -> sentence-level grounding -> deterministic
  extractive fallback). The response model exposes only safe, MI-specific
  fields - never an AE session, case, audio path or provider detail - so this
  endpoint is structurally incapable of reading or mutating AE report state
  (it owns no session store and no engine).

- ``POST /mi/transcribe``: speech-to-text only via the existing ``PrismaSTT``
  adapter, with the same error-to-HTTP conventions as the voice endpoints.
  Timbre TTS is never invoked and no audio response is produced.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.errors import (
    CONFIGURATION_ERRORS,
    UPSTREAM_ERRORS,
    VALIDATION_ERRORS,
)
from app.safety import SafetyLayer, SafetySignal
from app.services.prisma import DEFAULT_LANGUAGE_CODE, PrismaSTT


class MiChatRequest(BaseModel):
    question: str = ""


class MiChatResponse(BaseModel):
    """The complete /mi/chat response surface (text-only answers)."""

    answer: str
    status: str
    category: str
    grounded: bool
    supported: bool
    source_page_numbers: List[int]
    retrieved_chunk_ids: List[str]
    urgent: bool
    suggest_ae_report: bool


class MiTranscribeResponse(BaseModel):
    transcript: str


def _upstream_http_error(exc: Exception) -> HTTPException:
    """Map one Prisma failure to the voice endpoints' HTTP conventions."""
    status = getattr(exc, "status_code", None)
    if status == 200 and "empty transcript" in str(exc).lower():
        return HTTPException(
            status_code=400,
            detail="The recording produced no speech. Please try again.",
        )
    if status == 403:
        return HTTPException(
            status_code=500,
            detail=(
                "Speech service authentication failed. The operator must check "
                "the API key and the account's available credits."
            ),
        )
    if status == 429:
        return HTTPException(
            status_code=429,
            detail="Speech service rate limit exceeded. Wait a moment and retry.",
        )
    return HTTPException(status_code=502, detail=f"Gnani service failure: {exc}")


def create_mi_chat_router(
    stt: PrismaSTT | None = None,
    mi_service=None,
) -> APIRouter:
    """Build the MI chat router, optionally with injected services (tests)."""
    router = APIRouter()

    def _service():
        # Lazy: the cached MI service (Mistral when enabled) is built on the
        # first chat request, never at import time.
        if mi_service is not None:
            return mi_service
        from app.api.voice_conversation import get_mi_service

        return get_mi_service()

    @router.post("/mi/chat", response_model=MiChatResponse)
    def mi_chat(payload: MiChatRequest) -> MiChatResponse:
        question = (payload.question or "").strip()
        if not question:
            raise HTTPException(
                status_code=400, detail="A question is required."
            )

        # Urgent symptoms take precedence over routine product information.
        # There is no engine here: the safety message is returned as text and
        # nothing is recorded.
        safety = SafetyLayer.classify_input(question)
        if safety.signal is SafetySignal.URGENT_LIFE_THREATENING:
            return MiChatResponse(
                answer=safety.message,
                status="urgent",
                category="urgent",
                grounded=False,
                supported=False,
                source_page_numbers=[],
                retrieved_chunk_ids=[],
                urgent=True,
                suggest_ae_report=False,
            )

        result = _service().handle(question)
        return MiChatResponse(
            answer=result.answer,
            status=result.status,
            category=result.category,
            grounded=bool(result.grounded),
            supported=bool(result.supported),
            source_page_numbers=[int(p) for p in (result.source_page_numbers or [])],
            retrieved_chunk_ids=[str(c) for c in (result.retrieved_chunk_ids or [])],
            urgent=False,
            # A side-effect report is offered the existing AE workflow instead
            # of being collected here; no case is created by this endpoint.
            suggest_ae_report=(
                safety.signal is SafetySignal.NORMAL_AE_REPORTING
            ),
        )

    @router.post("/mi/transcribe", response_model=MiTranscribeResponse)
    def mi_transcribe(
        audio_file: UploadFile = File(...),
        language: str = Form(DEFAULT_LANGUAGE_CODE),
    ) -> MiTranscribeResponse:
        raw = audio_file.file.read()
        if not raw:
            raise HTTPException(
                status_code=400, detail="Uploaded audio file is empty."
            )
        suffix = Path(audio_file.filename or "").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(raw)
            tmp_path = Path(handle.name)
        try:
            service = stt or PrismaSTT()
            transcription = service.transcribe(tmp_path, language_code=language)
        except VALIDATION_ERRORS as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CONFIGURATION_ERRORS as exc:
            raise HTTPException(
                status_code=500,
                detail=f"MedVaani services are not configured: {exc}",
            ) from exc
        except UPSTREAM_ERRORS as exc:
            raise _upstream_http_error(exc) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        text = (transcription.text or "").strip()
        if not text:
            raise HTTPException(
                status_code=400,
                detail="The recording produced no speech. Please try again.",
            )
        return MiTranscribeResponse(transcript=text)

    return router
