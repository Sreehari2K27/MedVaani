"""Conversational voice endpoint.

Routes each transcript to either the AE workflow or the MI/RAG workflow via
``route_transcript``, returns a unified JSON envelope, and maps Prisma/Timbre
errors to HTTP status codes (mirroring ``app.main``'s round-trip endpoint).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.config import get_gnani_api_key
from app.providers import (
    AnswerLLMProvider,
    FallbackLLMProvider,
    MockLLMProvider,
)
from app.language import (
    DEFAULT_LANGUAGE,
    detect_language,
    normalize_language,
    translate,
    voice_for,
)
from app.session import SessionStore
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)
from app.mi.guardrails import MIGHybridService
from app.routing import route_transcript
from app.services.prisma import PrismaSTT
from app.services.timbre import TimbreTTS
from app.api.errors import (
    CONFIGURATION_ERRORS,
    UPSTREAM_ERRORS,
    VALIDATION_ERRORS,
)

router = APIRouter()
session_store = SessionStore()

MI_INTENT = "product_info"

# Operator-side diagnostics only: never logs the API key, and any accidental
# occurrence of the configured key in an upstream message is masked.
logger = logging.getLogger(__name__)


def _safe_detail(text: str) -> str:
    key = get_gnani_api_key()
    if key:
        text = text.replace(key, "***")
    return text


def get_llm_provider():
    # Deterministic answer-aware provider wrapped in the fallback composite:
    # a future primary LLM can be inserted without changing turn handling.
    # No primary is configured today, so behavior is unchanged.
    try:
        fallback = AnswerLLMProvider()
    except Exception:
        fallback = MockLLMProvider()
    return FallbackLLMProvider(primary=None, fallback=fallback)


_default_mi_service = None


def get_mi_service():
    # Default service is cached: the retriever loads the chunk index and fits
    # TF-IDF once, not on every request.
    global _default_mi_service
    if _default_mi_service is None:
        _default_mi_service = MIGHybridService()
    return _default_mi_service


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:48]
    return sanitized or "default"


def _upstream_to_http_error(exc: Exception) -> HTTPException:
    """Map one Prisma/Timbre failure to the clearest HTTP error.

    Distinct upstream causes must not all look like a generic outage:
    authentication (403) is a configuration problem, rate limiting (429)
    needs a pause, and everything else (timeouts, malformed bodies, empty
    transcripts, 5xx) stays a 502 that preserves the provider's message.
    No secrets (API keys) are included in any message.
    """
    status = getattr(exc, "status_code", None)
    hint = getattr(exc, "hint", None)
    if status == 403:
        logger.warning(
            "Gnani STT rejected the request (HTTP 403): %s",
            _safe_detail(hint or "check the API key and the account's credits"),
        )
        return HTTPException(
            status_code=500,
            detail=hint
            or (
                "Speech service authentication failed. The operator must check "
                "the API key and the account's available credits."
            ),
        )
    if status == 429:
        return HTTPException(
            status_code=429,
            detail=hint or "Speech service rate limit exceeded. Wait a moment and retry.",
        )
    return HTTPException(status_code=502, detail=f"Gnani service failure: {exc}")


def _audio_path(workflow: str, session_id: str) -> Path:
    return Path("output") / f"conv_{workflow}_{_sanitize_filename_part(session_id)}.wav"


def _mi_payload(res) -> dict:
    return {
        "status": res.status,
        "category": res.category,
        "grounded": res.grounded,
        "supported": res.supported,
        "source_page_numbers": list(res.source_page_numbers),
        "retrieved_chunk_ids": list(res.retrieved_chunk_ids),
    }


def _language_metadata(
    conversation_language: str,
    spoken_text: str | None,
    voice: str | None,
    translated: bool,
    source: str,
) -> dict:
    # Additive, backward-compatible fields: English ``response_text`` is
    # unchanged for the UI; these describe the spoken form. ``None`` values
    # (text turns) mean no audio was produced for this turn.
    return {
        "conversation_language": conversation_language,
        "spoken_response_text": spoken_text,
        "voice": voice,
        "language_translated": translated,
        "language_source": source,
    }


def _synthesize_or_raise(synthesize, *args, **kwargs):
    """Run one TTS call, mapping service errors to the endpoint HTTP codes."""
    try:
        return synthesize(*args, **kwargs)
    except VALIDATION_ERRORS as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CONFIGURATION_ERRORS as exc:
        raise HTTPException(
            status_code=500, detail=f"MedVaani services are not configured: {exc}"
        ) from exc
    except UPSTREAM_ERRORS as exc:
        logger.warning(
            "Gnani TTS request failed (%s): %s",
            type(exc).__name__,
            _safe_detail(str(exc)),
        )
        raise HTTPException(
            status_code=502, detail=f"Gnani service failure: {exc}"
        ) from exc


def _empty_transcript_response(session, session_id: str, language: str, synthesize=None):
    """Retry envelope for an empty STT transcript: missing input, not an answer.

    Nothing is applied to the case and the workflow state is untouched; the
    pending question (e.g. the email address) is simply re-asked. Spoken
    feedback is best-effort: a TTS failure never turns the retry into an
    error response.
    """
    engine = session.ae_engine
    # The workflow's active conversational step (e.g. the pending email
    # question) takes precedence over the engine's plain missing fields.
    workflow = session.context.get("ae_workflow")
    pending = workflow._active_step() if workflow is not None else None
    if pending is None:
        pending = engine.next_field()
    if pending is not None and pending[0] == "contact_email":
        retry_text = "I couldn't catch the email address. Could you please repeat it?"
    else:
        retry_text = "Sorry, I couldn't catch that. Could you please repeat it?"

    conversation_language = (
        normalize_language(session.context.get("conversation_language"))
        or normalize_language(language)
        or DEFAULT_LANGUAGE
    )
    spoken = translate(retry_text, conversation_language)
    voice = voice_for(spoken.language) if spoken.translated else None
    audio_path = None
    if synthesize is not None:
        try:
            audio_path = _synthesize_or_raise(
                synthesize, spoken.text, "ae", session_id, spoken.language, voice=voice
            )
        except HTTPException:
            audio_path = None  # best-effort spoken feedback only

    return JSONResponse(
        {
            "session_id": session_id,
            "transcript": "",
            "intent": "unknown",
            "workflow": "ae",
            "reason": "empty_transcript",
            "response_text": retry_text,
            "audio_path": str(audio_path) if audio_path is not None else None,
            "current_state": engine.current_state.value,
            "case_id": engine.case.case_id,
            "priority": engine.case.priority,
            "human_review_required": engine.current_state.value == "HUMAN_REVIEW",
            "completed": engine.current_state.value == "CASE_COMPLETED",
            "missing_information": list(engine.missing_information()),
            "next_question": list(pending) if pending else None,
            "safety_status": "ok",
            "mi": None,
            "ae": {
                "extracted_information": {},
                "collected_information": session.context.get("ae_workflow")
                ._collected_snapshot()
                if session.context.get("ae_workflow")
                else {},
            },
            **_language_metadata(
                conversation_language,
                spoken.text if audio_path is not None else None,
                voice,
                spoken.translated,
                "session" if session.context.get("conversation_language") else "default",
            ),
        }
    )


def process_transcript_turn(
    session,
    session_id: str,
    transcript: str,
    *,
    conversation_timestamp,
    provider_language: str | None,
    request_language: str,
    hint_language: str,
    get_llm,
    get_mi,
    synthesize=None,
    ask_additional_information: bool = False,
) -> dict:
    """Shared post-transcription turn handling for the voice and text endpoints.

    Both endpoints feed one text transcript through the same language
    mirroring, transcript routing and MI/AE workflow handling, so AE Text and
    AE Voice follow the same authoritative workflow rules. ``synthesize`` is
    the conversation's TTS callable for voice turns and ``None`` for text
    turns (no STT/TTS); the returned envelope is identical apart from
    ``audio_path`` and the spoken-language metadata.
    """
    engine = session.ae_engine

    # Identify the spoken language: reliable provider metadata when the body
    # carries it, otherwise an offline transcript heuristic. The session
    # language is updated when a different language is clearly detected so the
    # patient can switch mid-conversation.
    detected = detect_language(transcript, provider_language)
    current_language = normalize_language(
        session.context.get("conversation_language")
    )
    if detected is not None and detected != current_language:
        session.context["conversation_language"] = detected
        current_language = detected
    conversation_language = current_language or hint_language
    if provider_language and detected is not None:
        language_source = "provider"
    elif detected is not None:
        language_source = "transcript"
    elif current_language is not None:
        language_source = "session"
    else:
        language_source = "default"

    decision = route_transcript(transcript)
    session.context["last_workflow"] = decision.workflow
    session.context["last_intent"] = decision.intent
    session.context["last_reason"] = decision.reason

    if decision.workflow == "mi":
        mi = get_mi()
        mi_res = mi.handle(transcript)
        if mi_res.allowed:
            intent = MI_INTENT
        else:
            intent = "treatment_query"

        audio_path = None
        if synthesize is not None:
            audio_path = _synthesize_or_raise(
                synthesize, mi_res.answer, "mi", session_id, request_language
            )

        return {
            "session_id": session_id,
            "transcript": transcript,
            "intent": intent,
            "workflow": "mi",
            "reason": decision.reason,
            "response_text": mi_res.answer,
            "audio_path": str(audio_path) if audio_path is not None else None,
            "current_state": engine.current_state.value,
            "case_id": engine.case.case_id,
            "priority": engine.case.priority,
            "human_review_required": engine.current_state.value == "HUMAN_REVIEW",
            "completed": engine.current_state.value == "CASE_COMPLETED",
            "missing_information": None,
            "next_question": None,
            "safety_status": "ok",
            "mi": _mi_payload(mi_res),
            "ae": None,
            # MI/RAG content is intentionally excluded from
            # translation (see task constraints); the language is
            # still tracked so the AE conversation keeps mirroring.
            **_language_metadata(
                conversation_language,
                mi_res.answer if synthesize is not None else None,
                None,
                False,
                language_source,
            ),
        }

    # AE workflow. Cache one orchestrator per session so cross-turn state
    # (e.g. the one-time physician-consultation warning) persists across
    # separate HTTP requests instead of resetting every turn. Identity and
    # contact values are read back for explicit confirmation; session state
    # (including a pending confirmation) lives on session.ae_engine.
    wf = session.context.get("ae_workflow")
    if wf is None:
        wf = ConversationalAEWorkflow(
            engine,
            get_llm(),
            confirm_fields=VOICE_CONFIRMATION_FIELDS,
            ask_additional_information=ask_additional_information,
        )
        session.context["ae_workflow"] = wf
    res = wf.process_turn(
        transcript, context={"conversation_timestamp": conversation_timestamp}
    )

    # The reply's translated form is produced for BOTH voice and text turns so
    # the envelopes stay consistent: the English ``response_text`` below is
    # unchanged for the on-screen UI; the translated form is all-or-nothing
    # (English fallback when a language has no catalog) so we never mix
    # languages. Only voice turns synthesize audio; text turns never call TTS.
    spoken = translate(res.response_text, conversation_language)
    audio_path = None
    spoken_text = spoken.text
    translated = spoken.translated
    voice = None
    if synthesize is not None:
        voice = voice_for(spoken.language) if spoken.translated else None
        audio_path = _synthesize_or_raise(
            synthesize, spoken.text, "ae", session_id, spoken.language, voice=voice
        )

    return {
        "session_id": session_id,
        "transcript": transcript,
        "intent": res.intent,
        "workflow": "ae",
        "reason": decision.reason,
        "response_text": res.response_text,
        "audio_path": str(audio_path) if audio_path is not None else None,
        "current_state": res.current_state,
        "case_id": res.case_id,
        "priority": res.priority,
        "human_review_required": res.human_review_required,
        "completed": res.completed,
        "missing_information": list(res.missing_information),
        "next_question": list(res.next_question) if res.next_question else None,
        "safety_status": res.safety_status,
        "mi": None,
        "ae": {
            "extracted_information": res.extracted_information,
            "collected_information": res.collected_information,
        },
        **_language_metadata(
            conversation_language,
            spoken_text,
            voice,
            translated,
            language_source,
        ),
    }


def create_voice_conversation_router(
    stt: PrismaSTT | None = None,
    tts: TimbreTTS | None = None,
    session_store: SessionStore | None = None,
    llm_provider = None,
    mi_service = None,
    ask_additional_information: bool = False,
):
    r = APIRouter()
    store = session_store or SessionStore()

    def _get_llm():
        if llm_provider is not None:
            return llm_provider
        return AnswerLLMProvider()

    def _get_mi():
        if mi_service is not None:
            return mi_service
        return get_mi_service()

    def _transcribe(temp_path: Path, language: str):
        service = stt or PrismaSTT()
        return service.transcribe(temp_path, language_code=language)

    def _synthesize(
        text: str,
        workflow: str,
        session_id: str,
        language: str,
        voice: str | None = None,
    ):
        service = tts or TimbreTTS()
        kwargs = {"language": language}
        if voice is not None:
            kwargs["voice"] = voice
        return service.synthesize_to_file(
            text, _audio_path(workflow, session_id), **kwargs
        )

    @r.post("/voice/conversation")
    async def voice_conversation(
        audio_file: UploadFile = File(...),
        session_id: str = Form(...),
        language: str = Form("en-IN"),
    ):
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")

        # Save temp - preserve original suffix
        suffix = Path(audio_file.filename or "").suffix or ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            content = await audio_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty audio file")
            temp_file.write(content)
            temp_file.close()

            session = store.get_or_create(session_id)

            # Mirror the patient's spoken language. The form ``language`` is an
            # initial hint only; once a language has been established for the
            # session it drives STT so the transcript is in the right language.
            hint = normalize_language(language) or DEFAULT_LANGUAGE
            stt_language = (
                normalize_language(session.context.get("conversation_language"))
                or hint
            )

            try:
                transcription = _transcribe(Path(temp_file.name), stt_language)
            except VALIDATION_ERRORS as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except CONFIGURATION_ERRORS as exc:
                raise HTTPException(
                    status_code=500, detail=f"MedVaani services are not configured: {exc}"
                ) from exc
            except UPSTREAM_ERRORS as exc:
                if getattr(exc, "status_code", None) == 200 and (
                    "empty transcript" in str(exc).lower()
                ):
                    # An empty transcript is missing input, never an answer:
                    # nothing is applied and the pending question is re-asked.
                    return _empty_transcript_response(
                        session, session_id, language, synthesize=_synthesize
                    )
                raise _upstream_to_http_error(exc) from exc

            # Shared turn handling (routing, safety envelope, workflows) also
            # used by POST /text/conversation below.
            return JSONResponse(
                process_transcript_turn(
                    session,
                    session_id,
                    transcription.text,
                    # The STT provider's own timestamp is the conversation
                    # reference for resolving relative onset wording
                    # ("today morning", "yesterday").
                    conversation_timestamp=getattr(
                        transcription, "timestamp", None
                    ),
                    provider_language=getattr(
                        transcription, "detected_language", None
                    ),
                    request_language=language,
                    hint_language=hint,
                    get_llm=_get_llm,
                    get_mi=_get_mi,
                    synthesize=_synthesize,
                    ask_additional_information=ask_additional_information,
                )
            )
        finally:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    @r.post("/text/conversation")
    async def text_conversation(
        session_id: str = Form(...),
        message: str = Form(...),
        language: str = Form("en-IN"),
    ):
        """Typed conversational turn sharing the voice endpoint's workflow.

        Bypasses STT/TTS entirely: the message text is fed through the same
        transcript routing, ConversationalAEWorkflow state, safety envelope,
        consent behaviour and additional-information handling as a voice
        transcript. Returns the same response envelope with ``audio_path`` null.
        """
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="message is required")

        session = store.get_or_create(session_id)
        hint = normalize_language(language) or DEFAULT_LANGUAGE
        return JSONResponse(
            process_transcript_turn(
                session,
                session_id,
                message.strip(),
                # No STT means no provider timestamp; onset wording then keeps
                # its original phrasing (never resolved against a server clock).
                conversation_timestamp=None,
                provider_language=None,
                request_language=language,
                hint_language=hint,
                get_llm=_get_llm,
                get_mi=_get_mi,
                synthesize=None,
                ask_additional_information=ask_additional_information,
            )
        )

    return r


# Keep default router for backward compat
router = create_voice_conversation_router()