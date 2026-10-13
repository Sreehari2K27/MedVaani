"""Tests for the optional closing "anything else?" intake step.

Covers the workflow behaviour (asked once near the end of intake, free-form
answer retained verbatim, extraction limited to empty fields, explicit decline,
no re-asking) and the voice endpoint wiring.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.providers.base import LLMResult
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    ADDITIONAL_INFORMATION_QUESTION,
    ConversationalAEWorkflow,
)

REQUIRED_FIELDS = (
    ("product", "Lantus (fictional)"),
    ("adverse_event", "Headache (demo)"),
    ("patient_identifier", "PT-001"),
    ("patient_age", "45"),
    ("reporter_role", "self"),
    ("contact_phone", "9999990000"),
    ("consent_to_contact", True),
    ("onset_date", "yesterday"),
    ("seriousness", "no"),
    ("outcome", "recovering"),
)


def _filled_engine(case_id: str = "AE-ai-demo") -> AEStateEngine:
    engine = AEStateEngine.create_case(case_id)
    for field, value in REQUIRED_FIELDS:
        engine.update(field, value)
    return engine


class _NeutralProvider:
    """Deterministic provider that extracts nothing."""

    def interpret(self, message, context=None):
        return LLMResult(intent="unknown", extracted_fields={})


class _ScriptedProvider:
    """Returns fixed extracted fields per user message (deterministic)."""

    def __init__(self, fields_by_message):
        self._fields = fields_by_message

    def interpret(self, message, context=None):
        return LLMResult(
            intent="ae_answer",
            extracted_fields=dict(self._fields.get(message, {})),
        )


def test_closing_question_is_asked_once_and_blocks_completion():
    engine = _filled_engine()
    workflow = ConversationalAEWorkflow(
        engine, _NeutralProvider(), ask_additional_information=True
    )

    result = workflow.process_turn("Thank you.")

    assert ADDITIONAL_INFORMATION_QUESTION in result.response_text
    assert result.completed is False
    assert engine.current_state is not WorkflowState.CASE_COMPLETED
    assert result.next_question is None


def test_free_form_answer_is_retained_and_completes():
    engine = _filled_engine()
    workflow = ConversationalAEWorkflow(
        engine, _NeutralProvider(), ask_additional_information=True
    )
    workflow.process_turn("Thank you.")

    answer = "I also felt dizzy in the evenings."
    result = workflow.process_turn(answer)

    assert result.completed is True
    assert engine.current_state is WorkflowState.CASE_COMPLETED
    assert engine.case.additional_information == answer
    assert engine.additional_information_status == "provided"
    collected = result.collected_information
    assert collected["additional_information"] == answer
    assert collected["additional_information_status"] == "provided"
    # The free text is kept separate from the structured fields.
    assert engine.case.adverse_event == "Headache (demo)"


def test_decline_marks_none_provided_and_does_not_re_ask():
    engine = _filled_engine()
    workflow = ConversationalAEWorkflow(
        engine, _NeutralProvider(), ask_additional_information=True
    )
    workflow.process_turn("Thank you.")

    result = workflow.process_turn("No, nothing else.")

    assert result.completed is True
    assert engine.case.additional_information is None
    assert engine.additional_information_status == "declined"
    assert result.collected_information["additional_information_status"] == "declined"
    assert ADDITIONAL_INFORMATION_QUESTION not in result.response_text

    # A later turn must not re-open the closing question.
    later = workflow.process_turn("Actually nothing.")
    assert ADDITIONAL_INFORMATION_QUESTION not in later.response_text


def test_extraction_from_closing_answer_only_fills_empty_fields():
    engine = _filled_engine()
    assert engine.case.medical_care is None
    answer = "I also needed to see a doctor about it."
    provider = _ScriptedProvider(
        {answer: {"outcome": "recovered", "medical_care": "yes"}}
    )
    workflow = ConversationalAEWorkflow(
        engine, provider, ask_additional_information=True
    )
    workflow.process_turn("Thank you.")

    result = workflow.process_turn(answer)

    assert engine.case.additional_information == answer
    # Already-captured outcome is never overwritten by the closing answer.
    assert engine.case.outcome == "recovering"
    # An empty structured field may be filled from it.
    assert engine.case.medical_care == "yes"
    assert result.completed is True


def test_closing_step_is_off_by_default():
    engine = _filled_engine()
    workflow = ConversationalAEWorkflow(engine, _NeutralProvider())

    result = workflow.process_turn("Thank you.")

    assert result.completed is True
    assert ADDITIONAL_INFORMATION_QUESTION not in result.response_text


def test_report_includes_additional_information_subsection_fields():
    engine = _filled_engine()
    engine.record_additional_information("There was also a mild itch.")
    snapshot = engine.to_dict()
    assert snapshot["additional_information"] == "There was also a mild itch."
    assert snapshot["additional_information_status"] == "provided"

    engine.decline_additional_information()
    snapshot = engine.to_dict()
    assert snapshot["additional_information"] is None
    assert snapshot["additional_information_status"] == "declined"


class _ScriptSTT:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    def transcribe(self, audio_path, language_code="en-IN"):
        return SimpleNamespace(text=self._transcripts.pop(0))


class _RecordingTTS:
    def __init__(self):
        self.calls = []

    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        self.calls.append({"text": text, "language": language, "voice": voice})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _build_endpoint(stt, tts, store, ask):
    router = create_voice_conversation_router(
        stt=stt,
        tts=tts,
        session_store=store,
        llm_provider=_NeutralProvider(),
        ask_additional_information=ask,
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, session_id):
    return client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": "en-IN"},
    )


def test_voice_endpoint_asks_then_records_additional_information():
    store = SessionStore()
    stt = _ScriptSTT(["Thank you.", "I also felt dizzy in the evenings."])
    client = _build_endpoint(stt, _RecordingTTS(), store, ask=True)

    session = store.get_or_create("ai-endpoint")
    for field, value in REQUIRED_FIELDS:
        session.ae_engine.update(field, value)

    asked = _post(client, "ai-endpoint")
    assert asked.status_code == 200
    body = asked.json()
    assert ADDITIONAL_INFORMATION_QUESTION in body["response_text"]
    assert body["completed"] is False

    answered = _post(client, "ai-endpoint")
    assert answered.status_code == 200
    final = answered.json()
    assert final["completed"] is True
    collected = final["ae"]["collected_information"]
    assert collected["additional_information"] == "I also felt dizzy in the evenings."
    assert collected["additional_information_status"] == "provided"
