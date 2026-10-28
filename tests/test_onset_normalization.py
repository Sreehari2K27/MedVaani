"""Focused tests for onset-date normalization and worsening preservation.

Covers: relative wording resolved against a fixed conversation timestamp,
explicit dates, unknown/ambiguous timestamps, original-wording retention,
period-only time precision (no invented clock time), the readable
``DD-MMM-YYYY`` report format, worsening kept distinct from generic
``not_recovered``, and STT timestamp plumbing through the voice endpoint.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.main import FastAPI
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine
from app.workflow.onset import (
    format_iso_date,
    format_report_date,
    normalize_onset_statement,
    resolve_reference,
)
from app.workflow.service.conversational import ConversationalAEWorkflow

# Fixed conversation date used by every regression below: 9 October 2026.
FIXED_TIMESTAMP = "2026-10-09T08:30:00+05:30"
FIXED_REFERENCE = datetime.fromisoformat(FIXED_TIMESTAMP)


def _engine_ready_for_onset(case_id: str) -> AEStateEngine:
    engine = AEStateEngine.create_case(case_id)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "45")
    engine.update("consent_to_contact", True)
    engine.update("contact_phone", "9999990000")
    return engine


def _engine_ready_for_outcome(case_id: str) -> AEStateEngine:
    engine = _engine_ready_for_onset(case_id)
    engine.update("onset_date", "yesterday")
    engine.update("seriousness", "no")
    return engine


# ---- report formatting ------------------------------------------------------


def test_fixed_conversation_date_displays_as_dd_mmm_yyyy():
    assert format_report_date(date(2026, 10, 9)) == "09-Oct-2026"
    assert format_iso_date("2026-10-09") == "09-Oct-2026"
    assert format_report_date(date(2026, 1, 3)) == "03-Jan-2026"


# ---- spoken-out and relative onset wording ---------------------------------


@pytest.mark.parametrize(
    "statement,expected_iso,expected_display",
    [
        ("fourteenth of May twenty twenty six", "2026-05-14", "14-May-2026"),
        ("the 14th of May 2026", "2026-05-14", "14-May-2026"),
        ("14th May 2026", "2026-05-14", "14-May-2026"),
        ("twenty seventh May two thousand", "2000-05-27", "27-May-2000"),
        ("twenty seventh june nineteen ninety eight", "1998-06-27", "27-Jun-1998"),
        ("May 14 2026", "2026-05-14", "14-May-2026"),
        ("2026-05-14", "2026-05-14", "14-May-2026"),
        ("14/05/2026", "2026-05-14", "14-May-2026"),
    ],
)
def test_spoken_and_numeric_dates_normalize_to_dd_mmm_yyyy(
    statement, expected_iso, expected_display
):
    result = normalize_onset_statement(statement, FIXED_REFERENCE)

    assert result.normalized_iso == expected_iso
    assert result.display == expected_display
    assert result.source == "explicit"
    # The original wording is preserved verbatim for the audit trail.
    assert result.original == statement


def test_relative_dates_anchor_to_the_conversation_timestamp():
    # Fixed reference: 9 October 2026 (local offset frame).
    for statement, expected_iso in (
        ("yesterday", "2026-10-08"),
        ("today", "2026-10-09"),
        ("two days ago", "2026-10-07"),
        ("two days back", "2026-10-07"),
        ("three days ago", "2026-10-06"),
    ):
        result = normalize_onset_statement(statement, FIXED_REFERENCE)
        assert result.normalized_iso == expected_iso, statement
        assert result.source == "normalized", statement
        assert result.period is None  # no clock time is ever invented


def test_malayalam_relative_today_morning_resolves_with_a_reference():
    # The observed browser phrase: "ഇന്നു രാവിലെ" = this morning; both
    # spellings of today resolve against the local reference date, and the
    # period normalizes to the English "morning" value.
    for statement in ("ഇന്നു രാവിലെ", "ഇന്ന് രാവിലെ"):
        result = normalize_onset_statement(statement, FIXED_REFERENCE)
        assert result.normalized_iso == "2026-10-09", statement
        assert result.period == "morning", statement
        assert result.source == "normalized", statement


def test_malayalam_relative_without_a_reference_stays_unknown_and_verbatim():
    result = normalize_onset_statement("ഇന്നു രാവിലെ", None)
    assert result.normalized_iso is None
    assert result.source == "unknown"
    # The original wording is retained exactly as reported for review.
    assert result.original == "ഇന്നു രാവിലെ"


@pytest.mark.parametrize(
    "statement",
    [
        "sometime in May",
        "May 2026",
        "the fourteenth",
        "fourteenth of May",  # no year: never guessed
    ],
)
def test_partial_or_ambiguous_spoken_dates_stay_unresolved(statement):
    result = normalize_onset_statement(statement, FIXED_REFERENCE)

    assert result.normalized_iso is None
    assert result.source == "unknown"
    # The original wording is retained exactly as reported.
    assert result.original == statement


def test_workflow_report_shows_normalized_onset_and_unknown_time():
    engine = _engine_ready_for_onset("onset-spoken-report")
    engine.update("seriousness", "no")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("fourteenth of May twenty twenty six")

    collected = result.collected_information
    # Verbatim wording for audit; DD-MMM-YYYY for the report display.
    assert collected["onset_date"] == "fourteenth of May twenty twenty six"
    assert collected["onset_date_normalized"] == "2026-05-14"
    assert collected["onset_date_display"] == "14-May-2026"
    assert collected["onset_date_source"] == "explicit"
    # No time was supplied: the period stays unset (shown as unknown).
    assert collected["onset_period"] is None
    assert engine.to_dict()["onset_date_display"] == "14-May-2026"


# ---- reference resolution ---------------------------------------------------


def test_reference_uses_timestamp_local_offset():
    resolved = resolve_reference(FIXED_TIMESTAMP)
    assert resolved is not None
    assert resolved.date() == date(2026, 10, 9)


def test_naive_or_utc_timestamp_is_not_a_reliable_local_frame():
    assert resolve_reference("2026-10-09T08:30:00") is None
    assert resolve_reference("2026-10-08T20:00:00Z") is None
    assert resolve_reference(None) is None
    assert resolve_reference("not-a-timestamp") is None


def test_explicit_timezone_converts_utc_and_naive_instants():
    converted = resolve_reference("2026-10-08T20:00:00Z", "+05:30")
    assert converted is not None
    assert converted.date() == date(2026, 10, 9)

    naive = resolve_reference("2026-10-09T08:30:00", "+05:30")
    assert naive is not None
    assert naive.date() == date(2026, 10, 9)


# ---- relative wording -------------------------------------------------------


def test_today_morning_normalizes_without_inventing_a_time():
    result = normalize_onset_statement("today morning", FIXED_REFERENCE)
    assert result.normalized_iso == "2026-10-09"
    assert "T" not in result.normalized_iso  # date only, no invented clock time
    assert result.period == "morning"
    assert result.source == "normalized"
    assert result.original == "today morning"
    assert result.display == "09-Oct-2026"


def test_yesterday_evening_normalizes_to_previous_day():
    result = normalize_onset_statement("yesterday evening", FIXED_REFERENCE)
    assert result.normalized_iso == "2026-10-08"
    assert result.period == "evening"
    assert result.source == "normalized"


def test_two_days_ago_normalizes_with_unknown_time():
    result = normalize_onset_statement("It started two days ago", FIXED_REFERENCE)
    assert result.normalized_iso == "2026-10-07"
    assert result.period is None
    assert result.source == "normalized"


def test_two_days_back_normalizes_with_unknown_time():
    # "back" is a common spoken variant of "ago" and must resolve the same way.
    for statement in ("It started two days back", "It started 2 days back"):
        result = normalize_onset_statement(statement, FIXED_REFERENCE)
        assert result.normalized_iso == "2026-10-07", statement
        assert result.period is None, statement
        assert result.source == "normalized", statement


def test_two_days_back_without_reference_keeps_wording():
    result = normalize_onset_statement("It started two days back", None)
    assert result.original == "It started two days back"
    assert result.normalized_iso is None
    assert result.source == "unknown"


# ---- explicit dates ---------------------------------------------------------


def test_explicit_dates_are_preserved_without_a_reference():
    cases = {
        "15 October 2026": "2026-10-15",
        "October 15, 2026": "2026-10-15",
        "2026-09-30": "2026-09-30",
        "15/09/2026": "2026-09-15",
    }
    for statement, expected in cases.items():
        result = normalize_onset_statement(statement)
        assert result.normalized_iso == expected, statement
        assert result.source == "explicit", statement


def test_ambiguous_numeric_date_is_not_guessed():
    # Both components could be a month; the original wording is kept instead.
    result = normalize_onset_statement("09/10/2026")
    assert result.normalized_iso is None
    assert result.source == "unknown"
    assert result.original == "09/10/2026"


def test_missing_reference_retains_wording_and_marks_date_unknown():
    result = normalize_onset_statement("yesterday evening", None)
    assert result.original == "yesterday evening"
    assert result.normalized_iso is None
    assert result.source == "unknown"
    # The period is still reported even when the calendar date is unknown.
    assert result.period == "evening"


# ---- workflow integration ---------------------------------------------------


def test_workflow_normalizes_onset_from_conversation_timestamp():
    engine = _engine_ready_for_onset("onset-wf-normalized")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(
        "It started this morning",
        context={"conversation_timestamp": FIXED_TIMESTAMP},
    )

    collected = result.collected_information
    assert engine.case.onset_date == "It started this morning"
    assert engine.case.onset_date_normalized == "2026-10-09"
    assert engine.case.onset_period == "morning"
    assert engine.case.onset_date_source == "normalized"
    assert collected["onset_date_display"] == "09-Oct-2026"
    # The onset requirement is satisfied and the conversation advances without
    # an extra confirmation turn.
    assert "onset_date" not in result.missing_information
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"

    snapshot = engine.to_dict()
    assert snapshot["onset_date"] == "It started this morning"
    assert snapshot["onset_date_display"] == "09-Oct-2026"
    assert snapshot["onset_date_source"] == "normalized"


def test_workflow_without_timestamp_marks_date_unknown_and_advances():
    engine = _engine_ready_for_onset("onset-wf-unknown")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("It started this morning")

    assert engine.case.onset_date == "It started this morning"
    assert engine.case.onset_date_normalized is None
    assert engine.case.onset_period == "morning"
    assert engine.case.onset_date_source == "unknown"
    assert result.collected_information["onset_date_display"] is None
    assert "onset_date" not in result.missing_information
    assert result.next_question[0] == "seriousness"


def test_workflow_normalizes_two_days_back_from_conversation_timestamp():
    engine = _engine_ready_for_onset("onset-wf-days-back")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(
        "It started two days back",
        context={"conversation_timestamp": FIXED_TIMESTAMP},
    )

    assert engine.case.onset_date == "It started two days back"
    assert engine.case.onset_date_normalized == "2026-10-07"
    assert engine.case.onset_period is None
    assert engine.case.onset_date_source == "normalized"
    assert result.collected_information["onset_date_display"] == "07-Oct-2026"
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


# ---- worsening preserved ----------------------------------------------------


def test_worsening_is_preserved_as_its_own_outcome():
    engine = _engine_ready_for_outcome("onset-wf-worsening")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("The symptoms are getting worse")

    assert engine.case.outcome == "worsening"
    assert result.collected_information["outcome"] == "worsening"
    assert engine.to_dict()["outcome"] == "worsening"


def test_negated_worsening_is_not_marked_as_worsening():
    engine = _engine_ready_for_outcome("onset-wf-not-worse")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("I am not getting worse")

    assert engine.case.outcome == "not_recovered"


def test_still_ongoing_remains_not_recovered():
    engine = _engine_ready_for_outcome("onset-wf-ongoing")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("The headache is still ongoing")

    assert engine.case.outcome == "not_recovered"


# ---- endpoint plumbing ------------------------------------------------------


class TimestampedSTT:
    """Offline STT stand-in that carries the provider timestamp per turn."""

    def __init__(self, transcripts: list[str], timestamp: str | None) -> None:
        self._transcripts = list(transcripts)
        self._timestamp = timestamp

    def transcribe(self, audio_path, language_code="en-IN"):
        if not self._transcripts:
            raise AssertionError("Unexpected extra turn: transcript queue empty")
        return SimpleNamespace(text=self._transcripts.pop(0), timestamp=self._timestamp)


class FakeTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _build_client(transcripts: list[str], timestamp: str | None) -> TestClient:
    router = create_voice_conversation_router(
        stt=TimestampedSTT(transcripts, timestamp),
        tts=FakeTTS(),
        session_store=SessionStore(),
        llm_provider=AnswerLLMProvider(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _turn(client: TestClient, session_id: str) -> dict:
    resp = client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": "en-IN"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _script_to_onset() -> list[str]:
    # Approved order: report -> Step 4 verification -> reporter identity ->
    # patient name and age (combined read-back) -> contact number (read-back)
    # -> consent -> optional email declined -> severity -> incident date.
    return [
        "I took Lantus yesterday and today I have severe headache and vomiting. "
        "I want to report this adverse event.",
        "yes",  # Step 4: medication verified
        "I am the patient",
        "Alpha Jones",
        "45 years old",  # age completes the details -> combined read-back
        "yes",  # Step 6B.2: patient details confirmed
        "9999990000",  # patient contact number -> read-back
        "yes",  # contact number confirmed
        "Yes, I consent",  # Step 7: consent to contact
        "no",  # Step 8: the optional email address is declined
        "no",  # Step 9: not severe -> the incident date is asked next
        "It started this morning",
    ]


def test_voice_endpoint_threads_stt_timestamp_into_onset_display():
    client = _build_client(_script_to_onset(), FIXED_TIMESTAMP)
    session_id = "onset-endpoint"
    for _ in range(11):
        _turn(client, session_id)
    onset_turn = _turn(client, session_id)

    # Pre-existing envelope keys are unchanged (additive collected fields only).
    assert onset_turn["workflow"] == "ae"
    for key in (
        "session_id", "transcript", "intent", "workflow", "reason", "response_text",
        "audio_path", "current_state", "case_id", "priority", "human_review_required",
        "completed", "missing_information", "next_question", "safety_status", "mi", "ae",
    ):
        assert key in onset_turn, key
    assert set(onset_turn["ae"]) == {"extracted_information", "collected_information"}

    collected = onset_turn["ae"]["collected_information"]
    assert collected["onset_date"] == "It started this morning"
    assert collected["onset_date_display"] == "09-Oct-2026"
    assert collected["onset_date_source"] == "normalized"
    assert "I heard" not in onset_turn["response_text"]


def test_voice_endpoint_without_valid_frame_uses_the_aware_receipt_time():
    # With no STT timestamp (or a naive one), the voice endpoint follows the
    # text conversation's established receipt-time rule: the server's aware
    # receipt time is the conversation frame; a naive instant is never
    # assigned a guessed timezone nor treated as UTC. The reporter's wording
    # is still kept byte for byte.
    from datetime import datetime as datetime_mod

    client = _build_client(_script_to_onset(), None)
    session_id = "onset-endpoint-no-ts"
    for _ in range(11):
        _turn(client, session_id)
    onset_turn = _turn(client, session_id)

    collected = onset_turn["ae"]["collected_information"]
    assert collected["onset_date"] == "It started this morning"
    expected_receipt_date = format_iso_date(
        datetime_mod.now().astimezone().date().isoformat()
    )
    assert collected["onset_date_display"] == expected_receipt_date
    assert collected["onset_date_source"] == "normalized"
    assert collected["onset_period"] == "morning"


def test_voice_endpoint_gnani_timestamp_prefers_aware_receipt_frame(monkeypatch):
    # Gnani's underscored naive timestamp (fixture format, tests/test_prisma.py:74)
    # is not a reliable local frame: the endpoint falls back to the
    # controlled aware receipt time below and never treats the naive
    # instant as UTC or assigns a guessed timezone to it.
    import app.api.voice_conversation as vc

    fixed = datetime.fromisoformat(
        "2026-10-23T09:00:00+05:30"
    )

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(vc, "datetime", FixedClock)

    client = _build_client(_script_to_onset(), "20251226_143052.123")
    session_id = "onset-endpoint-gnani-ts"
    for _ in range(11):
        _turn(client, session_id)
    onset_turn = _turn(client, session_id)

    collected = onset_turn["ae"]["collected_information"]
    assert collected["onset_date"] == "It started this morning"
    assert collected["onset_date_display"] == format_iso_date("2026-10-23")
    assert collected["onset_period"] == "morning"
    assert collected["onset_date_source"] == "normalized"


def test_hindi_today_morning_resolves_from_the_reference():
    # "आज सुबह" ("today morning", Hindi words spoken in a Marathi session):
    # the date resolves from the conversation reference and the morning
    # period is recorded - no clock time is ever invented, and the original
    # wording is preserved verbatim.
    import datetime

    result = normalize_onset_statement(
        "आज सुबह",
        reference=datetime.datetime(2026, 10, 27, 9, 0),
    )
    assert result.normalized_iso == "2026-10-27"
    assert result.period == "morning"
    assert result.source == "normalized"
    assert result.original == "आज सुबह"


def test_hindi_today_morning_without_a_reference_stays_unresolved():
    result = normalize_onset_statement("आज सुबह")
    assert result.normalized_iso is None
    assert result.period == "morning"
    assert result.source == "unknown"


def test_acknowledgement_is_never_stored_as_onset():
    # The reported failure: the acknowledgement "अच्छा" answered the onset
    # question and was stored as the onset date. A non-date answer must not
    # be stored: the onset stays unknown and the bounded clarification
    # re-asks; a valid relative expression still stores and resolves.
    import datetime

    engine = AEStateEngine.create_case("AE-onset-ack")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju P K")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    from app.workflow.service.conversational import (
        VOICE_CONFIRMATION_FIELDS,
        ConversationalAEWorkflow,
    )

    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    asked = workflow.process_turn("no")  # severity -> onset pending
    assert asked.next_question is not None
    assert asked.next_question[0] == "onset_date"

    acked = workflow.process_turn("अच्छा")
    assert engine.case.onset_date is None  # never stored
    assert acked.next_question is not None
    assert acked.next_question[0] == "onset_date"  # re-asked (bounded loop)

    from app.workflow.onset import is_recognizable_onset

    assert is_recognizable_onset("अच्छा") is False
    assert is_recognizable_onset("yesterday") is True
    assert is_recognizable_onset(
        "आज सुबह"
    ) is True
    assert is_recognizable_onset("27 May 2000") is True
