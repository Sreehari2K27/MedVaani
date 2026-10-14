"""Regression tests for the explicit patient-name confirmation sub-state.

These tests exercise only the production ``AEStateEngine``: capture a
candidate name, read it back, accept yes/correct, allow a correction after
no, and block advancement until the name is confirmed.
"""
from __future__ import annotations

import pytest

from app.providers.mock import MockLLMProvider
from app.workflow.ae_engine import (
    AECaseError,
    AECaseValidationError,
    AEStateEngine,
    WorkflowState,
)
from app.workflow.service.conversational import (
    NAME_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
    _looks_like_value,
)


def _engine_with_reporter() -> AEStateEngine:
    """Engine with everything captured except the patient's name."""
    engine = AEStateEngine.create_case("AE-confirm-001")
    engine.update("product", "Demo Drug")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_age", "45")
    return engine


def test_name_capture_is_read_back_and_blocks_advancement():
    engine = _engine_with_reporter()

    question = engine.begin_confirmation("patient_identifier", "Demo Patient 001")

    # Approved Step 6B.2: the name and the already-captured date/age are read
    # back together.
    assert question == (
        "I have your details as Demo Patient 001, 45. Is that correct?"
    )
    assert engine.next_field() == ("confirmation", question)
    # Candidate is captured but not committed until confirmed.
    assert engine.case.patient_identifier is None
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "Demo Patient 001"
    }
    assert engine.case.pending_confirmation == "patient_identifier"
    assert "patient_identifier" in engine.missing_information()
    assert engine.can_complete() is False
    assert engine.current_state is not WorkflowState.PATIENT_IDENTIFIED
    assert engine.to_dict()["pending_confirmation"] == "patient_identifier"


def test_yes_confirms_name_commits_it_and_allows_advancement():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "Demo Patient 001")

    next_question = engine.respond_to_confirmation("yes")

    assert engine.case.patient_identifier == "Demo Patient 001"
    assert engine.case.pending_confirmation is None
    assert engine.case.confirmation_candidate == {}
    assert "patient_identifier" in engine.case.confirmed_fields
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED
    assert "patient_identifier" not in engine.missing_information()
    assert engine.next_field() is not None
    assert engine.next_field()[0] == "contact_phone"
    assert next_question is not None


def test_correct_and_punctuated_affirmatives_are_accepted():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "Demo Patient 001")
    engine.respond_to_confirmation("Correct.")
    assert engine.case.patient_identifier == "Demo Patient 001"
    assert engine.case.pending_confirmation is None

    second = _engine_with_reporter()
    second.begin_confirmation("patient_identifier", "Demo Patient 001")
    second.respond_to_confirmation("Yes!")
    assert second.case.patient_identifier == "Demo Patient 001"
    assert second.case.pending_confirmation is None


def test_no_keeps_name_unconfirmed_and_reasks_for_correction():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "John Smith")

    reasked = engine.respond_to_confirmation("no")

    # The rejection re-asks for the name and the date of birth (combined
    # Step 6B.1 question) instead of advancing.
    assert reasked == (
        "Could you please tell me your first and last name, along with your "
        "date of birth?"
    )
    assert engine.next_field() == ("patient_identifier", reasked)
    assert engine.case.patient_identifier is None
    assert engine.case.confirmation_candidate == {}
    assert engine.case.pending_confirmation == "patient_identifier"
    assert engine.case.confirmation_response == "no"
    assert "patient_identifier" in engine.missing_information()
    assert engine.can_complete() is False
    assert engine.current_state is not WorkflowState.PATIENT_IDENTIFIED


def test_correction_after_no_is_read_back_then_confirmed():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "John Smith")
    engine.respond_to_confirmation("no")

    # The reporter supplies the corrected name through the normal update path.
    engine.update("patient_identifier", "Jane Doe")
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "Jane Doe"
    }
    assert engine.case.confirmation_response is None
    assert engine.case.patient_identifier is None
    assert engine.next_field() == (
        "confirmation",
        "I have your details as Jane Doe, 45. Is that correct?",
    )

    engine.respond_to_confirmation("yes")
    assert engine.case.patient_identifier == "Jane Doe"
    assert engine.case.pending_confirmation is None
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED


def test_unrecognised_reply_is_treated_as_a_correction():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "John Smith")

    question = engine.respond_to_confirmation("Jane Doe")

    assert question == "I have your details as Jane Doe, 45. Is that correct?"
    assert engine.case.pending_confirmation == "patient_identifier"
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "Jane Doe"
    }
    assert engine.case.patient_identifier is None


def test_update_cannot_bypass_a_pending_confirmation():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "John Smith")

    with pytest.raises(AECaseValidationError):
        engine.update("patient_identifier", "Sneaky Name")

    assert engine.case.patient_identifier is None
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "John Smith"
    }
    assert engine.next_field()[0] == "confirmation"


def test_confirmation_replies_require_a_pending_confirmation():
    engine = _engine_with_reporter()

    with pytest.raises(AECaseValidationError):
        engine.respond_to_confirmation("yes")

    engine.begin_confirmation("patient_identifier", "John Smith")
    with pytest.raises(AECaseValidationError):
        engine.begin_confirmation("contact_phone", "9999990000")
    with pytest.raises(AECaseValidationError):
        engine.begin_confirmation("patient_identifier", "   ")


def test_plain_name_update_without_confirmation_still_commits():
    # No confirmation requested: legacy behaviour is unchanged.
    engine = _engine_with_reporter()
    engine.update("patient_identifier", "Demo Patient 001")

    assert engine.case.patient_identifier == "Demo Patient 001"
    assert engine.case.pending_confirmation is None
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED
    assert engine.next_field()[0] == "contact_phone"


def test_pending_confirmation_blocks_completion_error():
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", "John Smith")

    assert engine.can_complete() is False
    with pytest.raises(AECaseError):
        engine.complete()
    assert engine.current_state is not WorkflowState.CASE_COMPLETED


# ---- Devanagari (Indic-script) name corrections via the workflow ----------


def _workflow_with_stale_candidate(stale: str = "कर्मा तेल"):
    """Workflow whose pending name read-back carries a misheard candidate."""
    engine = _engine_with_reporter()
    engine.begin_confirmation("patient_identifier", stale)
    wf = ConversationalAEWorkflow(
        engine,
        MockLLMProvider(),
        confirm_fields=NAME_CONFIRMATION_FIELDS,
    )
    return engine, wf


def test_devanagari_name_correction_is_recognized_and_read_back():
    engine, wf = _workflow_with_stale_candidate()

    result = wf.process_turn("जया बच्चन")

    # The correction replaces the stale candidate and is read back with the
    # already-captured age (approved Step 6B.2); it is not committed until
    # the reporter confirms it.
    assert "I have your details as जया बच्चन, 45." in result.response_text
    assert "कर्मा तेल" not in result.response_text
    assert engine.case.confirmation_candidate == {"patient_identifier": "जया बच्चन"}
    assert engine.case.patient_identifier is None
    assert engine.next_field()[0] == "confirmation"


def test_full_hindi_name_sentence_is_stripped_to_the_name():
    engine, wf = _workflow_with_stale_candidate()

    result = wf.process_turn("जी मेरा नाम जया बच्चन है")

    # The conversational wrapper is stripped; only the name is read back.
    assert engine.case.confirmation_candidate == {"patient_identifier": "जया बच्चन"}
    assert "I have your details as जया बच्चन, 45." in result.response_text


def test_reject_with_nahi_then_devanagari_name_commits_and_advances():
    engine, wf = _workflow_with_stale_candidate()

    rejected = wf.process_turn("नहीं")
    # Only the incorrect detail is asked for: the existing candidate is held
    # until a corrected name or date replaces it.
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "कर्मा तेल"
    }
    assert engine.case.patient_identifier is None
    assert rejected.response_text  # asks which detail was incorrect

    corrected = wf.process_turn("जया बच्चन")
    assert engine.case.confirmation_candidate == {"patient_identifier": "जया बच्चन"}

    committed = wf.process_turn("हाँ")
    assert engine.case.patient_identifier == "जया बच्चन"
    assert engine.case.pending_confirmation is None
    # The interview advances to the next required field.
    assert engine.next_field()[0] == "contact_phone"
    assert committed.response_text


def test_latin_name_correction_still_works_through_the_workflow():
    engine, wf = _workflow_with_stale_candidate(stale="Karma Oil")

    wf.process_turn("Jaya Bachchan")
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "Jaya Bachchan"
    }

    wf.process_turn("yes")
    assert engine.case.patient_identifier == "Jaya Bachchan"
    assert engine.next_field()[0] == "contact_phone"


def test_looks_like_value_script_precision():
    assert _looks_like_value("जया बच्चन") is True
    assert _looks_like_value("Jaya Bachchan") is True
    # Devanagari digits and bare numerals are not plausible names.
    assert _looks_like_value("१२३४५") is False
    assert _looks_like_value("12345") is False
    assert _looks_like_value("ok") is False


# ---- Confirmation-attempt accounting ---------------------------------------


def test_clear_rejection_does_not_increment_failed_attempts():
    for reply in ("no", "नहीं"):
        engine, wf = _workflow_with_stale_candidate()

        wf.process_turn(reply)

        # The rejection only asks which detail was incorrect: the existing
        # candidate is held (not discarded) and no failed attempt counts.
        assert engine.case.confirmation_candidate == {
            "patient_identifier": "कर्मा तेल"
        }
        assert wf._confirmation_attempts.get("patient_identifier", 0) == 0
        assert engine.current_state is not WorkflowState.HUMAN_REVIEW


def test_rejection_then_devanagari_name_completes_without_human_review():
    engine, wf = _workflow_with_stale_candidate()

    wf.process_turn("नहीं")
    wf.process_turn("जया बच्चन")
    committed = wf.process_turn("हाँ")

    assert engine.case.patient_identifier == "जया बच्चन"
    assert engine.case.pending_confirmation is None
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert committed.human_review_required is False
    assert "reviewed by a member of our team" not in committed.response_text
    assert engine.next_field()[0] == "contact_phone"


def test_repeated_ambiguous_responses_still_park_for_review_at_the_limit():
    engine, wf = _workflow_with_stale_candidate()

    replies = ["maybe", "not sure", "hmm?"]
    for reply in replies[:-1]:
        result = wf.process_turn(reply)
        # Still clarifying; not parked yet.
        assert engine.current_state is not WorkflowState.HUMAN_REVIEW
        assert result.human_review_required is False

    final = wf.process_turn(replies[-1])
    assert wf._confirmation_attempts["patient_identifier"] >= 3
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert final.human_review_required is True
    assert "reviewed by a member of our team" in final.response_text


def test_ambiguous_bounded_across_interleaved_rejections():
    # Rejections do not reset the failed-attempt counter: ambiguity stays
    # bounded at the existing limit even when interleaved with clear "no"s.
    engine, wf = _workflow_with_stale_candidate()

    wf.process_turn("maybe")     # attempt 1
    wf.process_turn("नहीं")      # clear rejection: counter unchanged
    wf.process_turn("not sure")  # attempt 2
    wf.process_turn("no")        # clear rejection: counter unchanged
    final = wf.process_turn("hmm?")  # attempt 3 -> human review

    assert wf._confirmation_attempts["patient_identifier"] >= 3
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert final.human_review_required is True


def test_english_confirmation_flow_unchanged_with_attempts_accounting():
    engine, wf = _workflow_with_stale_candidate(stale="Karma Oil")

    readback = wf.process_turn("Jaya Bachchan")
    assert engine.case.confirmation_candidate == {
        "patient_identifier": "Jaya Bachchan"
    }
    assert wf._confirmation_attempts.get("patient_identifier", 0) >= 1

    committed = wf.process_turn("yes")
    assert engine.case.patient_identifier == "Jaya Bachchan"
    assert wf._confirmation_attempts.get("patient_identifier", 0) == 0
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert committed.human_review_required is False
