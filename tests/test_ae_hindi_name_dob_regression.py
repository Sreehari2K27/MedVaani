"""Regression tests: Hindi/Hinglish reporter-name and patient name+DOB capture.

Observed defects:
- ``ज़रूर मेरा नाम बिट्टू जोरदार है`` was read back in full as the reporter's
  name (Hindi filler/wrapper/trail were only stripped on corrections),
- ``अल्फा बीटा और जन्मदिन पच्चीस में दो हजार पाँच`` stored the whole sentence
  as the patient name and the Hindi date form
  ``पच्चीस मई दो हजार चार`` never parsed, so the valid correction was counted
  as an unclear reply and the field-attempt bound parked the case to human
  review with ``patient_age='unknown'`` — producing a report while consent,
  seriousness, onset and contact were still outstanding.

Pinned here: name/DOB extracted independently, Hindi DOB normalized to
``25-May-2004``, corrections replacing the earlier candidate, unparseable
dates staying unconfirmed behind a focused clarification, no completed report
while required fields remain outstanding, and the existing English name/date,
confirmation and urgent-safety behaviour staying intact.

Production wiring (``VOICE_CONFIRMATION_FIELDS``), deterministic providers,
no live calls.
"""
from __future__ import annotations

from app.providers.answer import (
    AnswerLLMProvider,
    _parse_answer,
    normalized_dob_in,
)
from app.safety import SafetyLayer, SafetySignal
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)

DESCRIPTION = "I took Lantus and have a headache."
CONFIRM_PRODUCT = "yes"
HINDI_NAME_SENTENCE = "ज़रूर मेरा नाम बिट्टू जोरदार है"
CLEAN_NAME = "बिट्टू जोरदार"
COMBINED_VALID = "अल्फा बीटा और जन्मदिन पच्चीस मई दो हजार चार"
COMBINED_VALID_2005 = "अल्फा बीटा और जन्मदिन पच्चीस मई दो हजार पाँच"
# Ambiguous = no month word at all ("पच्चीस दो हजार …"): the STT month
# variants मे/में now parse as May in the complete date shape, so a missing
# month is the form that must stay unparsed behind a focused clarification.
COMBINED_INVALID = "अल्फा बीटा और जन्मदिन पच्चीस दो हजार पाँच"
HINDI_DOB = "पच्चीस मई दो हजार चार"
HINDI_DOB_BAD = "पच्चीस दो हजार चार"


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    return engine, workflow


def _at_patient_identifier(case_id: str):
    engine, workflow = _workflow(case_id)
    workflow.process_turn(DESCRIPTION)
    workflow.process_turn(CONFIRM_PRODUCT)
    asked = workflow.process_turn("मैं हूँ")  # identity: self branch
    assert asked.next_question is not None
    assert asked.next_question[0] == "patient_identifier"
    return engine, workflow


def _at_reporter_name(case_id: str):
    engine, workflow = _workflow(case_id)
    workflow.process_turn(DESCRIPTION)
    workflow.process_turn(CONFIRM_PRODUCT)
    asked = workflow.process_turn("किसी और का")  # identity: reporter branch
    assert asked.next_question is not None
    assert asked.next_question[0] == "reporter_name"
    return engine, workflow


# ---- 1. reporter name extraction --------------------------------------------


def test_hindi_reporter_name_sentence_extracts_only_the_name():
    engine, workflow = _at_reporter_name("AE-name-hindi")

    result = workflow.process_turn(HINDI_NAME_SENTENCE)

    assert result.extracted_information.get("reporter_name") == CLEAN_NAME
    assert engine.case.pending_confirmation == "reporter_name"
    # The confirmation question carries the extracted name only.
    assert f"I have your name as {CLEAN_NAME}." in result.response_text
    assert "ज़रूर मेरा नाम" not in result.response_text
    assert HINDI_NAME_SENTENCE not in result.response_text


def test_reporter_name_rejection_and_correction_still_work():
    engine, workflow = _at_reporter_name("AE-name-hindi-fix")
    workflow.process_turn(HINDI_NAME_SENTENCE)

    rejected = workflow.process_turn("no")
    assert rejected.next_question is not None
    assert rejected.next_question[0] == "reporter_name"

    corrected = workflow.process_turn(CLEAN_NAME)
    assert f"I have your name as {CLEAN_NAME}." in corrected.response_text

    committed = workflow.process_turn("yes")
    assert engine.case.reporter_name == CLEAN_NAME
    assert engine.case.pending_confirmation is None
    assert committed.next_question is not None
    assert committed.next_question[0] == "reporter_contact_phone"


def test_english_reporter_name_lead_still_stripped():
    engine, workflow = _at_reporter_name("AE-name-english")
    workflow.process_turn("my name is Beta Jones")
    workflow.process_turn("yes")
    assert engine.case.reporter_name == "Beta Jones"


# ---- 2. patient name + DOB in one response ----------------------------------


def test_valid_combined_name_and_hindi_dob_in_one_response():
    engine, workflow = _at_patient_identifier("AE-combined-ok")

    result = workflow.process_turn(COMBINED_VALID)

    # Both values extracted independently: normalized DOB, clean name held
    # as the confirmation candidate, and the combined read-back asks once.
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.confirmation_candidate == {"patient_identifier": "अल्फा बीटा"}
    assert engine.case.patient_identifier is None  # not committed before "yes"
    assert (
        "I have your details as अल्फा बीटा, 25-May-2004."
        in result.response_text
    )

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "अल्फा बीटा"
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.pending_confirmation is None
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_unparseable_combined_date_splits_name_and_asks_only_for_dob():
    engine, workflow = _at_patient_identifier("AE-combined-bad")

    result = workflow.process_turn(COMBINED_INVALID)

    # The name is captured cleanly; the date attempt is never part of it and
    # is never stored: only a focused DOB question follows.
    assert engine.case.confirmation_candidate == {"patient_identifier": "अल्फा बीटा"}
    assert engine.case.patient_age is None
    assert "जन्मदिन" not in engine.case.confirmation_candidate["patient_identifier"]
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    assert "date of birth" in result.next_question[1].lower()
    assert result.completed is False
    assert engine.current_state.value != "HUMAN_REVIEW"


# ---- 3. Hindi DOB parsing ----------------------------------------------------


def test_hindi_dob_parses_to_normalized_date():
    parsed = _parse_answer(HINDI_DOB, allow_bare_dob=True)
    assert parsed.get("patient_age") == "25-May-2004"

    parsed_2005 = _parse_answer("पच्चीस मई दो हजार पाँच", allow_bare_dob=True)
    assert parsed_2005.get("patient_age") == "25-May-2005"

    assert normalized_dob_in(HINDI_DOB) == "25-May-2004"
    # Existing English forms are unchanged.
    assert (
        _parse_answer("25 May 2004", allow_bare_dob=True).get("patient_age")
        == "25-May-2004"
    )


# ---- 4. corrected DOB replaces the earlier candidate -------------------------


def test_valid_combined_name_and_malayalam_dob_in_one_response():
    # The observed Malayalam failure: the combined name + spoken-DOB answer
    # parsed neither value. With the Malayalam spoken-date table the name is
    # held for confirmation and the DOB is normalized, read back and committed.
    engine, workflow = _at_patient_identifier("AE-combined-ml")

    result = workflow.process_turn(
        "എന്റെ പേരാണ് രാജു വർഗീസ് ജന്മദിനം ഇരുപത് മെയ് രണ്ടായിരം"
    )

    assert engine.case.patient_age == "20-May-2000"
    assert engine.case.confirmation_candidate == {"patient_identifier": "രാജു വർഗീസ്"}
    assert "രാജു വർഗീസ്" in result.response_text
    assert "20-May-2000" in result.response_text

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "രാജു വർഗീസ്"
    assert engine.case.patient_age == "20-May-2000"
    assert engine.case.pending_confirmation is None
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_corrected_dob_replaces_earlier_candidate():
    engine, workflow = _at_patient_identifier("AE-dob-correct")
    workflow.process_turn(COMBINED_VALID_2005)  # holds 25-May-2005
    assert engine.case.patient_age == "25-May-2005"

    rejected = workflow.process_turn("no")
    assert rejected.next_question is not None
    assert "date of birth" in rejected.next_question[1].lower()

    corrected = workflow.process_turn(HINDI_DOB)
    assert engine.case.patient_age == "25-May-2004"
    assert "25-May-2004" in corrected.response_text
    assert "25-May-2005" not in corrected.response_text

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "अल्फा बीटा"
    assert engine.case.patient_age == "25-May-2004"


# ---- 5. unparseable DOB stays unconfirmed ------------------------------------


def test_unparseable_hindi_dob_stays_unconfirmed_behind_a_clarification():
    engine, workflow = _at_patient_identifier("AE-dob-unconfirmed")
    workflow.process_turn(COMBINED_INVALID)

    result = workflow.process_turn(HINDI_DOB_BAD)

    assert engine.case.patient_age is None
    assert engine.case.patient_age != "unknown"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    assert "date of birth" in result.response_text.lower()
    assert engine.current_state.value != "HUMAN_REVIEW"
    assert result.completed is False
    # The held name candidate and the pending read-back are untouched.
    assert engine.case.confirmation_candidate == {"patient_identifier": "अल्फा बीटा"}
    assert engine.case.pending_confirmation == "patient_identifier"


# ---- 6. completion guard ------------------------------------------------------


def test_no_completed_report_while_required_fields_outstanding():
    engine, workflow = _at_patient_identifier("AE-no-premature-report")
    seen_completed = []
    for message in (COMBINED_INVALID, HINDI_DOB_BAD, HINDI_DOB):
        result = workflow.process_turn(message)
        seen_completed.append(result.completed)
    confirmed = workflow.process_turn("yes")
    seen_completed.append(confirmed.completed)

    assert seen_completed == [False] * len(seen_completed)
    assert engine.current_state.value != "CASE_COMPLETED"
    assert engine.can_complete() is False
    missing = engine.missing_information()
    assert "consent_to_contact" in missing
    assert "seriousness" in missing
    assert "onset_date" in missing

    # A fresh case that has only captured the event description can never be
    # completed either (the premature-report invariant), and the urgent
    # interruption still classifies as the existing safety signal.
    fresh, fresh_workflow = _workflow("AE-no-premature-2")
    described = fresh_workflow.process_turn(DESCRIPTION)
    assert described.completed is False
    assert fresh.can_complete() is False
    urgent = SafetyLayer.classify_input(
        "I am having severe difficulty breathing", fresh.case
    )
    assert urgent.signal == SafetySignal.URGENT_LIFE_THREATENING


# ---- 5. observed browser-test DOB utterances (trailing-particle fix) ---------


def test_dob_with_trailing_particle_after_the_year_lead_parses():
    # The observed failure: "पच्चीस मई दो हजार है" was rejected because the
    # trailing है poisoned the year-unit lookup. The unambiguous month मई
    # makes the date complete: "दो हजार" alone is exactly 2000.
    assert normalized_dob_in("मेरा जन्म तिथि पच्चीस मई दो हजार है") == "25-May-2000"
    assert normalized_dob_in("पच्चीस मई दो हजार है") == "25-May-2000"
    # A unit word may also follow the particle.
    assert normalized_dob_in("पच्चीस मई दो हजार चार है") == "25-May-2004"


def test_variant_month_with_only_a_trailing_particle_stays_ambiguous():
    # मे/में still require the year unit word: the particle alone never
    # disambiguates them ("पच्चीस में दो हजार" can be "in 2000").
    assert normalized_dob_in("मेरा जन्म तिथि है पच्चीस में दो हजार") is None
    assert normalized_dob_in("पच्चीस में दो हजार है") is None
    assert normalized_dob_in("पच्चीस मे दो हज़ार") is None


def test_combined_name_and_ambiguous_dob_clarifies_then_corrected_dob_confirms():
    # The observed browser sequence: a combined name + ambiguous date turn
    # re-asks the DOB question (no guessed date, no restart), and the
    # corrected complete DOB ("मई" + trailing है) is extracted, read back and
    # committed — so the report carries the confirmed date, never "unknown".
    engine, workflow = _at_patient_identifier("AE-dob-hai-fix")

    first = workflow.process_turn(
        "मेरा नाम है श्री हरि नायर और मेरा जन्म तिथि है पच्चीस में दो हजार"
    )

    # The name is captured cleanly; the ambiguous date is never stored.
    candidate = engine.case.confirmation_candidate.get("patient_identifier")
    assert candidate and "जन्म" not in candidate
    assert engine.case.patient_age is None
    assert first.next_question is not None
    assert first.next_question[0] == "patient_age"
    assert "date of birth" in first.response_text.lower()

    corrected = workflow.process_turn("मेरा जन्म तिथि पच्चीस मई दो हजार है")

    # The corrected complete DOB is extracted immediately and read back for
    # confirmation together with the held name.
    assert "25-May-2000" in corrected.response_text
    assert engine.case.patient_age == "25-May-2000"
    assert engine.case.confirmation_candidate == {"patient_identifier": candidate}

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == candidate
    assert engine.case.patient_age == "25-May-2000"
    assert engine.case.patient_age != "unknown"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


def test_repeated_ambiguous_dob_is_bounded_with_a_targeted_clarification():
    # The observed ambiguous variants never loop forever: the repeat offers
    # the targeted DOB clarification, and the third unclear answer parks the
    # field as "unknown" for human review instead of looping the question.
    engine, workflow = _at_patient_identifier("AE-dob-hai-bounded")
    workflow.process_turn(
        "मेरा नाम है श्री हरि नायर और मेरा जन्म तिथि है पच्चीस में दो हजार"
    )

    second = workflow.process_turn("पच्चीस में दो हजार है")
    assert engine.case.patient_age is None
    assert "Could you say the date of birth again?" in second.response_text
    assert second.next_question is not None
    assert second.next_question[0] == "patient_age"

    third = workflow.process_turn("पच्चीस मे दो हज़ार")
    assert engine.case.patient_age == "unknown"
    assert third.human_review_required is True
    assert third.completed is False
