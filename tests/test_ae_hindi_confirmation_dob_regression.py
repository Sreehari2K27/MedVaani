"""Regression tests: Hindi confirmation affirmatives, Hindi DOB variants, and
the completion guard for the latest AE voice transcript.

Observed defects:
- at product verification ``हाँ सही पकड़े हैं`` was read as "ambiguous" and
  the same question looped (the Devanagari lead was not stripped and the
  "you've got it right" idiom was not a known continuation),
- ``मेरा नाम राजू रस्तोगी है और मेरा जन्मदिन पच्चीस मे दो हजार चार में हुआ
  है`` stored a mangled name and never parsed the date (STT month variants
  ``मे``/``में`` and the ``हज़ार`` nukta spelling were unknown), leaving the
  DOB parked as ``unknown`` behind a premature human-review report.

Pinned here: the exact affirmative phrase and common Hindi affirmatives
advancing the flow, negatives/ambiguous replies never confirming, name+DOB in
one Hindi utterance, the month/nukta date variants parsing to 25-May-2004,
ambiguous (month-less / unit-less) dates staying unconfirmed behind a focused
clarification, invalid-then-corrected candidates, and no premature completion
while required fields remain outstanding — with English confirmation/DOB and
urgent-safety behaviour intact.

Production wiring (``VOICE_CONFIRMATION_FIELDS``), deterministic providers,
no live calls.
"""
from __future__ import annotations

from app.providers.answer import AnswerLLMProvider, _parse_answer, normalized_dob_in
from app.safety import SafetyLayer, SafetySignal
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)

DESCRIPTION = "I took Lantus and have a headache."
AFFIRM_PHRASE = "हाँ सही पकड़े हैं"
COMBINED_HINDI = "मेरा नाम राजू रस्तोगी है और मेरा जन्मदिन पच्चीस मे दो हजार चार में हुआ है"
DOB_VARIANT_JANAM_TITHI = "मेरा जन्म तिथि पच्चीस मे दो हज़ार चार"
DOB_VARIANT_MEIN = "पच्चीस में दो हजार चार"


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    return engine, workflow


def _at_verification(case_id: str):
    engine, workflow = _workflow(case_id)
    workflow.process_turn(DESCRIPTION)
    return engine, workflow


# ---- A. Hindi affirmative confirmation ---------------------------------------


def test_exact_hindi_affirmative_phrase_confirms_medication():
    engine, workflow = _at_verification("AE-aff-exact")

    result = workflow.process_turn(AFFIRM_PHRASE)

    # Confirmed and advanced to the next prescribed question; the
    # verification question is not repeated.
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_role"
    assert "Just to make sure" not in result.response_text
    assert engine.case.product == "Lantus SoloStar"
    assert engine.current_state.value == "PRODUCT_IDENTIFIED"


def test_common_hindi_affirmatives_confirm_medication():
    for case_id, phrase in (
        ("AE-aff-haan", "हाँ"),
        ("AE-aff-jihaan", "जी हाँ"),
        ("AE-aff-bilkul", "बिल्कुल"),
    ):
        engine, workflow = _at_verification(case_id)
        result = workflow.process_turn(phrase)
        assert result.next_question is not None, phrase
        assert result.next_question[0] == "reporter_role", phrase
        assert "Just to make sure" not in result.response_text, phrase
        assert engine.case.product == "Lantus SoloStar", phrase


def test_negative_medication_replies_never_confirm():
    for case_id, phrase in (
        ("AE-neg-haan-nahi", "नहीं"),
        ("AE-neg-galat", "नही, यह गलत है"),
    ):
        engine, workflow = _at_verification(case_id)
        result = workflow.process_turn(phrase)
        # A rejection re-enters the medication name; it never advances to the
        # identity question and never switches the case out of scope.
        assert result.next_question is not None, phrase
        assert result.next_question[0] == "product", phrase
        assert engine.current_state.value == "PRODUCT_IDENTIFIED", phrase
        assert engine.case.product == "Lantus SoloStar", phrase


def test_ambiguous_medication_reply_reasks_without_confirming():
    engine, workflow = _at_verification("AE-amb-shayad")

    result = workflow.process_turn("शायद")

    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"
    assert engine.case.product == "Lantus SoloStar"
    assert engine.current_state.value == "PRODUCT_IDENTIFIED"
    assert result.completed is False


# ---- B. name + DOB in one Hindi utterance ------------------------------------


def test_name_plus_dob_in_single_hindi_utterance():
    engine, workflow = _at_verification("AE-namedob-one")
    workflow.process_turn("yes")  # confirm medication -> identity
    identified = workflow.process_turn("मैं हूँ")  # self branch
    assert identified.next_question[0] == "patient_identifier"

    result = workflow.process_turn(COMBINED_HINDI)

    # Name and date captured independently: clean candidate, normalized DOB,
    # and one combined read-back asking neither question again.
    assert engine.case.confirmation_candidate == {"patient_identifier": "राजू रस्तोगी"}
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.patient_identifier is None
    assert (
        "I have your details as राजू रस्तोगी, 25-May-2004."
        in result.response_text
    )
    assert "जन्मदिन" not in result.response_text

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "राजू रस्तोगी"
    assert engine.case.patient_age == "25-May-2004"
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


# ---- B. Hindi date variants ---------------------------------------------------


def test_hindi_date_variants_parse_to_25_may_2004():
    for text in (
        "पच्चीस मई दो हजार चार",       # canonical
        "पच्चीस मे दो हजार चार",        # STT variant मे
        "पच्चीस में दो हजार चार",       # STT variant में
        "पच्चीस मई दो हज़ार चार",       # nukta spelling हज़ाar
    ):
        parsed = _parse_answer(text, allow_bare_dob=True)
        assert parsed.get("patient_age") == "25-May-2004", text
        assert normalized_dob_in(text) == "25-May-2004", text

    assert (
        _parse_answer("पच्चीस मे दो हजार पाँच", allow_bare_dob=True).get(
            "patient_age"
        )
        == "25-May-2005"
    )
    # Existing English capture is unchanged.
    assert (
        _parse_answer("25 May 2004", allow_bare_dob=True).get("patient_age")
        == "25-May-2004"
    )


def test_ambiguous_dates_stay_unparsed_behind_a_clarification():
    engine, workflow = _at_verification("AE-dob-ambiguous")
    workflow.process_turn("yes")
    workflow.process_turn("मैं हूँ")
    asked_name = workflow.process_turn("राजू रस्तोगी")
    assert asked_name.next_question[0] in ("patient_age", "confirmation")

    for case_text in (
        "पच्चीस दो हजार चार",     # no month at all
        "पच्चीस में दो हजार",      # date-marker month without the unit word
    ):
        assert _parse_answer(case_text, allow_bare_dob=True).get("patient_age") is None, case_text

    result = workflow.process_turn("पच्चीस दो हजार चार")
    assert engine.case.patient_age is None
    assert engine.case.patient_age != "unknown"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    assert "date of birth" in result.response_text.lower()
    assert engine.current_state.value != "HUMAN_REVIEW"
    assert result.completed is False


def test_followup_dob_variants_keep_the_confirmed_candidate():
    engine, workflow = _at_verification("AE-dob-followups")
    workflow.process_turn("yes")
    workflow.process_turn("मैं हूँ")
    workflow.process_turn(COMBINED_HINDI)  # 25-May-2004 + read-back

    for followup in (DOB_VARIANT_JANAM_TITHI, DOB_VARIANT_MEIN):
        result = workflow.process_turn(followup)
        assert engine.case.patient_age == "25-May-2004", followup
        assert "25-May-2004" in result.response_text, followup
        assert engine.case.pending_confirmation == "patient_identifier", followup

    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "राजू रस्तोगी"
    assert engine.current_state.value != "HUMAN_REVIEW"


def test_invalid_candidate_then_correction_replaces_it():
    engine, workflow = _at_verification("AE-dob-invalid-then-fix")
    workflow.process_turn("yes")
    workflow.process_turn("मैं हूँ")
    workflow.process_turn("राजू रस्तोगी")

    invalid = workflow.process_turn("पच्चीस दो हजार चार")
    assert engine.case.patient_age is None
    assert invalid.next_question is not None
    assert invalid.next_question[0] == "patient_age"

    corrected = workflow.process_turn(DOB_VARIANT_MEIN)
    assert engine.case.patient_age == "25-May-2004"
    assert "25-May-2004" in corrected.response_text

    committed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "राजू रस्तोगी"
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.pending_confirmation is None
    assert committed.next_question is not None
    assert committed.next_question[0] == "contact_phone"


# ---- C. completion guard ------------------------------------------------------


def test_no_premature_completion_through_the_hindi_flow():
    engine, workflow = _workflow("AE-no-premature")
    completed_flags = []
    for message in (DESCRIPTION, "yes", "मैं हूँ", COMBINED_HINDI):
        result = workflow.process_turn(message)
        completed_flags.append(result.completed)

    assert completed_flags == [False] * len(completed_flags)
    assert engine.current_state.value != "CASE_COMPLETED"
    assert engine.can_complete() is False
    missing = engine.missing_information()
    assert "consent_to_contact" in missing
    assert "seriousness" in missing
    assert "onset_date" in missing

    # The urgent-safety interruption keeps its existing classification on a
    # fresh case and still never reports completion.
    urgent = SafetyLayer.classify_input(
        "I am having severe difficulty breathing", engine.case
    )
    assert urgent.signal == SafetySignal.URGENT_LIFE_THREATENING
    urgent_engine, urgent_workflow = _workflow("AE-urgent-intact")
    urgent_workflow.process_turn(DESCRIPTION)
    escalated = urgent_workflow.process_turn(
        "After taking Lantus I am having severe difficulty breathing."
    )
    assert escalated.safety_status == "urgent_life_threatening"
    assert escalated.human_review_required is True
    assert escalated.completed is False


def test_english_confirmation_and_dob_stay_intact():
    engine, workflow = _at_verification("AE-english-intact")

    advanced = workflow.process_turn("yes")  # English medication confirmation
    assert advanced.next_question[0] == "reporter_role"

    workflow.process_turn("myself")
    workflow.process_turn("Alpha Jones 25/05/2004")
    # English combined capture: clean name candidate + normalized DOB held
    # for the existing read-back (never committed before confirmation).
    assert engine.case.confirmation_candidate == {"patient_identifier": "Alpha Jones"}
    assert engine.case.patient_identifier is None
    assert engine.case.patient_age == "25-May-2004"
    assert engine.case.pending_confirmation == "patient_identifier"
    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.current_state.value != "HUMAN_REVIEW"
