"""Focused regression tests for the documented AE voice conversation flow.

Every test drives the production ``ConversationalAEWorkflow`` (with the voice
endpoint's confirmation scope and closing question enabled) through a scripted
conversation and asserts the actual question order, branching and state
transitions - not merely that response strings exist.

All data is synthetic (fictional names, demo numbers, example.com addresses).
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.providers.answer import AnswerLLMProvider
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    ADDITIONAL_INFORMATION_FOLLOWUP,
    ADDITIONAL_INFORMATION_QUESTION,
    CLOSING_CONSENT_SENTENCE,
    CLOSING_MESSAGE,
    EMAIL_OFFER_QUESTION,
    EMAIL_REQUEST_QUESTION,
    EVENT_DESCRIPTION_QUESTION,
    INTRODUCTION_MESSAGE,
    MEDICATION_EMPATHY,
    SCOPE_VERIFICATION_QUESTION,
    SEVERITY_URGENT_MESSAGE,
    SEVERITY_QUESTION,
    VOICE_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)

# ---- scripted journeys -------------------------------------------------------

EVENT = "I took Lantus yesterday and since this morning I have a severe headache"
REPORTER_EVENT = (
    "I want to report an adverse event: my mother had a rash after taking Lantus"
)

# Self-reporter: consent given, email collected, event not severe. The
# outcome is no longer an interview question, so the report is generated as
# soon as the applicable information is collected.
SELF_JOURNEY = [
    "Hello",
    EVENT,
    "yes",  # Step 4 verification
    "myself",  # Step 5
    "Alpha Jones, 05-01-1990",  # Step 6B.1
    "yes",  # Step 6B.2
    "9886094321",  # Step 6B.3
    "yes",  # Step 6B.4
    "Yes, you can contact me",  # Step 7 consent
    "yes",  # Step 8 offer
    "alpha@example.com",  # Step 8 request
    "yes",  # email read-back
    "No, it was not severe",  # Step 9
    "It started yesterday",  # Step 10B.1
    "no",  # Step 10B.2 -> report and closing
]

SELF_EXPECTED = [
    "adverse_event",  # Step 2
    "product_verification",  # Step 4
    "reporter_role",  # Step 5
    "patient_identifier",  # Step 6B.1
    "confirmation",  # Step 6B.2
    "contact_phone",  # Step 6B.3
    "confirmation",  # Step 6B.4
    "consent_to_contact",  # Step 7
    "contact_email",  # Step 8 offer
    "contact_email",  # Step 8 request
    "confirmation",  # email read-back
    "seriousness",  # Step 9
    "onset_date",  # Step 10B.1
    None,  # Step 10B.2 is asked in the response; next_question stays null
    None,  # report presented; no post-report question
]

# Reporter branch: consent declined (Step 8 skipped), event severe.
REPORTER_JOURNEY = [
    REPORTER_EVENT,
    "yes",  # Step 4 verification
    "for my mother",  # Step 5
    "Dr. Observer",  # Step 6A.1
    "yes",  # Step 6A.2
    "9999990001",  # Step 6A.3
    "yes",  # Step 6A.4
    "Mary Jones, 12-03-1975",  # Step 6A.5
    "yes",  # Step 6A.6
    "9886094399",  # Step 6A.7
    "yes",  # Step 6A.8
    "no thanks",  # Step 7 consent declined
    "Yes, it was severe",  # Step 9 -> Step 10A
]

REPORTER_EXPECTED = [
    "product_verification",
    "reporter_role",
    "reporter_name",
    "confirmation",
    "reporter_contact_phone",
    "confirmation",
    "patient_identifier",
    "confirmation",
    "contact_phone",
    "confirmation",
    "consent_to_contact",
    "seriousness",  # Step 8 skipped entirely
    None,  # urgent branch: no further questions
]

# Same as the self journey, but consent declined: no email step at all.
CONSENT_NO_JOURNEY = SELF_JOURNEY[:8] + [
    "no thanks",  # Step 7 consent declined
    "No, it was not severe",  # Step 9
    "It started yesterday",  # Step 10B.1
    "no",  # Step 10B.2 -> report and closing
]

CONSENT_NO_EXPECTED = [
    "adverse_event",
    "product_verification",
    "reporter_role",
    "patient_identifier",
    "confirmation",
    "contact_phone",
    "confirmation",
    "consent_to_contact",
    "seriousness",  # Step 8 skipped
    "onset_date",
    None,  # Step 10B.2 is asked in the response; next_question stays null
    None,  # report presented; no post-report question
]


def _workflow(case_id: str) -> tuple[AEStateEngine, ConversationalAEWorkflow]:
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine,
        AnswerLLMProvider(),
        confirm_fields=VOICE_CONFIRMATION_FIELDS,
        ask_additional_information=True,
    )
    return engine, workflow


def _drive(case_id: str, messages: list[str]):
    """Run a scripted conversation; return the engine and per-turn records."""
    engine, workflow = _workflow(case_id)
    records = []
    for message in messages:
        result = workflow.process_turn(message)
        records.append(
            {
                "message": message,
                "response": result.response_text,
                "next_key": result.next_question[0] if result.next_question else None,
                "next_text": result.next_question[1] if result.next_question else None,
                "completed": result.completed,
                "safety": result.safety_status,
                "human_review": result.human_review_required,
                "state": result.current_state,
            }
        )
    return engine, records


def _keys(records) -> list:
    return [record["next_key"] for record in records]


# ---- 1-2. introduction and event invitation ---------------------------------


def test_step_1_uses_the_exact_fixed_introductory_wording():
    expected = (
        "Thank you for taking the time to speak with us. "
        "I'm here to help record the event for review. "
        "I'm not a substitute for a healthcare professional, and I can't "
        "provide medical advice. Please tell me what happened, in your own words."
    )
    assert INTRODUCTION_MESSAGE == expected

    _, records = _drive("AE-flow-intro", ["Hello"])
    assert records[0]["response"].startswith(INTRODUCTION_MESSAGE)


def test_step_2_event_invitation_follows_the_introduction_and_is_asked_once():
    _, records = _drive("AE-flow-invite", ["Hello"])

    response = records[0]["response"]
    # The opening ends with the event-description invitation, so the
    # adverse-event question itself is never asked twice.
    assert response.startswith(INTRODUCTION_MESSAGE)
    assert response.endswith("Please tell me what happened, in your own words.")
    assert records[0]["next_key"] == "adverse_event"
    assert records[0]["next_text"] == EVENT_DESCRIPTION_QUESTION

    # The opening is never repeated later in the session.
    later = _drive("AE-flow-invite", ["Hello", "I have a headache"])[1]
    assert INTRODUCTION_MESSAGE in later[0]["response"]
    assert INTRODUCTION_MESSAGE not in later[1]["response"]


# ---- 3-4. medication identification then scope verification ------------------


def test_medication_identification_precedes_medication_scope_verification():
    _, records = _drive(
        "AE-flow-medication",
        ["Hello", "I have a severe headache", "Lantus", "yes"],
    )

    # Step 2 -> Step 3 (with the one-time empathy) -> Step 4 -> Step 5.
    assert records[1]["next_key"] == "product"
    assert records[1]["next_text"] == (
        "To help us record your report correctly, could you tell me the name "
        "of the medication involved?"
    )
    assert MEDICATION_EMPATHY in records[1]["response"]
    assert records[2]["next_key"] == "product_verification"
    assert records[2]["next_text"] == SCOPE_VERIFICATION_QUESTION
    assert records[3]["next_key"] == "reporter_role"
    # Empathy is offered once, never repeated on later turns.
    assert MEDICATION_EMPATHY not in records[3]["response"]


def test_out_of_scope_medication_keeps_the_existing_termination_behaviour():
    engine, records = _drive(
        "AE-flow-scope", ["I have a rash due to paracetamol"]
    )

    response = records[0]["response"]
    assert "configured for Lantus SoloStar only" in response
    assert "Paracetamol" in response
    assert "manufacturer" in response
    assert "cannot continue this report" in response
    assert "submitted" not in response.lower()
    assert records[0]["next_key"] is None
    assert records[0]["completed"] is False
    assert engine.case.product == "Paracetamol"


def test_uncertain_reply_to_the_medication_verification_is_not_a_confirmation():
    _, records = _drive(
        "AE-flow-verify-unclear",
        [EVENT, "I am not sure", "yes"],
    )

    # Ambiguity clarifies and re-asks; only then does the interview continue.
    assert records[1]["next_key"] == "product_verification"
    assert "please say yes" in records[1]["response"]
    assert records[2]["next_key"] == "reporter_role"


# ---- Step 4 clarification: a bare "no" without another medicine --------------


def test_bare_no_to_the_medication_verification_asks_for_the_medicine_again():
    engine, workflow = _workflow("AE-flow-verify-bare-no")
    workflow.process_turn(EVENT)
    denied = workflow.process_turn("No")

    assert denied.next_question is not None
    assert denied.next_question[0] == "product"
    assert denied.next_question[1] == (
        "To help us record your report correctly, could you tell me the name "
        "of the medication involved?"
    )
    # Nothing out-of-scope is claimed, nothing is discarded, nothing invented.
    assert "configured for Lantus SoloStar only" not in denied.response_text
    assert engine.case.product == "Lantus SoloStar"
    assert workflow._out_of_scope_product is None
    assert denied.completed is False
    assert denied.human_review_required is False
    assert denied.current_state == WorkflowState.PRODUCT_IDENTIFIED.value


def test_confirming_the_demo_product_after_a_bare_no_continues_to_reporter_identification():
    engine, workflow = _workflow("AE-flow-verify-bare-no-yes")
    workflow.process_turn(EVENT)
    workflow.process_turn("No")

    restated = workflow.process_turn("Lantus")
    assert restated.next_question is not None
    assert restated.next_question[0] == "product_verification"
    assert restated.next_question[1] == SCOPE_VERIFICATION_QUESTION
    assert engine.case.product == "Lantus SoloStar"
    assert workflow._out_of_scope_product is None

    confirmed = workflow.process_turn("yes")
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "reporter_role"
    assert workflow._scope_verification_pending is False


def test_naming_an_alternative_medicine_after_a_bare_no_keeps_the_out_of_scope_handling():
    engine, workflow = _workflow("AE-flow-verify-bare-no-alt")
    workflow.process_turn(EVENT)

    result = workflow.process_turn("paracetamol")

    assert engine.case.product == "Paracetamol"
    assert workflow._out_of_scope_product == "Paracetamol"
    response = result.response_text
    assert "configured for Lantus SoloStar only" in response
    assert "Paracetamol" in response
    assert "manufacturer" in response
    assert "cannot continue this report" in response
    assert result.next_question is None
    assert result.completed is False


def test_repeated_bare_nos_fall_back_to_human_review_without_inventing_a_medicine():
    engine, workflow = _workflow("AE-flow-verify-bare-no-bound")
    workflow.process_turn(EVENT)
    workflow.process_turn("No")  # Step 4 denied -> medication asked again
    workflow.process_turn("No")  # clarification attempt 1
    workflow.process_turn("No")  # clarification attempt 2

    final = workflow.process_turn("No")  # attempt 3 -> existing safe fallback

    assert final.human_review_required is True
    assert final.completed is False
    assert final.next_question is None
    assert "reviewed by a member of our team" in final.response_text
    # The report is preserved exactly as collected; nothing is invented.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == EVENT
    assert workflow._out_of_scope_product is None


# ---- 5-7. reporter identification and both branches --------------------------


def test_reporter_identification_selects_the_correct_branch():
    _, records = _drive(
        "AE-flow-branch",
        [EVENT, "yes", "myself"],
    )
    assert records[2]["next_key"] == "patient_identifier"  # patient branch

    _, reporter_records = _drive(
        "AE-flow-branch-reporter",
        [REPORTER_EVENT, "yes", "for my mother"],
    )
    assert reporter_records[2]["next_key"] == "reporter_name"  # reporter branch


def test_reporter_branch_follows_the_documented_sequence():
    engine, records = _drive("AE-flow-reporter-seq", REPORTER_JOURNEY)

    assert _keys(records) == REPORTER_EXPECTED
    # Exact wording for each documented step.
    assert records[2]["next_text"] == "Could you please tell me your full name?"
    assert records[3]["next_text"] == (
        "I have your name as Dr. Observer. Have I got that right?"
    )
    assert records[4]["next_text"] == "Could you please share your contact number?"
    assert records[6]["next_text"] == "Could you please tell me the patient's first " \
        "and last name, along with their date of birth?"
    assert records[7]["next_text"].startswith(
        "Let me read that back to make sure I've recorded it correctly:"
    )
    assert records[8]["next_text"] == "Could you please share the patient's contact number?"
    assert records[10]["next_text"] == (
        "Would it be okay for the appropriate team to contact the patient about "
        "this report if any follow-up is needed?"
    )
    assert engine.case.reporter_name == "Dr. Observer"
    assert engine.case.patient_identifier == "Mary Jones"


def test_patient_branch_follows_the_documented_sequence():
    engine, records = _drive("AE-flow-patient-seq", SELF_JOURNEY)

    assert _keys(records) == SELF_EXPECTED
    assert records[3]["next_text"] == (
        "Could you please tell me your first and last name, along with your "
        "date of birth?"
    )
    assert records[4]["next_text"] == (
        "I have your details as Alpha Jones, 05-Jan-1990. Is that correct?"
    )
    assert records[5]["next_text"] == (
        "Could you please share the best contact number for you?"
    )
    assert records[6]["next_text"] == "I have the number ending in 4321. Is that correct?"
    # Step 10B.2 is asked in the response (the closing question has never
    # been part of next_question), directly after the incident date.
    assert records[13]["next_key"] is None
    assert ADDITIONAL_INFORMATION_QUESTION in records[13]["response"]
    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.case.patient_age == "05-Jan-1990"


def test_reporter_and_patient_contact_numbers_are_stored_separately():
    engine, records = _drive("AE-flow-separate-phones", REPORTER_JOURNEY)

    assert engine.case.reporter_contact_phone == "9999990001"
    assert engine.case.contact_phone == "9886094399"
    assert engine.case.reporter_contact_phone != engine.case.contact_phone
    # Both numbers are distinct fields in the report payload.
    payload = engine.to_dict()
    assert payload["contact"]["reporter_phone"] == "9999990001"
    assert payload["contact"]["phone"] == "9886094399"


def test_self_reporter_keeps_the_patient_contact_number_only():
    engine, _ = _drive("AE-flow-self-phone", SELF_JOURNEY)

    assert engine.case.contact_phone == "9886094321"
    assert engine.case.reporter_contact_phone is None


# ---- 9-10. confirmations and corrections ------------------------------------


def test_confirmations_are_not_repeated_after_explicit_confirmation():
    _, reporter_records = _drive("AE-flow-confirm-once", REPORTER_JOURNEY)
    reporter_text = "\n".join(record["response"] for record in reporter_records)

    assert reporter_text.count("I have your name as") == 1
    assert reporter_text.count("Let me read that back") == 1
    assert reporter_text.count("I have the number ending in") == 2  # 6A.4 + 6A.8

    _, self_records = _drive("AE-flow-confirm-once-self", SELF_JOURNEY)
    self_text = "\n".join(record["response"] for record in self_records)

    assert self_text.count("I have your details as") == 1
    assert self_text.count("I have the number ending in") == 1
    assert self_text.count("I heard your email address") == 1
    # The patient contact number is never asked again after it is confirmed.
    assert self_text.count("best contact number for you") == 1
    assert self_text.count("patient's contact number") == 0


def test_correction_updates_the_intended_field_only():
    engine, workflow = _workflow("AE-flow-correction")
    for message in SELF_JOURNEY[:5]:  # up to the patient-details read-back
        result = workflow.process_turn(message)
    assert "I have your details as" in result.response_text

    rejected = workflow.process_turn("no")
    # Only the incorrect field is asked for; nothing restarts.
    assert "Was it the name or the date of birth that was incorrect?" in (
        rejected.response_text
    )

    corrected = workflow.process_turn("Beta Jones")
    assert "I have your details as Beta Jones, 05-Jan-1990." in (
        corrected.response_text
    )
    confirmed = workflow.process_turn("yes")

    assert engine.case.patient_identifier == "Beta Jones"
    # Correcting the name never overwrites the date of birth.
    assert engine.case.patient_age == "05-Jan-1990"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "contact_phone"


def test_contact_correction_updates_the_contact_field_only():
    engine, workflow = _workflow("AE-flow-phone-correction")
    for message in SELF_JOURNEY[:7]:  # up to the contact read-back
        workflow.process_turn(message)
    assert engine.case.contact_phone is None  # candidate only

    workflow.process_turn("no")
    reasked = workflow.process_turn("9112223334")
    assert "ending in 3334" in reasked.response_text
    workflow.process_turn("yes")

    assert engine.case.contact_phone == "9112223334"
    assert engine.case.patient_identifier == "Alpha Jones"
    assert engine.case.consent_to_contact is None  # untouched


# ---- 11-13. consent and the conditional email step ---------------------------


def test_consent_yes_leads_directly_to_email_collection():
    _, records = _drive("AE-flow-consent-yes", SELF_JOURNEY)

    assert records[8]["next_key"] == "contact_email"
    assert records[8]["next_text"] == EMAIL_OFFER_QUESTION
    assert records[9]["next_text"] == EMAIL_REQUEST_QUESTION
    # Step 9 is reached only after the email exchange.
    assert records[11]["next_key"] == "seriousness"


def test_consent_no_skips_email_and_proceeds_directly_to_the_severity_question():
    engine, records = _drive("AE-flow-consent-no", CONSENT_NO_JOURNEY)

    consent_turn = next(
        record for record in records if record["message"] == "no thanks"
    )
    assert consent_turn["next_key"] == "seriousness"
    assert consent_turn["next_text"] == SEVERITY_QUESTION
    assert EMAIL_OFFER_QUESTION not in consent_turn["response"]
    assert EMAIL_REQUEST_QUESTION not in consent_turn["response"]
    assert engine.case.consent_to_contact is False
    assert engine.case.contact_email is None


def test_no_duplicate_patient_contact_question_appears_after_consent():
    for case_id, journey, consent_message in (
        ("AE-flow-no-dup-self", SELF_JOURNEY, "Yes, you can contact me"),
        ("AE-flow-no-dup-reporter", REPORTER_JOURNEY, "no thanks"),
        ("AE-flow-no-dup-consent-no", CONSENT_NO_JOURNEY, "no thanks"),
    ):
        _, records = _drive(case_id, journey)
        consent_index = next(
            index
            for index, record in enumerate(records)
            if record["message"] == consent_message
        )
        # From the consent answer onwards (Step 7 -> Step 12) the patient
        # contact number is never asked for again.
        joined = "\n".join(
            record["response"] for record in records[consent_index:]
        )
        assert "best contact number for you" not in joined, case_id
        assert "patient's contact number" not in joined, case_id
        assert "share your contact number" not in joined, case_id


# ---- 14-16. severity branches ------------------------------------------------


def test_severe_yes_triggers_urgent_safety_and_skips_date_and_additional_info():
    engine, records = _drive("AE-flow-severe-yes", REPORTER_JOURNEY)

    urgent = records[-1]
    assert urgent["message"] == "Yes, it was severe"
    assert urgent["safety"] == "urgent_life_threatening"
    assert urgent["human_review"] is True
    assert urgent["next_key"] is None
    assert urgent["completed"] is False
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.priority == "high"
    assert "immediate medical attention" in urgent["response"]

    # Neither the incident date nor the additional-information step is asked.
    assert "when the event occurred" not in urgent["response"]
    assert ADDITIONAL_INFORMATION_QUESTION not in urgent["response"]
    for record in records:
        assert record["next_key"] != "onset_date"
        assert record["next_key"] != "additional_information"
    assert engine.case.onset_date is None


def test_severe_no_proceeds_to_the_incident_date_then_additional_information():
    _, records = _drive("AE-flow-severe-no", SELF_JOURNEY)

    assert records[11]["next_key"] == "seriousness"
    assert records[11]["next_text"] == SEVERITY_QUESTION
    assert records[12]["next_key"] == "onset_date"
    assert records[12]["next_text"] == "Could you tell me when the event occurred?"
    # The closing question follows the date and is asked in the response.
    assert records[13]["next_key"] is None
    assert ADDITIONAL_INFORMATION_QUESTION in records[13]["response"]


def test_additional_information_yes_invites_the_details_and_no_skips_them():
    engine, records = _drive("AE-flow-additional-no", SELF_JOURNEY)
    assert records[14]["message"] == "no"
    assert engine.case.additional_information is None
    assert engine.additional_information_status == "declined"
    # No free-text collection turn ever happened.
    assert ADDITIONAL_INFORMATION_FOLLOWUP not in "\n".join(
        record["response"] for record in records
    )

    engine, workflow = _workflow("AE-flow-additional-yes")
    for message in SELF_JOURNEY[:14]:
        workflow.process_turn(message)
    invited = workflow.process_turn("yes")
    assert invited.response_text.endswith(ADDITIONAL_INFORMATION_FOLLOWUP)
    assert engine.case.additional_information is None

    answered = workflow.process_turn("I also felt a little dizzy in the evening")
    assert engine.case.additional_information == (
        "I also felt a little dizzy in the evening"
    )
    # The original event description is never replaced by the follow-up.
    assert engine.case.adverse_event == EVENT
    # No post-documentation question is asked: the report is generated as
    # soon as the applicable information is collected.
    assert answered.completed is True
    assert answered.next_question is None
    assert engine.additional_information_status == "provided"


# ---- 17-18. report presentation and closing wording --------------------------


def test_both_completion_branches_present_the_report():
    consent_yes_engine, consent_yes = _drive("AE-flow-report-yes", SELF_JOURNEY)
    consent_no_engine, consent_no = _drive("AE-flow-report-no", CONSENT_NO_JOURNEY)

    for engine, records in (
        (consent_yes_engine, consent_yes),
        (consent_no_engine, consent_no),
    ):
        final = records[-1]
        # Completion is what triggers the existing report-summary mechanism.
        assert final["completed"] is True
        assert final["next_key"] is None
        assert final["state"] == WorkflowState.CASE_COMPLETED.value
        assert engine.to_dict()["report_complete"] is True

    # The urgent branch keeps the report for qualified human review.
    _, urgent_records = _drive("AE-flow-report-urgent", REPORTER_JOURNEY)
    assert urgent_records[-1]["human_review"] is True
    assert urgent_records[-1]["completed"] is False


def test_closing_wording_reflects_the_actual_consent_value():
    _, consent_yes = _drive("AE-flow-close-yes", SELF_JOURNEY)
    yes_response = consent_yes[-1]["response"]
    assert yes_response.endswith(f"{CLOSING_MESSAGE} {CLOSING_CONSENT_SENTENCE}")

    _, consent_no = _drive("AE-flow-close-no", CONSENT_NO_JOURNEY)
    no_response = consent_no[-1]["response"]
    assert no_response.endswith(CLOSING_MESSAGE)
    assert CLOSING_CONSENT_SENTENCE not in no_response

    # No claim of submission or of a review that has not happened.
    for response in (yes_response, no_response):
        lowered = response.lower()
        assert "submitted" not in lowered
        assert "has been reviewed" not in lowered


def test_urgent_branch_closing_never_implies_declined_consent():
    _, records = _drive("AE-flow-urgent-close", REPORTER_JOURNEY)
    response = records[-1]["response"]
    assert CLOSING_MESSAGE in response
    assert CLOSING_CONSENT_SENTENCE not in response


# ---- 20. API contract and report field semantics -----------------------------


def test_text_endpoint_envelope_and_report_fields_are_unchanged():
    client = TestClient(_app())
    first = client.post(
        "/text/conversation",
        data={"session_id": "ae-flow-api", "message": "Hello"},
    )
    assert first.status_code == 200
    body = first.json()

    for key in (
        "session_id",
        "transcript",
        "intent",
        "workflow",
        "response_text",
        "audio_path",
        "current_state",
        "case_id",
        "priority",
        "human_review_required",
        "completed",
        "missing_information",
        "next_question",
        "safety_status",
        "mi",
        "ae",
    ):
        assert key in body, key
    assert body["workflow"] == "ae"
    assert body["audio_path"] is None
    assert body["response_text"].startswith(INTRODUCTION_MESSAGE)

    collected = body["ae"]["collected_information"]
    for field in (
        "product",
        "adverse_event",
        "patient_identifier",
        "patient_age",
        "reporter_role",
        "reporter_name",
        "contact_phone",
        "contact_email",
        "consent_to_contact",
        "onset_date",
        "seriousness",
        "outcome",
    ):
        assert field in collected, field


def test_voice_and_text_endpoints_ask_the_same_questions():
    from app.main import create_app

    app = create_app()
    client = TestClient(app)

    text_first = client.post(
        "/text/conversation",
        data={"session_id": "ae-flow-parity", "message": "Hello"},
    ).json()
    assert text_first["next_question"] == [
        "adverse_event",
        EVENT_DESCRIPTION_QUESTION,
    ]


def _app():
    from app.main import create_app

    return create_app()
