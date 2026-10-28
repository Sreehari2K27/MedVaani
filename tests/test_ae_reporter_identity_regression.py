"""Regression tests: the reporter/patient identity branch must not loop on
the Hindi someone-else replies.

Observed loop: at the identity question the replies
``मैं किसी और की तरफ से बता रहा हूँ`` and ``किसी और का`` were not
recognised as "reporting for someone else", so the workflow re-asked the
identity question forever. Root cause: ``_ROLE_OTHER`` in
``app/providers/answer.py`` lacked the "किसी और … तरफ से" on-behalf-of
phrasing and the bare "किसी और (का)" variant (only "कोई और" was known).

Expected, covered here: both replies select the reporter branch, the branch
persists, the next question is the reporter's name, the identity question is
never re-asked for equivalent replies, and English/self flows are unchanged.

All synthetic: deterministic provider, in-process workflow, no live calls.
"""
from __future__ import annotations

import pytest

from app.providers.answer import AnswerLLMProvider
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import ConversationalAEWorkflow

DESCRIPTION = "I took Lantus and have a headache."
CONFIRM_PRODUCT = "yes"
ON_BEHIND_HINDI = "मैं किसी और की तरफ से बता रहा हूँ"
KISI_AUR_KA = "किसी और का"
REPORTER_NAME = "रमेश कुमार"


def _start_at_identity(case_id: str):
    """Drive the conversation up to the reporter/patient identity question."""
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn(DESCRIPTION)
    asked = workflow.process_turn(CONFIRM_PRODUCT)
    assert asked.next_question is not None
    assert asked.next_question[0] == "reporter_role"
    return engine, workflow


# ---- both Hindi replies select the reporter branch ---------------------------


def test_hindi_on_behalf_reply_selects_reporter_branch():
    engine, workflow = _start_at_identity("AE-identity-onbehalf")

    result = workflow.process_turn(ON_BEHIND_HINDI)

    assert engine.case.reporter_role == "someone_else"
    assert result.extracted_information.get("reporter_role") == "someone_else"
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_name"
    assert "name" in result.next_question[1].lower()
    assert "Just to check" not in result.response_text


def test_hindi_kisi_aur_ka_selects_reporter_branch():
    engine, workflow = _start_at_identity("AE-identity-kisiaurka")

    result = workflow.process_turn(KISI_AUR_KA)

    assert engine.case.reporter_role == "someone_else"
    assert result.extracted_information.get("reporter_role") == "someone_else"
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_name"
    assert "Just to check" not in result.response_text


# ---- the observed conversation must not loop --------------------------------


def test_identity_question_never_loops_after_branch_selected():
    engine, workflow = _start_at_identity("AE-identity-noloop")

    identity_asks = 1  # the question that brought us here
    turns = []
    for message in (ON_BEHIND_HINDI, KISI_AUR_KA, REPORTER_NAME):
        result = workflow.process_turn(message)
        turns.append(result)
        if result.next_question is not None and result.next_question[0] == "reporter_role":
            identity_asks += 1

    # The branch was chosen on the first Hindi reply and persisted: the
    # identity question is never asked again for equivalent replies.
    assert identity_asks == 1
    assert engine.case.reporter_role == "someone_else"
    assert turns[0].next_question[0] == "reporter_name"
    assert turns[1].next_question[0] == "reporter_name"

    # The flow continues down the approved reporter branch: name captured,
    # next comes the reporter's contact number.
    assert engine.case.reporter_name == REPORTER_NAME
    assert turns[2].next_question is not None
    assert turns[2].next_question[0] == "reporter_contact_phone"


# ---- extractor level (pending-field path) -----------------------------------


def test_extractor_recognizes_both_replies_when_identity_pending():
    provider = AnswerLLMProvider()
    context = {
        "next_field": [
            "reporter_role",
            "May I ask whether you're the person who experienced this event, "
            "or whether you're reporting it on someone else's behalf?",
        ]
    }
    for message in (ON_BEHIND_HINDI, KISI_AUR_KA):
        result = provider.interpret(message, context)
        assert result.extracted_fields.get("reporter_role") == "someone_else", message


# ---- unchanged behaviour -----------------------------------------------------


def test_english_identity_responses_still_work():
    engine, workflow = _start_at_identity("AE-identity-english")

    other = workflow.process_turn("someone else")
    assert engine.case.reporter_role == "someone_else"
    assert other.next_question[0] == "reporter_name"

    engine2, workflow2 = _start_at_identity("AE-identity-english-self")
    self_reply = workflow2.process_turn("myself")
    assert engine2.case.reporter_role == "self"
    assert self_reply.next_question[0] == "patient_identifier"


def test_hindi_self_reply_still_selects_self_branch():
    # The new someone-else phrasings must never hijack a self answer.
    engine, workflow = _start_at_identity("AE-identity-hindi-self")

    result = workflow.process_turn("मैं हूँ")

    assert engine.case.reporter_role == "self"
    assert result.next_question[0] == "patient_identifier"


# ---- Malayalam first-person answers must advance (reported defect) -----------
#
# Reported sequence: the identity question was answered in the first person
# ("it happened to me") and the assistant asked the same question again.
# Traced root cause: the Malayalam first-person cues in ``_ROLE_SELF`` were
# incomplete — plain "it happened to me" forms, the accusative-emphatic
# "എന്നെ തന്നെയാണ് …", the chillu possessive spelling ``എൻ്റെ`` the browser
# STT emits (already accepted by the name-lead stripper), "എനിക്കുണ്ടായ …"
# and "ഞാൻ തന്നെ" all classified as nothing, so the valid answers never
# reached ``reporter_role`` and the workflow re-asked. The state transition,
# confirmation handling and next-question selection were verified correct:
# classification is the only defect.

MALAYALAM_SELF_ANSWERS = [
    # Plain "it happened to me" (dative + past verb, no emphatic).
    "എനിക്ക് സംഭവിച്ചു",
    "അത് എനിക്ക് സംഭവിച്ചു",
    # Accusative-emphatic "to me myself it happened".
    "എന്നെ തന്നെയാണ് സംഭവിച്ചത്",
    # Chillu spelling എൻ്റെ: the audited combined name/DOB transcript
    # (test_ae_conversation_quality.py) carries it and the name-lead
    # stripper already accepts both spellings.
    "എൻ്റെ സ്വന്തം അനുഭവമാണ്",
    "എൻ്റെ അനുഭവമാണ്",
    # "The experience I had".
    "ഇത് എനിക്കുണ്ടായ അനുഭവമാണ്",
    # Emphatic first person, and the anchored bare "it is I".
    "ഞാൻ തന്നെ",
    "ഞാനാണ്",
]


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(a, id=f"ml-self-{i}")
        for i, a in enumerate(MALAYALAM_SELF_ANSWERS)
    ],
)
def test_malayalam_first_person_answer_advances_to_patient_details(answer):
    index = MALAYALAM_SELF_ANSWERS.index(answer)
    engine, workflow = _start_at_identity(f"AE-identity-ml-self-{index}")

    result = workflow.process_turn(answer)

    assert engine.case.reporter_role == "self", answer
    assert result.extracted_information.get("reporter_role") == "self", answer
    assert result.next_question is not None, answer
    assert result.next_question[0] == "patient_identifier", answer
    assert "Just to check" not in result.response_text, answer


def test_english_happened_to_me_advances_to_patient_details():
    engine, workflow = _start_at_identity("AE-identity-english-happened")

    result = workflow.process_turn("it happened to me")

    assert engine.case.reporter_role == "self"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier"


def test_ambiguous_malayalam_affirmative_still_triggers_clarification():
    engine, workflow = _start_at_identity("AE-identity-ml-ambiguous")

    result = workflow.process_turn("അതെ")

    # A bare affirmative answers neither branch: no role is ever guessed,
    # and the bounded clarification (not human review) is the response.
    assert engine.case.reporter_role is None
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_role"
    assert "Just to check" in result.response_text
    assert result.human_review_required is False


def test_identity_question_not_repeated_after_malayalam_self_answer():
    engine, workflow = _start_at_identity("AE-identity-ml-norepeat")

    identity_asks = 1  # the question that brought us here
    result = workflow.process_turn("എനിക്ക് സംഭവിച്ചു")
    if (
        result.next_question is not None
        and result.next_question[0] == "reporter_role"
    ):
        identity_asks += 1

    assert identity_asks == 1
    assert engine.case.reporter_role == "self"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier"

    # The self branch continues normally: with this harness's default
    # (no read-back) confirmation scope the identity values commit directly
    # and the role question never returns. The read-back confirmation and
    # correction flows are covered by the existing confirmation tests.
    identity_turn = workflow.process_turn(
        "എൻ്റെ പേരാണ് രാജു പാണ്ഡെ പിന്നെ ജന്മതിഥി ഇരുപത് മെയ് രണ്ടായിരം"
    )
    assert engine.case.patient_identifier == "രാജു പാണ്ഡെ"
    assert engine.case.patient_age == "20-May-2000"
    assert identity_turn.next_question is not None
    assert identity_turn.next_question[0] != "reporter_role"


def test_chillu_possessive_someone_else_answer_still_selects_that_branch():
    # The chillu spelling എൻ്റെ is now accepted on both sides: a chillu
    # possessive before a relative must stay someone_else (checked before
    # _ROLE_SELF, as for every mixed answer).
    engine, workflow = _start_at_identity("AE-identity-ml-chillu-other")

    result = workflow.process_turn("ഇല്ല, എൻ്റെ ഭാര്യക്ക് സംഭവിച്ചത്")

    assert engine.case.reporter_role == "someone_else"
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_name"


@pytest.mark.parametrize("answer", MALAYALAM_SELF_ANSWERS[:3])
def test_extractor_recognizes_malayalam_self_answers_when_identity_pending(
    answer,
):
    provider = AnswerLLMProvider()
    context = {
        "next_field": [
            "reporter_role",
            "May I ask whether you're the person who experienced this event, "
            "or whether you're reporting it on someone else's behalf?",
        ]
    }
    result = provider.interpret(answer, context)
    assert result.extracted_fields.get("reporter_role") == "self", answer


# ---- Malayalam other-person report phrases (reported defect) ------------------
#
# Reported sequence: the identity question was answered with the Malayalam
# "reporting about another person" phrases; neither was recognised, the
# question repeated once, and the second unclear reply tripped the bounded
# review park - which surfaced as a "report summary" with no details
# collected. Both phrases must select the other-person branch, and the park
# must be recoverable rather than terminal.

ML_OTHER_REPORT_PHRASES = [
    # The two reported phrases (both റ-cluster spellings of "report").
    "മറ്റൊരാള്ക്കാണ് റിപ്പോര്ട്ട് ചെയ്യണത്",
    "മറ്റാരുടെ റിപ്പോർട്ടാണ്",
    # Reasonable spelling variations: the other dative spelling and the
    # other report-word spelling.
    "മറ്റൊരാൾക്കാണ് റിപ്പോർട്ട് ചെയ്യണത്",
    "മറ്റൊരാൾക്ക് ആണ് റിപ്പോര്ട്ട്",
    # The later reported variants: the ള് dative spelling before വേണ്ടി,
    # and the possessive-pronoun follow-up.
    "മറ്റൊരാള്ക്ക് വേണ്ടി റിപ്പോര്ട്ട് ചെയ്യണം",
    "മറ്റാരുടേതോ",
]


@pytest.mark.parametrize(
    "phrase",
    [
        pytest.param(p, id=f"ml-other-{i}")
        for i, p in enumerate(ML_OTHER_REPORT_PHRASES)
    ],
)
def test_malayalam_other_person_report_phrases_select_that_branch(phrase):
    provider = AnswerLLMProvider()
    context = {
        "next_field": [
            "reporter_role",
            "May I ask whether you're the person who experienced this event, "
            "or whether you're reporting it on someone else's behalf?",
        ]
    }
    result = provider.interpret(phrase, context)
    assert result.extracted_fields.get("reporter_role") == "someone_else", phrase


def _start_at_identity_voice(case_id: str):
    """Drive the voice-scope workflow up to the identity question."""
    from app.workflow.service.conversational import VOICE_CONFIRMATION_FIELDS

    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn(DESCRIPTION)
    asked = workflow.process_turn(CONFIRM_PRODUCT)
    assert asked.next_question is not None
    assert asked.next_question[0] == "reporter_role"
    return engine, workflow


def test_reported_variant_phrases_lead_to_reporter_details_next_turn():
    # The very next assistant turn after the user selects someone else must
    # ask for the REPORTER's details - never a repeated identity question
    # and never a report summary. The possessive-pronoun follow-up selects
    # the same branch when it is the reply after an ambiguous attempt.
    engine, workflow = _start_at_identity_voice("AE-identity-ml-other-variant")

    branch = workflow.process_turn("മറ്റൊരാള്ക്ക് വേണ്ടി റിപ്പോര്ട്ട് ചെയ്യണം")
    assert engine.case.reporter_role == "someone_else"
    assert branch.next_question is not None
    assert branch.next_question[0] == "reporter_name"
    # Selecting the branch is not a terminal event.
    assert branch.completed is False
    assert branch.human_review_required is False

    engine2, workflow2 = _start_at_identity_voice("AE-identity-ml-other-pronoun")
    workflow2.process_turn("hmm")  # ambiguous: clarification, attempt 1
    followup = workflow2.process_turn("മറ്റാരുടേതോ")
    assert engine2.case.reporter_role == "someone_else"
    assert followup.next_question is not None
    assert followup.next_question[0] == "reporter_name"


def test_malayalam_other_person_branch_collects_reporter_details_first():
    # Selecting the other-person branch must lead into the reporter-detail
    # collection (reporter name read-back, then the reporter's own contact
    # number) BEFORE any patient-specific detail is asked, with the two
    # identities kept apart.
    engine, workflow = _start_at_identity_voice("AE-identity-ml-other-journey")

    branch = workflow.process_turn("മറ്റൊരാള്ക്കാണ് റിപ്പോര്ട്ട് ചെയ്യണത്")
    assert engine.case.reporter_role == "someone_else"
    assert branch.next_question is not None
    assert branch.next_question[0] == "reporter_name"

    name = workflow.process_turn("Suresh Gopi")
    assert engine.case.reporter_name is None  # held for the read-back
    assert name.next_question is not None
    assert name.next_question[0] == "confirmation"

    name_confirmed = workflow.process_turn("yes")
    assert engine.case.reporter_name == "Suresh Gopi"
    assert name_confirmed.next_question is not None
    assert name_confirmed.next_question[0] == "reporter_contact_phone"

    phone = workflow.process_turn("nine eight eight six zero nine four three two one")
    assert engine.case.reporter_contact_phone is None  # held for the read-back
    assert phone.next_question is not None
    assert phone.next_question[0] == "confirmation"

    phone_confirmed = workflow.process_turn("yes")
    assert engine.case.reporter_contact_phone == "9886094321"
    # Only now is the PATIENT identity asked - never the reporter's fields
    # again, and never the patient's values filled from the reporter's.
    assert phone_confirmed.next_question is not None
    assert phone_confirmed.next_question[0] == "patient_identifier"
    assert engine.case.patient_identifier is None


def test_other_person_flow_separates_reporter_and_patient_to_completion():
    # The complete other-person journey: reporter details, then patient
    # details, then the remaining required fields, finishing only when
    # everything required is satisfied.
    engine, workflow = _start_at_identity_voice("AE-identity-ml-other-full")

    workflow.process_turn("മറ്റാരുടെ റിപ്പോർട്ടാണ്")
    workflow.process_turn("Suresh Gopi")
    workflow.process_turn("yes")
    workflow.process_turn("nine eight eight six zero nine four three two one")
    workflow.process_turn("yes")

    details = workflow.process_turn("Priya Nair 15 March 1995")
    assert engine.case.patient_age == "15-Mar-1995"
    assert details.next_question is not None
    assert details.next_question[0] == "confirmation"
    details_confirmed = workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Priya Nair"
    # Reporter and patient identities remain distinct in the state.
    assert engine.case.reporter_name == "Suresh Gopi"
    assert engine.case.reporter_name != engine.case.patient_identifier
    assert engine.case.reporter_contact_phone == "9886094321"
    assert details_confirmed.next_question is not None
    assert details_confirmed.next_question[0] == "contact_phone"

    workflow.process_turn("nine two two four four one nine one zero three")
    workflow.process_turn("yes")  # patient contact number confirmed
    workflow.process_turn("Yes, I consent")  # follow-up consent
    workflow.process_turn("no")  # email declined (optional)
    severity = workflow.process_turn("no")
    assert severity.next_question is not None
    assert severity.next_question[0] == "onset_date"

    from app.workflow.ae_engine import WorkflowState

    workflow.process_turn("yesterday")
    assert engine.current_state is WorkflowState.CASE_COMPLETED
    snapshot = workflow._collected_snapshot()
    # The report keeps both identities and both numbers separate.
    assert snapshot["reporter_name"] == "Suresh Gopi"
    assert snapshot["reporter_contact_phone"] == "9886094321"
    assert snapshot["patient_identifier"] == "Priya Nair"
    assert snapshot["patient_age"] == "15-Mar-1995"
    assert snapshot["contact_phone"] == "9224419103"
    assert snapshot["consent_to_contact"] is True


def test_unclear_identity_replies_park_recoverably_and_a_valid_answer_resumes():
    # Two unclear replies still park the case for review (bounded, never a
    # completion), and a later valid identity answer RESUMES the interview
    # with every collected detail preserved.
    from app.workflow.ae_engine import WorkflowState

    engine, workflow = _start_at_identity_voice("AE-identity-ml-park-resume")
    workflow.process_turn("hmm")  # attempt 1 -> clarification

    parked = workflow.process_turn("not sure")  # attempt 2 -> review park
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.completed is False
    assert parked.human_review_required is True
    # Report readiness distinguishes incomplete review from completion: the
    # unresolved field and every missing required field stay visible.
    assert parked.missing_information
    assert "reporter_role" in parked.missing_information
    assert engine.case.reporter_role is None
    # Nothing collected was discarded.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event

    resumed = workflow.process_turn("മറ്റൊരാള്ക്കാണ് റിപ്പോര്ട്ട് ചെയ്യണത്")

    assert engine.case.reporter_role == "someone_else"
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert resumed.human_review_required is False
    assert resumed.next_question is not None
    assert resumed.next_question[0] == "reporter_name"
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event


def test_marathi_self_experience_phrase_selects_the_patient_branch():
    # The reported browser transcript: "अनुभव मला स्वत ला आला आहे" (the
    # experience came to me myself) is a first-person answer and must
    # select the self-reporter branch, never the other-person branch and
    # never leave the role unclassified.
    phrase = "अनुभव मला स्वत ला आला आहे"
    result = AnswerLLMProvider().interpret(
        phrase,
        {
            "next_field": [
                "reporter_role",
                "May I ask whether you're the person who experienced this "
                "event, or whether you're reporting it on someone else's behalf?",
            ]
        },
    )
    assert result.extracted_fields.get("reporter_role") == "self"

    engine, workflow = _start_at_identity("AE-identity-mr-self-experience")
    advanced = workflow.process_turn(phrase)

    assert engine.case.reporter_role == "self"
    assert advanced.next_question is not None
    assert advanced.next_question[0] == "patient_identifier"


# ---- reported Hindi self-answers (role question repeated) ---------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "यह मेरे साथ ही हुई घटना है",
        "ये मेरा अब खुद का अनुभव है",
    ],
    ids=["happened-to-me-myself", "my-own-experience"],
)
def test_reported_hindi_self_answers_advance_without_repeating(phrase):
    # The reported browser failure: both clear Hindi first-person answers
    # left the role unclassified, so the reporter-role question repeated.
    # Each must set the self role and advance to the next required question
    # — the role question is never asked a second time.
    engine, workflow = _start_at_identity(f"AE-identity-hi-reported-{phrase[:6]}")

    result = workflow.process_turn(phrase)

    assert engine.case.reporter_role == "self", phrase
    assert result.next_question is not None, phrase
    assert result.next_question[0] == "patient_identifier", phrase


def test_hindi_self_answer_then_other_person_answer_stays_other():
    # Precedence is preserved: a mixed answer naming another person stays
    # someone_else even though self cues appear earlier in the sentence.
    mixed = "यह मेरे साथ नहीं, मेरी पत्नी को हुआ"
    result = AnswerLLMProvider().interpret(
        mixed, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields.get("reporter_role") == "someone_else"
