"""Regression tests for multi-turn AE answer extraction.

Focus: the production answer path that turned
"I am the patient reporting this. I am thirty four years old"
into reporter_role="self" and patient_age="34" while preserving the patient
identifier gating and the existing API intent contract ("unknown").
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
from types import SimpleNamespace
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.providers.answer import (
    AnswerLLMProvider,
    _extract_age,
    _normalize_product,
    normalized_dob_in,
    parse_word_number,
)
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine, WorkflowState
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    VOICE_CONFIRMATION_FIELDS,
)

EXACT_TRANSCRIPT = "I am the patient reporting this. I am thirty four years old"
INITIAL_REPORT = (
    "I took Lantus yesterday and since this morning I have a "
    "severe headache and vomiting."
)


# ---- word-number age parser ------------------------------------------------


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("thirty four", 34),
        ("forty five", 45),
        ("seventeen", 17),
        ("one hundred", 100),
        ("one hundred and five", 105),
        ("two thousand", 2000),
        ("two thousand and five", 2005),
        ("hundred", None),
        ("am thirty four", None),
    ],
)
def test_parse_word_number(phrase, expected):
    assert parse_word_number(phrase) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I am thirty four years old", "34"),
        ("I am 34 years old", "34"),
        ("I am forty five years old", "45"),
        ("I am one hundred years old", "100"),
        ("I am 34 years of age", "34"),
        ("Age is forty five", "45"),
        ("My age is 34", "34"),
        ("I am two hundred years old", None),
        ("I am 200 years old", None),
        ("I am 0 years old", None),
        ("My number is 9886094321", None),
        ("Call me at 9886094321 I am thirty four years old", "34"),
    ],
)
def test_extract_age(text, expected):
    assert _extract_age(text) == expected


# ---- provider level --------------------------------------------------------


def test_self_reporter_and_word_number_age_are_extracted():
    provider = AnswerLLMProvider()
    result = provider.interpret(
        EXACT_TRANSCRIPT,
        {"case_state": "EVENT_IDENTIFIED", "next_field": ["patient_identifier", "q"]},
    )

    assert result.intent == "unknown"
    assert result.extracted_fields["reporter_role"] == "self"
    assert result.extracted_fields["patient_age"] == "34"


def test_numeric_age_is_extracted():
    result = AnswerLLMProvider().interpret(
        "I am 34 years old",
        {"next_field": ["patient_identifier", "q"]},
    )
    assert result.extracted_fields["patient_age"] == "34"


def test_implausible_age_not_extracted():
    result = AnswerLLMProvider().interpret(
        "I am two hundred years old",
        {"next_field": ["patient_identifier", "q"]},
    )
    assert "patient_age" not in result.extracted_fields


def test_phone_number_not_misread_as_age():
    result = AnswerLLMProvider().interpret(
        "My number is 9886094321",
        {"next_field": ["patient_identifier", "q"]},
    )
    assert "patient_age" not in result.extracted_fields


@pytest.mark.parametrize(
    "text",
    [
        "I am the patient reporting this. I am thirty four years old",
        "I'm the patient",
        "I am reporting as the patient",
    ],
)
def test_self_reporter_language_variants(text):
    result = AnswerLLMProvider().interpret(text, {})
    assert result.extracted_fields["reporter_role"] == "self"


def test_someone_else_reporter_language():
    result = AnswerLLMProvider().interpret("My wife is reporting this", {})
    assert result.extracted_fields["reporter_role"] == "someone_else"


# ---- reporter-role colloquial self-answers ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "yes, it's me",
        "Yes, it's me",
        "it's me",
        "that's me",
        "my own experience",
        "my own side effect",
        "this happened to me",
        "I am the patient",
    ],
)
def test_colloquial_self_reporter_answers(text):
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "self"


@pytest.mark.parametrize(
    "text",
    [
        "no, it's my wife",
        "no, it was my son",
        "it's not me, it's my daughter",
    ],
)
def test_someone_else_still_wins_over_self_cue_words(text):
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["reporter_role", "Are you the patient?"]}
    )
    assert result.extracted_fields["reporter_role"] == "someone_else"


def test_workflow_advances_on_colloquial_self_reporter_answer():
    engine = AEStateEngine.create_case("AE-role-colloquial")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    first = workflow.process_turn(INITIAL_REPORT)
    # The documented Step 4 medication verification precedes Step 5.
    assert first.next_question[0] == "product_verification"
    workflow.process_turn("yes")

    second = workflow.process_turn("Yes, it's me")
    assert engine.case.reporter_role == "self"
    # The conversation moves on instead of repeating the same question.
    assert second.next_question is not None
    assert second.next_question[0] == "patient_identifier"


def test_workflow_advances_on_my_own_experience_answer():
    engine = AEStateEngine.create_case("AE-role-my-own")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn(INITIAL_REPORT)
    workflow.process_turn("yes")  # Step 4: medication verified
    second = workflow.process_turn("My own experience")

    assert engine.case.reporter_role == "self"
    assert second.next_question is not None
    assert second.next_question[0] == "patient_identifier"


# ---- workflow level --------------------------------------------------------


def test_workflow_captures_age_and_role_keeping_identifier_gating():
    engine = AEStateEngine.create_case("AE-demo-001")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    first = workflow.process_turn(INITIAL_REPORT)
    assert first.intent == "ae_report"
    # Event and medication are both captured; the approved order reports the
    # furthest satisfied state (EVENT_IDENTIFIED, then PRODUCT_IDENTIFIED).
    assert engine.current_state is WorkflowState.PRODUCT_IDENTIFIED
    workflow.process_turn("yes")  # Step 4: medication verified

    second = workflow.process_turn(EXACT_TRANSCRIPT)

    assert engine.case.reporter_role == "self"
    assert engine.case.patient_age == "34"
    # Reporter identity for a self-reporter stays gated on patient_identifier.
    assert engine.current_state is WorkflowState.PRODUCT_IDENTIFIED
    assert second.next_question is not None
    assert second.next_question[0] == "patient_identifier"
    assert second.intent == "unknown"


def test_self_role_alone_does_not_reach_reporter_identified():
    engine = AEStateEngine.create_case("AE-demo-002")
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Headache (demo)")
    engine.update("patient_age", "34")
    engine.update("reporter_role", "self")

    # Product and event are captured; the approved order advances past
    # EVENT_IDENTIFIED to PRODUCT_IDENTIFIED, and a self-reporter is not
    # fully identified until the patient identifier is captured.
    assert engine.current_state is WorkflowState.PRODUCT_IDENTIFIED
    assert engine.next_field() is not None
    assert engine.next_field()[0] == "patient_identifier"

    engine.update("patient_identifier", "Demo Patient 001")
    assert engine.current_state is WorkflowState.PATIENT_IDENTIFIED


# ---- endpoint level (offline) ---------------------------------------------


def create_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 100)


class FakeSTT:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    def transcribe(self, audio_path, language_code="en-IN"):
        if not self._transcripts:
            return SimpleNamespace(text="")
        return SimpleNamespace(text=self._transcripts.pop(0))


class FakeTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _post(client, session_id, tmp):
    wav_path = Path(tmp) / "audio.wav"
    create_test_wav(wav_path)
    with open(wav_path, "rb") as f:
        return client.post(
            "/voice/conversation",
            files={"audio_file": ("audio.wav", f, "audio/wav")},
            data={"session_id": session_id},
        )


def test_multi_turn_endpoint_captures_age_and_role():
    stt = FakeSTT([INITIAL_REPORT, "yes", EXACT_TRANSCRIPT])
    tts = FakeTTS()
    session_store = SessionStore()
    app = FastAPI()
    app.include_router(
        create_voice_conversation_router(
            stt=stt, tts=tts, session_store=session_store
        )
    )
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp:
        first = _post(client, "ae-demo-001", tmp)
        _post(client, "ae-demo-001", tmp)  # Step 4: medication verified
        second = _post(client, "ae-demo-001", tmp)

    assert first.status_code == 200
    assert first.json()["workflow"] == "ae"
    assert first.json()["intent"] == "ae_report"

    body = second.json()
    assert body["workflow"] == "ae"
    assert body["intent"] == "unknown"
    assert body["current_state"] == WorkflowState.PRODUCT_IDENTIFIED.value
    assert body["completed"] is False
    assert body["next_question"][0] == "patient_identifier"
    assert body["ae"]["extracted_information"]["reporter_role"] == "self"
    assert body["ae"]["extracted_information"]["patient_age"] == "34"
    collected = body["ae"]["collected_information"]
    assert collected["reporter_role"] == "self"
    assert collected["patient_age"] == "34"


# ---- real entry point: app.main.create_app (uvicorn app.main:app) ----------

REAL_AUDIO_1 = (
    "i took lantus yesterday and since this morning i have a severe headache and vomiting"
)
REAL_AUDIO_2 = "i am the patient reporting this i am thirty four years old"
REAL_SESSION_ID = "AE-DEMO-001"


def test_real_entry_exact_reproduction_ae_demo_001():
    """Exact real /voice/conversation reproduction, Audio 1 -> Audio 2.

    Runs through app.main.create_app (the factory behind ``uvicorn
    app.main:app``) with the real Prisma-style lowercase transcripts and
    session id AE-DEMO-001. Audio 2 is answered after the documented Step 4
    medication verification, and asserts the endpoint applies it:
    patient_age=34, reporter_role=self, and the next question stays
    patient_identifier (identifier gating intact).
    """
    from app.main import create_app

    stt = FakeSTT([REAL_AUDIO_1, "yes", REAL_AUDIO_2])
    app = create_app(prisma=stt, timbre=FakeTTS())
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp:
        first = _post(client, REAL_SESSION_ID, tmp)
        verified = _post(client, REAL_SESSION_ID, tmp)
        second = _post(client, REAL_SESSION_ID, tmp)

    assert first.status_code == 200
    assert verified.status_code == 200
    assert second.status_code == 200
    one, checked, two = first.json(), verified.json(), second.json()

    # Audio 1: report identified, case open, medication verification asked
    # (the documented Step 4 precedes reporter identification).
    assert one["workflow"] == "ae"
    assert one["intent"] == "ae_report"
    assert one["current_state"] == WorkflowState.PRODUCT_IDENTIFIED.value
    assert one["next_question"][0] == "product_verification"

    # Step 4 confirmed: the recorded medication stands and reporter
    # identification (Step 5) is next.
    assert checked["ae"]["collected_information"]["product"] == "Lantus SoloStar"
    assert checked["next_question"][0] == "reporter_role"

    # Session and case continuity across the audios.
    assert two["session_id"] == REAL_SESSION_ID
    assert two["case_id"] == one["case_id"]

    # Intent contract unchanged for answer turns.
    assert two["workflow"] == "ae"
    assert two["intent"] == "unknown"

    # Audio 2 extraction applied by the real endpoint.
    extracted = two["ae"]["extracted_information"]
    collected = two["ae"]["collected_information"]
    assert extracted["reporter_role"] == "self"
    assert extracted["patient_age"] == "34"
    assert collected["reporter_role"] == "self"
    assert collected["patient_age"] == "34"
    assert "patient_age" not in two["missing_information"]
    assert "reporter_role" not in two["missing_information"]

    # No false progression: patient_identifier is still required.
    assert two["current_state"] == WorkflowState.PRODUCT_IDENTIFIED.value
    assert two["next_question"][0] == "patient_identifier"
    assert "patient_identifier" in two["missing_information"]
    assert two["completed"] is False


# ---- spoken digit phone extraction -----------------------------------------

SPOKEN_PHONE = "nine eight eight six zero nine four three two one"
CONTACT_PENDING = ["contact_phone", "What phone number would you like us to use?"]


def test_spoken_digit_phone_is_extracted():
    result = AnswerLLMProvider().interpret(SPOKEN_PHONE, {})
    assert result.intent == "unknown"
    assert result.extracted_fields["contact_phone"] == "9886094321"


def test_spoken_digit_phone_is_extracted_when_contact_is_pending():
    result = AnswerLLMProvider().interpret(SPOKEN_PHONE, {"next_field": CONTACT_PENDING})
    assert result.intent == "unknown"
    assert result.extracted_fields["contact_phone"] == "9886094321"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("9886094321", "9886094321"),
        ("+919886094321", "919886094321"),
        ("my number is 9886094321", "9886094321"),
    ],
)
def test_numeric_phone_is_extracted(text, expected):
    result = AnswerLLMProvider().interpret(text, {"next_field": CONTACT_PENDING})
    assert result.intent == "unknown"
    assert result.extracted_fields["contact_phone"] == expected


def test_age_text_is_not_extracted_as_phone():
    spoken_age = AnswerLLMProvider().interpret(
        "I am thirty four years old", {"next_field": CONTACT_PENDING}
    )
    assert spoken_age.extracted_fields.get("patient_age") == "34"
    assert "contact_phone" not in spoken_age.extracted_fields

    numeric_age = AnswerLLMProvider().interpret(
        "I am 34 years old", {"next_field": CONTACT_PENDING}
    )
    assert numeric_age.extracted_fields.get("patient_age") == "34"
    assert "contact_phone" not in numeric_age.extracted_fields

    for text in ("34", "thirty four"):
        result = AnswerLLMProvider().interpret(text, {"next_field": CONTACT_PENDING})
        assert "contact_phone" not in result.extracted_fields, text


# ---- spoken Hindi phone digits -----------------------------------------------

HINDI_SPOKEN_PHONE = "नाइन टू टू फोर फोर वन वन वन"
HINDI_DIGIT_WORDS_PHONE = "दो दो चार चार एक एक एक"


def test_devanagari_transliterated_digit_words_extract_phone():
    result = AnswerLLMProvider().interpret(HINDI_SPOKEN_PHONE, {})
    assert result.extracted_fields["contact_phone"] == "92244111"

    pending = AnswerLLMProvider().interpret(
        HINDI_SPOKEN_PHONE, {"next_field": CONTACT_PENDING}
    )
    assert pending.extracted_fields["contact_phone"] == "92244111"


def test_hindi_number_words_extract_phone():
    result = AnswerLLMProvider().interpret(
        HINDI_DIGIT_WORDS_PHONE, {"next_field": CONTACT_PENDING}
    )
    assert result.extracted_fields["contact_phone"] == "2244111"


def test_hindi_spoken_phone_with_punctuation_and_spacing():
    result = AnswerLLMProvider().interpret(
        "नाइन, टू! टू? फोर, फोर। वन वन वन", {"next_field": CONTACT_PENDING}
    )
    assert result.extracted_fields["contact_phone"] == "92244111"


def test_mixed_hindi_english_digit_words_extract_phone():
    result = AnswerLLMProvider().interpret(
        "nine नाइन eight ऐट seven सेवन six सिक्स five",
        {"next_field": CONTACT_PENDING},
    )
    assert result.extracted_fields["contact_phone"] == "998877665"


@pytest.mark.parametrize(
    "text",
    [
        "पच्चीस साल",
        "दो दिन से पेट दर्द",
        "मुझे कल से सिर में दर्द है",
        "नाइन टू टू फोर",  # fewer than seven digits
    ],
)
def test_hindi_non_phone_text_stores_no_phone(text):
    fields = AnswerLLMProvider().interpret(
        text, {"next_field": CONTACT_PENDING}
    ).extracted_fields
    assert "contact_phone" not in fields


def test_hindi_spoken_phone_completes_contact_capture():
    engine = AEStateEngine.create_case("AE-phone-hi")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    assert engine.next_field()[0] == "contact_phone"
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("नाइन टू टू फोर फोर वन वन वन")

    assert engine.case.contact_phone == "92244111"
    assert result.next_question is not None
    # Consent was already captured; the approved order asks severity next
    # (severity assessment before the incident date).
    assert result.next_question[0] == "seriousness"


def test_unclarifiable_phone_parks_report_for_human_review():
    engine = AEStateEngine.create_case("AE-phone-loop")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    # Unparseable replies: one targeted clarification, then the report is
    # parked (bounded) instead of repeating the digit-by-digit request forever.
    workflow.process_turn("hello")
    clarified = workflow.process_turn("hello")
    assert "digit by digit" in clarified.response_text
    parked = workflow.process_turn("hello")

    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.human_review_required is True
    assert parked.completed is False
    assert engine.case.contact_phone == "unknown"
    # Previously captured report data is preserved.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.consent_to_contact is True


# One synthetic spoken-digit sequence per additional locale (zero..seven,
# using each locale's reviewed digit words), mirroring the Hindi fixtures
# above. No real contact information.
_LOCALE_SPOKEN_PHONES = {
    "ml-IN": ("പൂജ്യം ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ്", "01234567"),
    "bn-IN": ("শূন্য এক দুই তিন চার পাঁচ ছয় সাত", "01234567"),
    "gu-IN": ("શૂન્ય એક બે ત્રણ ચાર પાંચ છ સાત", "01234567"),
    "kn-IN": ("ಶೂನ್ಯ ಒಂದು ಎರಡು ಮೂರು ನಾಲು ಐದು ಆರು ಏಳು", "01234567"),
    "mr-IN": ("शून्य एक दोन तीन चार पाच सहा सात", "01234567"),
    "pa-IN": ("ਸਿਫ਼ਰ ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ", "01234567"),
    "ta-IN": ("பூஜ்ஜியம் ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு", "01234567"),
    "te-IN": ("సున్నా ఒకటి రెండు మూడు నాలుగు ఐదు ఆరు ఏడు", "01234567"),
}


@pytest.mark.parametrize(
    ("locale", "spoken", "expected"),
    [(loc, *pair) for loc, pair in sorted(_LOCALE_SPOKEN_PHONES.items())],
    ids=str,
)
def test_spoken_digit_words_extract_phone_in_every_locale(locale, spoken, expected):
    result = AnswerLLMProvider().interpret(spoken, {"next_field": CONTACT_PENDING})
    assert result.extracted_fields["contact_phone"] == expected, spoken


@pytest.mark.parametrize(
    ("locale", "spoken", "expected"),
    [(loc, *pair) for loc, pair in sorted(_LOCALE_SPOKEN_PHONES.items())],
    ids=str,
)
def test_locale_spoken_phone_tolerates_punctuation_and_spacing(
    locale, spoken, expected
):
    punctuated = ", ".join(spoken.split()) + "।"
    result = AnswerLLMProvider().interpret(punctuated, {"next_field": CONTACT_PENDING})
    assert result.extracted_fields["contact_phone"] == expected, punctuated


def test_alternate_zero_spellings_map_to_zero():
    from app.providers.answer import _ALL_LOCALE_DIGIT_WORDS

    for word in ("ਸਿਫ਼ਰ", "ਸਿਫਰ", "பூஜ்ஜியம்", "பூஜ்யம்", "సున్నా", "శూన్యం"):
        assert _ALL_LOCALE_DIGIT_WORDS.get(word) == "0", word


@pytest.mark.parametrize(
    "text",
    [
        "ഇരുപത്",                      # an age word, never a phone digit
        "ഒന്ന് രണ്ട് മൂന്ന്",              # fewer than seven digits
        "ഇരുപത് മേയ് രണ്ടായിരത്തി ആറ്",    # a spoken date of birth
        "ഇന്ന് തലവേദന തുടങ്ങി",          # unrelated speech
    ],
)
def test_malayalam_non_phone_text_stores_no_phone(text):
    fields = AnswerLLMProvider().interpret(
        text, {"next_field": CONTACT_PENDING}
    ).extracted_fields
    assert "contact_phone" not in fields


MALAYALAM_TRANSLITERATED_PHONE = (
    "നൈൻ ടു ടു ഫോർ ഫോർ സീറോ സീറോ സീറോ സീറോ"
)


def test_malayalam_transliterated_english_digits_extract_phone():
    # The observed browser repro: English digits spoken by a Malayalam
    # speaker, transcribed as Malayalam-script letter names.
    result = AnswerLLMProvider().interpret(
        MALAYALAM_TRANSLITERATED_PHONE, {"next_field": CONTACT_PENDING}
    )
    # Nine spoken digits: the established minimum is 7, validation unchanged.
    assert result.extracted_fields["contact_phone"] == "922440000"


def test_malayalam_phone_with_vaan_three_digits_extracts_10_digits():
    # The observed browser transcript: "nine two two four four one nine one
    # zero three" spoken with the letter names വൺ (1) and ത്രീ (3), verified
    # positionally against the reporter's expected 9224419103.
    result = AnswerLLMProvider().interpret(
        "നൈൻ ടു ടു ഫോർ ഫോർ വൺ നൈൻ വൺ സീറോ ത്രീ",
        {"next_field": CONTACT_PENDING},
    )
    assert result.extracted_fields["contact_phone"] == "9224419103"


def test_malayalam_transliterated_phone_completes_contact_capture():
    engine = AEStateEngine.create_case("AE-phone-ml-translit")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    assert engine.next_field()[0] == "contact_phone"
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(MALAYALAM_TRANSLITERATED_PHONE)

    assert engine.case.contact_phone == "922440000"
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


def test_malayalam_phone_is_read_back_and_only_committed_explicitly():
    # Extract -> read back -> explicit confirmation -> commit: an unconfirmed
    # number is never written to the case.
    from app.workflow.service.conversational import VOICE_CONFIRMATION_FIELDS

    engine = AEStateEngine.create_case("AE-phone-ml-confirm")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    assert engine.next_field()[0] == "contact_phone"

    spoken = workflow.process_turn(
        "നൈൻ ടു ടു ഫോർ ഫോർ വൺ നൈൻ വൺ സീറോ ത്രീ"
    )
    assert engine.case.contact_phone is None  # held for read-back
    # The established phone read-back speaks only the last four digits.
    assert "9103" in spoken.response_text

    committed = workflow.process_turn("yes")

    assert engine.case.contact_phone == "9224419103"
    assert committed.next_question is not None
    assert committed.next_question[0] == "seriousness"


def test_malayalam_onpathu_chillu_spelling_extracts_digits_and_dob():
    # The browser STT emitted the chillu spelling ഒൻപത് of nine (both
    # spellings are standard Malayalam); it was unmapped, breaking the digit
    # run in both phone answers of the observed two-turn failure.
    from app.providers.answer import _extract_spoken_phone

    t1 = "ഒൻപത് രണ്ട് രണ്ട് നാല് നാല് ഒന്ന് ഒൻപത് ഒന്ന് പൂജ്യം"
    t2 = t1 + " മൂന്ന്"
    # The first answer is genuinely nine digits and extracts as such; the
    # second completes to ten.
    assert _extract_spoken_phone(t1) == "922441910"
    assert _extract_spoken_phone(t2) == "9224419103"
    # The same spelling serves a spoken date of birth.
    assert normalized_dob_in("ഒൻപത് മേയ് രണ്ടായിരം") == "09-May-2000"


def test_two_turn_phone_repro_updates_candidate_and_confirm_commits():
    # The exact observed two-turn sequence: the incomplete first answer
    # extracts a shorter candidate and is read back; the corrected full
    # spoken restatement updates the pending candidate (never committed
    # without an explicit yes); confirming commits the 10-digit number.
    from app.workflow.service.conversational import VOICE_CONFIRMATION_FIELDS

    engine = AEStateEngine.create_case("AE-phone-ml-2turn")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju PK")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    assert engine.next_field()[0] == "contact_phone"

    first = workflow.process_turn(
        "ഒൻപത് രണ്ട് രണ്ട് നാല് നാല് ഒന്ന് ഒൻപത് ഒന്ന് പൂജ്യം"
    )
    assert engine.case.contact_phone is None
    assert "1910" in first.response_text  # read back: last four digits only

    corrected = workflow.process_turn(
        "ഒൻപത് രണ്ട് രണ്ട് നാല് നാല് ഒന്ന് ഒൻപത് ഒന്ന് പൂജ്യം മൂന്ന്"
    )
    assert engine.case.contact_phone is None  # still unconfirmed
    assert "9103" in corrected.response_text  # the NEW candidate read back

    committed = workflow.process_turn("yes")

    assert engine.case.contact_phone == "9224419103"
    assert committed.next_question is not None
    assert committed.next_question[0] == "seriousness"


@pytest.mark.parametrize(
    ("candidate", "expected_count", "expected_tail"),
    [
        ("9224419103", "10-digit", "9103"),
        # The observed doubled-leading-nine artifact is now audible at
        # confirmation time.
        ("99224419103", "11-digit", "9103"),
        ("922441910", "9-digit", "1910"),
        ("9224419", "7-digit", "4419"),
    ],
)
def test_phone_readback_speaks_the_digit_count_not_the_number(
    candidate, expected_count, expected_tail
):
    # The privacy policy is unchanged: the number itself is never read
    # aloud; only its digit count and the last four digits are spoken.
    engine = AEStateEngine.create_case("AE-phone-count")
    engine.update("product", "Lantus SoloStar")
    engine.update("contact_phone", candidate)
    engine.case.pending_confirmation = "contact_phone"

    question = engine.next_field()
    assert question is not None and question[0] == "confirmation"
    text = question[1]
    assert f"I have a {expected_count} number ending in {expected_tail}" in text
    # The full number is never spoken (the count and tail only).
    assert candidate not in text or len(candidate) <= 4


def test_unverified_malayalam_transliterations_stay_unmapped():
    # Documented gap: വൺ/ത്രീ/ഫൈവ്/സിക്സ്/സെവൻ are not verified; runs
    # containing them must not become a guessed phone number.
    from app.providers.answer import _extract_spoken_phone

    assert (
        _extract_spoken_phone("വൺ ത്രീ ഫൈവ് സിക്സ് സെവൻ നൈൻ ടു ഫോർ")
        is None
    )


def test_malayalam_spoken_phone_completes_contact_capture():
    engine = AEStateEngine.create_case("AE-phone-ml")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    assert engine.next_field()[0] == "contact_phone"
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("പൂജ്യം ഒന്ന് രണ്ട് മൂന്ന് നാല് അഞ്ച് ആറ് ഏഴ്")

    assert engine.case.contact_phone == "01234567"
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


# ---- original description preservation -------------------------------------


def test_fresh_ae_report_keeps_the_complete_original_sentence():
    result = AnswerLLMProvider().interpret(INITIAL_REPORT, {})
    assert result.intent == "ae_report"
    # The reporter's full wording is the description; the base provider's
    # canned short event must never replace it.
    assert result.extracted_fields["adverse_event"] == INITIAL_REPORT


def test_workflow_preserves_original_description_and_deduplicates_additions():
    engine = AEStateEngine.create_case("AE-preserve-001")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn(INITIAL_REPORT)
    original = engine.case.adverse_event
    assert original == INITIAL_REPORT
    workflow.process_turn("yes")  # Step 4: medication verified

    # A symptom already contained in the original sentence is not duplicated.
    workflow.process_turn("severe headache")
    assert engine.case.adverse_event == original

    # A genuinely new symptom is appended once; the original wording stays.
    workflow.process_turn("also nausea")
    assert engine.case.adverse_event == f"{original}; also nausea"


def test_later_ae_classified_turn_does_not_replace_original_description():
    engine = AEStateEngine.create_case("AE-preserve-002")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn(INITIAL_REPORT)
    original = engine.case.adverse_event
    product = engine.case.product

    # The base provider classifies this as ae_report with a canned event; it
    # must not overwrite what the reporter already described.
    workflow.process_turn("The headache is worse after taking lantus")

    assert engine.case.adverse_event == original
    assert engine.case.product == product


# ---- medical-care downgrade protection -------------------------------------


def _workflow_with_medical_care_yes(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn(INITIAL_REPORT)
    workflow.process_turn("yes")  # Step 4: medication verified
    engine.update("medical_care", "yes")  # hospitalization already reported
    return engine, workflow


def test_medical_care_yes_is_not_downgraded_without_explicit_correction():
    engine, workflow = _workflow_with_medical_care_yes("AE-medical-001")
    result = workflow.process_turn("I didn't go to the hospital")
    assert engine.case.medical_care == "yes"
    assert "medical_care" not in result.extracted_information


def test_explicit_correction_downgrades_medical_care():
    engine, workflow = _workflow_with_medical_care_yes("AE-medical-002")
    workflow.process_turn("Actually, I didn't go to the hospital")
    assert engine.case.medical_care == "no"


def test_medical_care_unknown_is_not_recorded_over_existing_yes():
    result = AnswerLLMProvider().interpret(
        "I don't remember",
        {
            "collected": {
                "product": "Lantus SoloStar",
                "adverse_event": "rash",
                "medical_care": "yes",
            },
            "next_field": ["medical_care", "Did you get any medical care?"],
        },
    )
    assert "medical_care" not in result.extracted_fields


def test_medical_care_unknown_is_recorded_without_an_existing_yes():
    result = AnswerLLMProvider().interpret(
        "I don't remember",
        {"next_field": ["medical_care", "Did you get any medical care?"]},
    )
    assert result.extracted_fields["medical_care"] == "unknown"


# ---- spoken email normalization --------------------------------------------

_EMAIL_PENDING = [
    "contact_email",
    "What email address would you like us to use?",
]


def _pending_email(text: str) -> dict:
    """extracted_fields when the workflow is waiting for an email address."""
    return AnswerLLMProvider().interpret(
        text, {"next_field": _EMAIL_PENDING}
    ).extracted_fields


def test_canonical_email_is_extracted_in_both_paths():
    # Pending-field path.
    assert _pending_email("john.doe@example.com")["contact_email"] == (
        "john.doe@example.com"
    )
    # General answer path (no pending question).
    general = AnswerLLMProvider().interpret(
        "my email is john.doe@example.com", {}
    )
    assert general.extracted_fields["contact_email"] == "john.doe@example.com"


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("john dot doe at gmail dot com", "john.doe@gmail.com"),
        ("john at the rate gmail dot com", "john@gmail.com"),
        ("john.doe at gmail dot com", "john.doe@gmail.com"),
        ("john at gmail dot com", "john@gmail.com"),
        ("my email is john at gmail dot com", "john@gmail.com"),
        ("reach me at john at gmail dot com", "john@gmail.com"),
    ],
)
def test_spoken_email_at_and_dot_variants(spoken, expected):
    assert _pending_email(spoken)["contact_email"] == expected


def test_spaced_email_components_are_normalized_in_both_paths():
    spaced = "john . doe @ gmail . com"
    assert _pending_email(spaced)["contact_email"] == "john.doe@gmail.com"
    general = AnswerLLMProvider().interpret(spaced, {})
    assert general.extracted_fields["contact_email"] == "john.doe@gmail.com"


@pytest.mark.parametrize(
    "text",
    [
        "john at gmail",  # missing domain dot / TLD
        "john at",  # missing domain entirely
        "sreehari s nair at gmail dot com",  # spaces inside a component
        "call me at 9886094321",  # not an email address
        "a at b dot com or c at d dot com",  # ambiguous: two addresses
    ],
)
def test_invalid_or_ambiguous_spoken_email_is_not_stored(text):
    assert "contact_email" not in _pending_email(text)
    general = AnswerLLMProvider().interpret(text, {})
    assert "contact_email" not in general.extracted_fields


# ---- DOB / age answers -------------------------------------------------------

_TODAY_YEAR = date.today().year


def _pending_age_fields(text: str) -> dict:
    """extracted_fields while the workflow is waiting for age or DOB."""
    return AnswerLLMProvider().interpret(
        text,
        {
            "next_field": [
                "patient_age",
                "How old are you, or what is your date of birth?",
            ]
        },
    ).extracted_fields


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I was born on 12 March 1980", "12-Mar-1980"),
        ("my date of birth is 05/01/1980", "05-Jan-1980"),
        ("1980-03-12", "12-Mar-1980"),
        ("I was born in 1980", "born in 1980"),  # year only: never invented
        ("my dob is 12 Mar 1980", "12-Mar-1980"),
    ],
)
def test_supported_dob_formats_are_normalized(text, expected):
    assert _pending_age_fields(text)["patient_age"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "my date of birth is 12 March",  # missing year
        "my date of birth is 12/1980",  # incomplete date components
        "I was born in 20",  # incomplete year
        "my date of birth is 05/01/20",  # two-digit year
    ],
)
def test_incomplete_dob_is_not_stored(text):
    fields = _pending_age_fields(text)
    assert "patient_age" not in fields
    assert "dob_age_conflict" not in fields


def test_dob_only_answer_is_normalized_without_a_derived_age():
    fields = _pending_age_fields("my date of birth is 12 March 1980")
    assert fields["patient_age"] == "12-Mar-1980"
    # The stored value is the DOB itself, never an invented age in years.
    assert not fields["patient_age"].isdigit()


def test_consistent_dob_and_age_do_not_trigger_a_clarification():
    year = _TODAY_YEAR - 45  # born 45 years ago: agrees with "45 years old"
    fields = _pending_age_fields(f"I am 45 years old and I was born in {year}")
    assert "dob_age_conflict" not in fields
    # The complete DOB is preserved verbatim; no age is derived from it.
    assert fields["patient_age"] == f"born in {year}"


def test_conflicting_dob_and_age_hold_both_instead_of_choosing():
    year = _TODAY_YEAR - 20  # born 20 years ago disagrees with "45 years old"
    fields = _pending_age_fields(f"I am 45 years old and I was born in {year}")
    # Neither value is stored: both are held for one clarification.
    assert "patient_age" not in fields
    assert fields["dob_age_conflict"] == {"dob": f"born in {year}", "age": "45"}


def test_workflow_asks_one_clarification_for_conflicting_dob_and_age():
    engine = AEStateEngine.create_case("AE-dob-age-demo")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    assert engine.next_field()[0] == "patient_age"

    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    year = _TODAY_YEAR - 20
    result = workflow.process_turn(f"I am 45 years old and I was born in {year}")

    # Neither value was silently chosen; both are held for clarification.
    assert engine.case.patient_age is None
    assert workflow._dob_age_conflict == {"dob": f"born in {year}", "age": "45"}
    # The marker is internal: it never reaches the response envelope.
    assert "dob_age_conflict" not in result.extracted_information

    # Exactly one targeted clarification replaces the generic age question.
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    prompt = result.next_question[1]
    assert f"born in {year}" in prompt
    assert "45 years" in prompt
    assert prompt in result.response_text
    assert "How old are you, or what is your date of birth?" not in result.response_text
    # A plain clarification: not a review escalation and not completion.
    assert result.human_review_required is False
    assert result.completed is False

    # The next age answer resolves it; the held conflict is cleared.
    resolved = workflow.process_turn("I am 45 years old")
    assert engine.case.patient_age == "45"
    assert workflow._dob_age_conflict is None
    assert resolved.next_question is not None
    assert resolved.next_question[0] != "patient_age"


def test_dob_age_clarification_template_translates_to_hindi():
    from app.language import translate
    from app.workflow.service.conversational import _dob_age_clarification

    prompt = _dob_age_clarification("12 March 1980", "20")
    result = translate(prompt, "hi-IN")

    # All-or-nothing translation must succeed: both sentences are covered.
    assert result.translated is True
    assert result.language == "hi-IN"
    assert "जन्मतिथि" in result.text
    assert "Your date of birth" not in result.text


# ---- Hindi affirmation / negation recognition --------------------------------


def _pending_reply(field_key: str, text: str) -> dict:
    """extracted_fields while the workflow is waiting on one yes/no field."""
    return AnswerLLMProvider().interpret(
        text, {"next_field": [field_key, "q"]}
    ).extracted_fields


@pytest.mark.parametrize(
    "text",
    ["haan", "han", "ji haan", "हाँ", "हां", "जी हाँ", "haan, yes", "yes, haan"],
)
def test_hindi_affirmations_answer_yes(text):
    assert _pending_reply("consent_to_contact", text).get("consent_to_contact") is True


@pytest.mark.parametrize(
    "text",
    ["nahi", "nahin", "नहीं", "नही", "no, nahi", "nahi, no", "नहीं, thanks"],
)
def test_hindi_negations_answer_no(text):
    assert _pending_reply("consent_to_contact", text).get("consent_to_contact") is False


def test_mixed_english_hindi_replies():
    assert (
        _pending_reply("consent_to_contact", "haan, theek hai").get(
            "consent_to_contact"
        )
        is True
    )
    assert (
        _pending_reply("consent_to_contact", "nahi, I do not consent").get(
            "consent_to_contact"
        )
        is False
    )


def test_negation_precedence_for_consent():
    # A clear (leading) negation never reads as consent, even when a later
    # word could affirm...
    assert (
        _pending_reply("consent_to_contact", "nahi, haan").get("consent_to_contact")
        is False
    )
    # ...while an affirmation-led English idiom stays a yes.
    assert (
        _pending_reply("consent_to_contact", "yes, no problem").get(
            "consent_to_contact"
        )
        is True
    )


@pytest.mark.parametrize(
    "text,expected",
    [("haan", "yes"), ("जी हाँ", "yes"), ("nahi", "no"), ("नहीं", "no")],
)
def test_hindi_seriousness_replies(text, expected):
    assert _pending_reply("seriousness", text).get("seriousness") == expected


def test_english_yes_no_paths_are_unchanged():
    assert _pending_reply("consent_to_contact", "yes").get("consent_to_contact") is True
    assert _pending_reply("consent_to_contact", "no").get("consent_to_contact") is False
    assert _pending_reply("seriousness", "not really").get("seriousness") == "no"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("haan", "yes"),
        ("han", "yes"),
        ("ji haan", "yes"),
        ("हाँ", "yes"),
        ("हां", "yes"),
        ("जी हाँ", "yes"),
        ("yes, haan", "yes"),
        ("nahi", "no"),
        ("nahin", "no"),
        ("नहीं", "no"),
        ("नही", "no"),
        ("no, nahi", "no"),
        ("nahi, haan", "ambiguous"),
    ],
)
def test_hindi_confirmation_replies(text, expected):
    from app.workflow.service.conversational import _classify_confirmation_turn

    verdict, _ = _classify_confirmation_turn(text)
    assert verdict == expected


# ---- Hindi / Hinglish field answers -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "mere saath hua",
        "mere saath hi hua",
        "मेरे साथ हुआ",
        "mujhe hua",
        "मुझे हुआ",
        "main khud",
        "मैं खुद",
        "mera apna experience",
        "मेरा अपना अनुभव",
        "main hoon",
        "mai hu",
        "मैं हूँ",
        "मै हूं",
        "yes, mere saath hua",
    ],
)
def test_hindi_self_reporter_answers(text):
    assert _pending_reply("reporter_role", text).get("reporter_role") == "self"


@pytest.mark.parametrize(
    "text",
    [
        "meri patni",
        "mere pati",
        "मेरी पत्नी",
        "मेरा बेटा",
        "mera beta",
        "meri wife",
        "mere dost",
        "मेरे दोस्त",
        "doosre ke liye",
        "दूसरे के लिए",
        "koi aur",
        "कोई और",
        "no, meri wife",
    ],
)
def test_hindi_someone_else_reporter_answers(text):
    assert (
        _pending_reply("reporter_role", text).get("reporter_role")
        == "someone_else"
    )


# ---- first-person reporter answers (retry-bug regressions) --------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm the one who's experiencing this problem",
        "I am the one having this problem",
        "I'm the one who is facing this",
        "I am the one going through this",
    ],
)
def test_the_one_experiencing_self_answers(text):
    assert _pending_reply("reporter_role", text).get("reporter_role") == "self"


def test_english_self_answers_keep_working():
    # Previously-working natural self answers (listed reproduction included).
    assert (
        _pending_reply("reporter_role", "I am reporting my own experience").get(
            "reporter_role"
        )
        == "self"
    )
    assert (
        _pending_reply("reporter_role", "My own experience").get("reporter_role")
        == "self"
    )


@pytest.mark.parametrize(
    "text",
    [
        "नहीं मैं ही पेशेंट हूँ।",
        "main hi patient hoon",
        "मैं ही हूँ",
        "main hi hoon",
        "मेरा ही अनुभव है ये",
        "मेरा ही अनुभव है",
        "mera hi anubhav hai ye",
        "अपना ही अनुभव है ये",
        "apna hi anubhav hai ye",
    ],
)
def test_hindi_hi_family_self_answers(text):
    assert _pending_reply("reporter_role", text).get("reporter_role") == "self"


@pytest.mark.parametrize(
    "text",
    [
        "I'm reporting for my wife",
        "I'm the one reporting for my wife",
        "मेरी पत्नी के लिए",
    ],
)
def test_other_reporter_answers_still_win_over_self_phrases(text):
    assert (
        _pending_reply("reporter_role", text).get("reporter_role")
        == "someone_else"
    )


def test_the_one_experiencing_answer_advances_workflow():
    engine = AEStateEngine.create_case("AE-role-the-one")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("I'm the one who's experiencing this problem")

    assert engine.case.reporter_role == "self"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier"
    # Collected report data is untouched.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == "headache"


def test_hindi_hi_answer_advances_workflow_and_preserves_data():
    engine = AEStateEngine.create_case("AE-role-hindi-hi")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("नहीं मैं ही पेशेंट हूँ।")

    assert engine.case.reporter_role == "self"
    assert result.next_question is not None
    assert result.next_question[0] == "patient_identifier"
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == "headache"


@pytest.mark.parametrize(
    "text",
    ["haan", "nahi", "shayad", "pata nahi", "theek hai", "dobara kahiye"],
)
def test_ambiguous_hindi_role_replies_store_nothing(text):
    assert "reporter_role" not in _pending_reply("reporter_role", text)


@pytest.mark.parametrize(
    "text",
    [
        "theek hai",
        "thik hai",
        "ठीक है",
        "sahmat",
        "सहमत",
        "bilkul",
        "बिल्कुल",
        "bilkul, theek hai",
        "yes, sahmat",
    ],
)
def test_more_hindi_affirmations_answer_yes(text):
    assert _pending_reply("consent_to_contact", text).get("consent_to_contact") is True


@pytest.mark.parametrize(
    "text",
    ["sahmat nahi", "सहमत नहीं", "bilkul nahi", "बिल्कुल नहीं"],
)
def test_hindi_consent_trailing_negations_answer_no(text):
    assert _pending_reply("consent_to_contact", text).get("consent_to_contact") is False


def test_ambiguous_hindi_consent_reply_is_not_stored():
    assert "consent_to_contact" not in _pending_reply("consent_to_contact", "shayad")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("gambhir hai", "yes"),
        ("गंभीर है", "yes"),
        ("gambhir nahi hai", "no"),
        ("गंभीर नहीं है", "no"),
        ("shayad", "unknown"),
        ("शायद", "unknown"),
        ("pata nahi", "unknown"),
        ("पता नहीं", "unknown"),
    ],
)
def test_hindi_seriousness_answers(text, expected):
    assert _pending_reply("seriousness", text).get("seriousness") == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("theek ho gaya", "recovered"),
        ("behtar ho gaya", "recovered"),
        ("ठीक हो गई", "recovered"),
        ("बेहतर हो गया", "recovered"),
        ("theek ho raha hai", "recovering"),
        ("bigad gaya", "worsening"),
        ("बिगड़ गया", "worsening"),
        ("बिगड गया", "worsening"),
        ("nahi bigad gaya", "not_recovered"),
        ("theek nahi", "not_recovered"),
        ("ठीक नहीं", "not_recovered"),
        ("abhi tak hai", "not_recovered"),
        ("अभी भी है", "not_recovered"),
        ("mar gaya", "fatal"),
        ("मर गया", "fatal"),
        ("pata nahi", "unknown"),
    ],
)
def test_hindi_outcome_answers(text, expected):
    assert _pending_reply("outcome", text).get("outcome") == expected


def test_ambiguous_hindi_outcome_reply_is_not_stored():
    assert "outcome" not in _pending_reply("outcome", "shayad")


def test_hindi_unknown_and_repeat_replies_are_never_stored_as_values():
    # "Don't know / don't remember" and "please repeat" must never become a
    # name, an onset date or an event description.
    assert "patient_identifier" not in _pending_reply(
        "patient_identifier", "pata nahi"
    )
    assert "patient_identifier" not in _pending_reply(
        "patient_identifier", "याद नहीं"
    )
    assert "onset_date" not in _pending_reply("onset_date", "dobara kahiye")
    assert "adverse_event" not in _pending_reply("adverse_event", "दोबारा")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("theek hai", "yes"),
        ("ठीक है", "yes"),
        ("sahmat", "yes"),
        ("bilkul", "yes"),
        ("sahmat nahi", "ambiguous"),
        ("pata nahi", "ambiguous"),
        ("nahi yaad", "ambiguous"),
        ("dobara", "ambiguous"),
        ("kya", "ambiguous"),
        ("samajh nahi aaya", "ambiguous"),
    ],
)
def test_more_hindi_confirmation_replies(text, expected):
    from app.workflow.service.conversational import _classify_confirmation_turn

    verdict, _ = _classify_confirmation_turn(text)
    assert verdict == expected


def test_ambiguous_hindi_reply_preserves_report_data():
    engine = AEStateEngine.create_case("AE-hindi-ambiguous")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn(INITIAL_REPORT)
    original_event = engine.case.adverse_event
    original_product = engine.case.product

    result = workflow.process_turn("shayad")

    # Nothing is inferred: the original event description, the product and
    # every other structured field stay exactly as captured.
    assert engine.case.adverse_event == original_event
    assert engine.case.product == original_product
    assert engine.case.patient_age is None
    assert result.extracted_information == {}
    assert result.completed is False


# ---- Hindi / Hinglish adverse-event descriptions ------------------------------


def test_devanagari_first_report_captures_full_event_description():
    result = AnswerLLMProvider().interpret("मुझे कल से सिर में दर्द है", {})

    assert result.intent == "ae_report"
    # The reporter's own wording is stored verbatim - never translated or
    # reduced to a canned symptom.
    assert result.extracted_fields["adverse_event"] == "मुझे कल से सिर में दर्द है"


def test_romanized_hindi_report_captures_product_and_full_description():
    result = AnswerLLMProvider().interpret(
        "lantus lene ke baad mujhe dard ho raha hai", {}
    )

    assert result.intent == "ae_report"
    assert result.extracted_fields["product"] == "Lantus SoloStar"
    assert (
        result.extracted_fields["adverse_event"]
        == "lantus lene ke baad mujhe dard ho raha hai"
    )


def test_mixed_hindi_english_description_is_captured_verbatim():
    message = "Lantus lene ke baad I have a severe headache and vomiting"
    result = AnswerLLMProvider().interpret(message, {})

    assert result.intent == "ae_report"
    assert result.extracted_fields["adverse_event"] == message


def test_hindi_additional_symptom_is_appended_not_replacing():
    engine = AEStateEngine.create_case("AE-hindi-event-merge")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I took Lantus and have a headache.")
    original = engine.case.adverse_event
    workflow.process_turn("yes")  # Step 4: medication verified

    workflow.process_turn("ab mujhe ulti bhi ho rahi hai")

    # The original description is kept and the new symptom is merged in.
    assert engine.case.adverse_event.startswith(original)
    assert engine.case.adverse_event.endswith("ab mujhe ulti bhi ho rahi hai")


def test_english_ultimately_is_not_a_symptom():
    result = AnswerLLMProvider().interpret("I ultimately felt fine", {})

    # "ulti" must only match as the Hindi symptom, never inside English words.
    assert "adverse_event" not in result.extracted_fields


def test_hindi_description_survives_later_turns_and_confirmation():
    from app.workflow.service.conversational import NAME_CONFIRMATION_FIELDS

    engine = AEStateEngine.create_case("AE-hindi-event-preserve")
    engine.update("product", "Lantus SoloStar")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=NAME_CONFIRMATION_FIELDS
    )
    workflow.process_turn("मुझे कल से सिर में दर्द है")
    original = engine.case.adverse_event
    assert original == "मुझे कल से सिर में दर्द है"

    # Ambiguous and repeat-request replies never touch the description.
    workflow.process_turn("shayad")
    workflow.process_turn("dobara kahiye")
    assert engine.case.adverse_event == original

    # Reporter role, then the patient details (name and date of birth, per
    # the approved Step 6B.1): the candidate is not committed until the
    # combined read-back is confirmed, and the description is untouched
    # throughout.
    workflow.process_turn("myself")
    workflow.process_turn("Demo Patient")
    assert engine.case.patient_identifier is None
    assert engine.case.adverse_event == original

    workflow.process_turn("27/05/2000")
    workflow.process_turn("yes")
    assert engine.case.patient_identifier == "Demo Patient"
    assert engine.case.adverse_event == original


# ---- spoken-English DOB ("twenty seven may two thousand") --------------------


def _age_pending_engine(case_id: str) -> AEStateEngine:
    """Engine with everything captured; the age/DOB question is next."""
    engine = AEStateEngine.create_case(case_id)
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    assert engine.next_field()[0] == "patient_age"
    return engine


def test_spoken_dob_exact_repro_phrase_is_normalized_and_advances():
    engine = _age_pending_engine("AE-spoken-dob-repro")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("twenty seven may two thousand")

    # The DOB is normalized to DD-MMM-YYYY - never converted into a derived
    # age and never left in the raw spoken form.
    assert engine.case.patient_age == "27-May-2000"
    assert engine.case.patient_age != str(_TODAY_YEAR - 2000)
    # The workflow advances to the next question instead of re-asking the
    # age in years. The approved order collects the patient contact number
    # before consent.
    assert result.next_question is not None
    assert result.next_question[0] == "contact_phone"
    assert "How old are you, in years?" not in result.response_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("twenty seven may two thousand", "27-May-2000"),
        ("27 may two thousand", "27-May-2000"),
        (
            "twenty seven may two thousand five",
            "27-May-2005",
        ),
        (
            "twenty seven may two thousand and I have a headache",
            "27-May-2000",
        ),
        (
            "I was born on twenty seven may two thousand",
            "27-May-2000",
        ),
        # Decade-style years are complete and unambiguous ("nineteen ninety"
        # is how 1990 is spoken).
        ("twenty seven may nineteen ninety", "27-May-1990"),
        ("twenty seven june nineteen ninety eight", "27-Jun-1998"),
    ],
)
def test_spoken_english_dob_is_normalized(text, expected):
    assert _pending_age_fields(text)["patient_age"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "may two thousand",  # missing day
        "twenty seven may",  # missing year
    ],
)
def test_ambiguous_spoken_dob_is_not_stored(text):
    fields = _pending_age_fields(text)
    assert "patient_age" not in fields
    assert "dob_age_conflict" not in fields


def test_spoken_dob_conflicting_with_age_asks_one_clarification():
    engine = _age_pending_engine("AE-spoken-dob-conflict")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(
        "I am 20 years old and twenty seven may two thousand"
    )

    # Neither value is silently chosen; both are held for one clarification
    # (the date of birth in its normalized DD-MMM-YYYY form).
    assert engine.case.patient_age is None
    assert workflow._dob_age_conflict == {
        "dob": "27-May-2000",
        "age": "20",
    }
    assert result.next_question is not None
    assert result.next_question[0] == "patient_age"
    prompt = result.next_question[1]
    assert "27-May-2000" in prompt
    assert "20 years" in prompt
    assert result.human_review_required is False


# ---- product scope (Lantus SoloStar demo product) ----------------------------


@pytest.mark.parametrize(
    "text",
    [
        "lantus",
        "lantis",
        "lantos",
        "lanthus",
        "lan tus",
        "lantern",
        "solostar",
        "Lantus SoloStar",
        "Lantus Solo Star",
        "Lanta Solar Star",
        "Lantus Solar Star",
        "सोलो स्टार",
        "सोलोस्टार",
    ],
)
def test_lantus_spelling_variants_canonicalize(text):
    assert _pending_reply("product", text).get("product") == "Lantus SoloStar"


def test_variant_report_preserves_original_description():
    engine = AEStateEngine.create_case("AE-lantern-variant")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("I took lantern and have a headache")

    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == "I took lantern and have a headache"


@pytest.mark.parametrize(
    "text",
    [
        "लेंटस",
        "लैंटस",
        "लैन्टस",
        "लेन्टस",
        "लेंटस मेडिकेशन",
        "लैन्टस टैबलेट",
    ],
)
def test_hindi_lantus_transliterations_canonicalize(text):
    assert _pending_reply("product", text).get("product") == "Lantus SoloStar"


def test_hindi_first_report_canonicalizes_lantus_and_preserves_description():
    engine = AEStateEngine.create_case("AE-hindi-lantus-report")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    report = "मैंने कल लेंटस लिया और आज सिर दर्द हो रहा है"
    result = workflow.process_turn(report)

    # The transliterated product is recognized as the demo product, so the
    # report is NOT terminated as out of scope; the documented Step 4
    # medication verification is asked before reporter identification.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == report
    assert "configured for Lantus SoloStar only" not in result.response_text
    assert result.completed is False
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"

    # Step 4 confirmed: intake continues at reporter identification (Step 5).
    verified = workflow.process_turn("yes")
    assert "configured for Lantus SoloStar only" not in verified.response_text
    assert "person who experienced" in verified.response_text
    assert verified.next_question is not None
    assert verified.next_question[0] == "reporter_role"


def test_hindi_paracetamol_is_still_identified_separately():
    # The new Lantus aliases must not interfere with the conservative
    # non-Lantus medicine table: paracetamol keeps its own canonical name.
    assert _pending_reply("product", "पैरासिटामोल").get("product") == "Paracetamol"


def test_other_suspected_medicine_gets_scope_message_and_keeps_description():
    from app.language import translate

    engine = AEStateEngine.create_case("AE-scope-ibuprofen")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I took ibuprofen and have a rash")

    result = workflow.process_turn("ibuprofen")

    # The event is NOT attributed to Lantus: the real suspect is stored.
    assert engine.case.product == "ibuprofen"
    assert engine.case.adverse_event == "I took ibuprofen and have a rash"
    # The reply explains the prototype scope and directs the user onward.
    assert "configured for Lantus SoloStar only" in result.response_text
    assert "ibuprofen" in result.response_text
    assert "manufacturer" in result.response_text
    assert "adverse-event reporting channel" in result.response_text
    # No submission claim, ever.
    assert "submitted" not in result.response_text.lower()
    assert result.completed is False
    # Both scope sentences are covered by the Hindi catalog.
    assert translate(result.response_text, "hi-IN").translated is True


def test_concomitant_medicine_is_preserved_without_scope_message():
    engine = AEStateEngine.create_case("AE-concomitant")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    first = workflow.process_turn("I also take metformin")
    assert engine.case.concomitant_medications == ["metformin"]
    assert engine.case.product == "Lantus SoloStar"
    assert "configured for Lantus SoloStar only" not in first.response_text

    second = workflow.process_turn("plus I take aspirin")
    assert engine.case.concomitant_medications == ["metformin", "aspirin"]
    assert "configured for Lantus SoloStar only" not in second.response_text


def test_vague_or_self_mentions_are_not_concomitant_medicines():
    engine = AEStateEngine.create_case("AE-concomitant-guard")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("I also take care of my kids")
    workflow.process_turn("I also take Lantus")

    assert engine.case.concomitant_medications == []


def test_urgent_stays_ahead_of_scope_message():
    engine = AEStateEngine.create_case("AE-scope-urgent")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I took ibuprofen and have a rash")

    urgent = workflow.process_turn(
        "I took ibuprofen and am having a severe heart attack"
    )

    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert "configured for Lantus SoloStar only" not in urgent.response_text


# ---- Hindi suspect medicine & first-person replies ----------------------------


@pytest.mark.parametrize(
    "text",
    [
        "पैरासिटामोल",
        "पैरासीटामोल",
        "पेरासिटामोल",
        "paracetamol",
        "paraacetamol",
        "paracetomol",
    ],
)
def test_paracetamol_variants_canonicalize_as_suspect_product(text):
    assert _pending_reply("product", text).get("product") == "Paracetamol"


def test_hindi_paracetamol_first_report_captures_suspect_product():
    result = AnswerLLMProvider().interpret(
        "मुझे लगता है पैरासिटामोल के कारण हुआ होगा ये", {}
    )

    assert result.intent == "ae_report"
    assert result.extracted_fields["product"] == "Paracetamol"


def test_english_due_to_paracetamol_captures_suspect_product():
    result = AnswerLLMProvider().interpret("I have a rash due to paracetamol", {})

    assert result.extracted_fields["product"] == "Paracetamol"


def test_paracetamol_without_causation_is_not_the_suspect_product():
    result = AnswerLLMProvider().interpret("I also take paracetamol", {})

    # Mentioned as a concomitant medicine, not as the cause.
    assert "product" not in result.extracted_fields
    assert result.extracted_fields["concomitant_medications"] == ["paracetamol"]


def test_hindi_paracetamol_repro_shows_scope_and_ends_intake():
    engine = AEStateEngine.create_case("AE-paracetamol-hi")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("मुझे लगता है पैरासिटामोल के कारण हुआ होगा ये")

    # The suspect keeps its identity; the scope message is shown and routine
    # intake ends (no further questions, never completed).
    assert engine.case.product == "Paracetamol"
    assert "configured for Lantus SoloStar only" in result.response_text
    assert "Paracetamol" in result.response_text
    assert "cannot continue this report" in result.response_text
    assert result.next_question is None
    assert result.completed is False
    assert workflow._out_of_scope_product == "Paracetamol"


def test_solo_star_devanagari_answer_continues_intake():
    engine = AEStateEngine.create_case("AE-solo-star-hi")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("सोलो स्टार के बारे में")

    # Recognised as the demo product: no scope message; the documented Step 4
    # medication verification is asked before intake continues.
    assert engine.case.product == "Lantus SoloStar"
    assert workflow._out_of_scope_product is None
    assert "configured for Lantus SoloStar only" not in result.response_text
    assert result.next_question is not None
    assert result.next_question[0] == "product_verification"

    verified = workflow.process_turn("yes")
    assert verified.next_question is not None
    assert verified.next_question[0] == "adverse_event"


@pytest.mark.parametrize(
    "text",
    ["Lantus Solar Star", "Lanta Solar Star", "Lantus Solo Star"],
)
def test_english_speech_variants_continue_intake(text):
    engine = AEStateEngine.create_case("AE-solar-star")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(text)

    assert engine.case.product == "Lantus SoloStar"
    assert workflow._out_of_scope_product is None
    assert "configured for Lantus SoloStar only" not in result.response_text


def test_out_of_scope_session_rejects_further_routine_turns():
    engine = AEStateEngine.create_case("AE-oos-session")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I have a rash due to paracetamol")

    # The session is now out of scope: ordinary replies change nothing and
    # no reporter/patient/consent/contact question is ever asked again.
    follow = workflow.process_turn("myself")

    assert follow.extracted_information == {}
    assert engine.case.reporter_role is None
    assert engine.case.patient_identifier is None
    assert engine.case.consent_to_contact is None
    assert follow.next_question is None
    assert follow.completed is False
    assert follow.human_review_required is False
    assert "cannot continue this report" in follow.response_text
    # The out-of-scope medicine's identity is preserved.
    assert engine.case.product == "Paracetamol"


def test_ambiguous_product_reply_asks_again_instead_of_ending():
    engine = AEStateEngine.create_case("AE-ambiguous-product")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    # Step 2 is answered so the medication question (Step 3) is on the table.
    workflow.process_turn("I have a rash")
    result = workflow.process_turn("maybe something?")

    # Nothing guessed, nothing terminated: the medication question is re-asked.
    assert engine.case.product is None
    assert workflow._out_of_scope_product is None
    assert "configured for Lantus SoloStar only" not in result.response_text
    assert result.next_question is not None
    assert result.next_question[0] == "product"
    assert "Which medicine were you using?" in result.response_text


def test_concomitant_medicine_does_not_end_intake():
    engine = AEStateEngine.create_case("AE-concomitant-no-end")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("I also take metformin")

    assert engine.case.concomitant_medications == ["metformin"]
    assert workflow._out_of_scope_product is None
    assert "cannot continue this report" not in result.response_text
    assert result.next_question is not None
    assert result.next_question[0] == "reporter_role"


def test_urgent_still_escalates_after_out_of_scope():
    engine = AEStateEngine.create_case("AE-oos-urgent")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())
    workflow.process_turn("I have a rash due to paracetamol")

    urgent = workflow.process_turn("I am having a severe heart attack")

    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    assert urgent.completed is False


def test_paracetamol_suspect_keeps_scope_message_in_english():
    engine = AEStateEngine.create_case("AE-paracetamol-en")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("I have a rash due to paracetamol")

    assert engine.case.product == "Paracetamol"
    assert engine.case.adverse_event == "I have a rash due to paracetamol"
    assert "configured for Lantus SoloStar only" in result.response_text
    assert "Paracetamol" in result.response_text


@pytest.mark.parametrize(
    "text",
    [
        "मैं ही हूँ जिसे यह हुआ है",
        "मैं ही हूँ जिसे यह हिस्सा हुआ है",
        "यही अनुभव है",
        "ये मेरा ही अनुभव है",
        "मेरा ही अनुभव है",
    ],
)
def test_hindi_first_person_replies_are_self(text):
    assert _pending_reply("reporter_role", text).get("reporter_role") == "self"


@pytest.mark.parametrize(
    "text",
    [
        "I'm reporting for my mother",
        "मेरी पत्नी को यह हुआ",
        "मेरी माँ का यही अनुभव है",
    ],
)
def test_other_reporter_replies_still_win_over_self_cues(text):
    assert (
        _pending_reply("reporter_role", text).get("reporter_role")
        == "someone_else"
    )


# ---- endpoint: spoken digit phone (Prisma-style) ----------------------------

PATIENT_ID_TRANSCRIPT = "Demo Patient 001"


def test_real_entry_spoken_phone_reaches_contact_captured():
    """Full AE journey through app.main: report -> Step 4 verification ->
    identity -> patient name and date of birth (with combined read-back
    confirmation) -> spoken phone (with read-back confirmation) -> consent,
    verifying the patient contact number is captured and confirmed before
    consent is asked."""
    from app.main import create_app

    stt = FakeSTT(
        [
            REAL_AUDIO_1,
            "yes",  # Step 4: medication verified
            REAL_AUDIO_2,  # role self + age
            PATIENT_ID_TRANSCRIPT,  # name; age is known -> combined read-back
            "yes",  # Step 6B.2: patient details confirmed
            SPOKEN_PHONE,  # Step 6B.3 -> contact read-back
            "yes",  # Step 6B.4: contact number confirmed
        ]
    )
    app = create_app(prisma=stt, timbre=FakeTTS())
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmp:
        one = _post(client, REAL_SESSION_ID, tmp).json()
        _post(client, REAL_SESSION_ID, tmp)  # Step 4 confirmed
        _post(client, REAL_SESSION_ID, tmp)  # role + age
        four = _post(client, REAL_SESSION_ID, tmp).json()  # name -> read-back
        five = _post(client, REAL_SESSION_ID, tmp).json()  # confirmed
        six = _post(client, REAL_SESSION_ID, tmp).json()  # phone read-back
        seven = _post(client, REAL_SESSION_ID, tmp).json()  # phone confirmed

    # Turn 4: the name is held for the combined name-and-date read-back (the
    # age was captured with Audio 2); the case does not advance before the
    # details are confirmed.
    assert four["current_state"] == WorkflowState.PRODUCT_IDENTIFIED.value
    assert four["next_question"][0] == "confirmation"
    assert PATIENT_ID_TRANSCRIPT in four["response_text"]
    assert four["ae"]["collected_information"]["patient_identifier"] is None

    # Turn 5: patient details confirmed; the contact number is asked next.
    assert (
        five["ae"]["collected_information"]["patient_identifier"]
        == PATIENT_ID_TRANSCRIPT
    )
    assert five["current_state"] == WorkflowState.PATIENT_IDENTIFIED.value
    assert five["next_question"][0] == "contact_phone"

    # Turn 6: the spoken number is extracted and read back before it is
    # committed (the candidate only, never the case value).
    assert six["next_question"][0] == "confirmation"
    assert six["ae"]["extracted_information"]["contact_phone"] == "9886094321"
    assert six["ae"]["collected_information"]["contact_phone"] is None

    # Turn 7: contact captured; consent (Step 7) is asked afterwards.
    assert seven["workflow"] == "ae"
    assert seven["intent"] == "unknown"
    assert seven["session_id"] == REAL_SESSION_ID
    assert seven["case_id"] == one["case_id"]
    collected = seven["ae"]["collected_information"]
    assert collected["contact_phone"] == "9886094321"
    assert seven["current_state"] == WorkflowState.CONTACT_CAPTURED.value
    assert seven["next_question"][0] == "consent_to_contact"
    assert seven["completed"] is False


# ---- Hindi number-word ages ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["मेरी उम्र पच्चीस साल है", "पच्चीस साल है", "पच्चीस"],
)
def test_hindi_age_repro_answers_are_captured(text):
    assert _pending_age_fields(text)["patient_age"] == "25"


def test_hindi_age_with_umr_cue_is_captured_opportunistically():
    result = AnswerLLMProvider().interpret("मेरी उम्र पच्चीस साल है", {})

    assert result.extracted_fields["patient_age"] == "25"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("मेरी उम्र चालीस साल है", "40"),
        ("मेरी उम्र नौ साल है", "9"),
        ("बीस साल", "20"),
        ("पैंतीस", "35"),
        ("मेरी उम्र पचीस साल है", "25"),  # common spelling variant
    ],
)
def test_other_hindi_number_words(text, expected):
    assert _pending_age_fields(text)["patient_age"] == expected


def test_hindi_onset_phrases_are_never_captured_as_age():
    # Opportunistic path (no age question pending): onset wording such as
    # "two years ago" must never become an age.
    for text in ("दो दिन पहले", "दो साल पहले"):
        result = AnswerLLMProvider().interpret(text, {})
        assert "patient_age" not in result.extracted_fields


@pytest.mark.parametrize(
    "text",
    ["अच्छा", "बहुत ज्यादा", "दो दिन पहले"],
)
def test_unrecognised_hindi_replies_store_no_age(text):
    assert "patient_age" not in _pending_age_fields(text)


def test_english_age_path_is_unchanged():
    assert _pending_age_fields("I am forty five years old")["patient_age"] == "45"


def test_unclarifiable_age_parks_report_after_targeted_clarification():
    engine = AEStateEngine.create_case("AE-age-loop")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "headache")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    assert engine.next_field()[0] == "patient_age"
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    # First failure keeps the normal age question; the second offers the
    # targeted clarification; the third parks the report for human review
    # instead of looping the generic question forever.
    workflow.process_turn("okay sure")
    targeted = workflow.process_turn("okay sure")
    assert "Could you say the date of birth again?" in targeted.response_text
    assert targeted.next_question is not None
    assert targeted.next_question[0] == "patient_age"

    parked = workflow.process_turn("okay sure")
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert parked.human_review_required is True
    assert parked.completed is False
    assert engine.case.patient_age == "unknown"
    # Previously collected report data is preserved.
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == "headache"


def test_targeted_age_clarification_translates_to_hindi():
    from app.language import translate

    result = translate("Could you say the date of birth again?", "hi-IN")
    assert result.translated is True


# ---- live-test failures: role, confirmation, consent, contact method ----------


def test_mujhe_hi_hua_is_self():
    assert (
        _pending_reply("reporter_role", "मुझे ही हुआ है ये").get("reporter_role")
        == "self"
    )
    assert (
        _pending_reply("reporter_role", "mujhe hi hua").get("reporter_role")
        == "self"
    )
    # Previously-working form keeps working.
    assert (
        _pending_reply("reporter_role", "मेरा ही अनुभव है").get("reporter_role")
        == "self"
    )


def test_hindi_kinship_reply_stays_someone_else():
    assert (
        _pending_reply("reporter_role", "मेरी बहन को यह हुआ").get("reporter_role")
        == "someone_else"
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("सही है।", "yes"),
        ("सही", "yes"),
        ("sahi hai", "yes"),
        ("हाँ हाँ", "yes"),
        ("सही नहीं है", "ambiguous"),
    ],
)
def test_hindi_confirmation_verdicts(text, expected):
    from app.workflow.service.conversational import _classify_confirmation_turn

    verdict, _ = _classify_confirmation_turn(text)
    assert verdict == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("जरूर", True),
        ("ज़रूर", True),
        ("jarur", True),
        ("हाँ", True),
        ("जरूर नहीं", False),
        ("नहीं जरूर", False),
    ],
)
def test_hindi_consent_with_jarur(text, expected):
    assert (
        _pending_reply("consent_to_contact", text).get("consent_to_contact")
        is expected
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ईमेल चलेगा", "email"),
        ("ईमेल", "email"),
        ("ईमेल से", "email"),
        ("ईमेल पर", "email"),
        ("फोन चलेगा", "phone"),
        ("दोनों चलेगा", "both"),
    ],
)
def test_hindi_contact_method_answers(text, expected):
    assert (
        _pending_reply("contact_method", text).get("preferred_contact_method")
        == expected
    )


def test_ambiguous_contact_method_reply_stores_nothing():
    assert "preferred_contact_method" not in _pending_reply(
        "contact_method", "बिल्कुल ठीक"
    )


def test_hindi_consent_answer_advances_to_the_email_question():
    engine = AEStateEngine.create_case("AE-method-hi")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Alpha Jones")
    engine.update("patient_age", "25")
    engine.update("contact_phone", "9999990000")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    # Step 7 consent, answered in Hindi. The approved order collects the
    # patient contact number before consent, so no contact-method question
    # is asked; consent then opens the Step 8 email question.
    consented = workflow.process_turn("हाँ")
    assert engine.case.consent_to_contact is True
    assert consented.next_question is not None
    assert consented.next_question[0] == "contact_email"

    # Step 8: the email address is offered, requested and captured.
    workflow.process_turn("yes")
    workflow.process_turn("demo.patient@example.com")
    workflow.process_turn("yes")
    assert engine.case.contact_email == "demo.patient@example.com"


# ---- audit: initial free-text description handling ---------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The problem came from lantus somehow",
        "लेंटस से बहुत तकलीफ हुई",
    ],
)
def test_product_anchored_vague_first_report_preserves_full_description(text):
    # Product names the suspected medicine but no report/symptom phrase is
    # recognised: the full original wording must still be captured verbatim,
    # and the report must not be terminated as out of scope.
    engine = AEStateEngine.create_case("AE-product-anchored")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn(text)

    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.adverse_event == text
    assert "configured for Lantus SoloStar only" not in result.response_text
    assert result.completed is False


@pytest.mark.parametrize(
    "text",
    [
        "davai lene ke baad kuch gadbad ho gayi",
        "दवा लेने के बाद बहुत तकलीफ हुई",
        "le liya aur tabiyat kharab ho gayi",
    ],
)
def test_hinglish_report_cues_capture_description_verbatim(text):
    # Narrow English/Hindi/Hinglish cues classify the first message without
    # inventing anything: the captured value is the message itself.
    engine = AEStateEngine.create_case("AE-hinglish-cues")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn(text)

    assert engine.case.adverse_event == text
    assert engine.case.unclassified_initial_input is None


def test_vague_first_message_is_stored_as_unclassified_initial_input():
    engine = AEStateEngine.create_case("AE-unclassified-vague")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    vague = "kuch gadbad ho rahi hai"
    result = workflow.process_turn(vague)

    # Not classified: no adverse event is invented, but the original first
    # message is preserved verbatim in its own field.
    assert engine.case.adverse_event is None
    assert engine.case.unclassified_initial_input == vague
    # Intake proceeds with the documented Step 2 invitation.
    assert "Please tell me what happened, in your own words." in result.response_text

    # A later structured answer never overwrites the original wording.
    workflow.process_turn("I have a rash")
    workflow.process_turn("lantus")
    assert engine.case.product == "Lantus SoloStar"
    assert engine.case.unclassified_initial_input == vague


def test_repeated_unclear_event_replies_park_review_with_empty_field():
    engine = AEStateEngine.create_case("AE-event-review-bound")
    engine.update("product", "Lantus SoloStar")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    # "ok" / "nothing" are non-answers to the event question: they must never
    # become the description, and the clarification loop must stay bounded.
    first = workflow.process_turn("ok")
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    # The documented Step 2 invitation is the first question asked.
    assert "Please tell me what happened, in your own words." in first.response_text

    second = workflow.process_turn("ok")
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert "describe the side effect" in second.response_text

    third = workflow.process_turn("nothing")
    # Bounded: human review with the field left genuinely empty (never a
    # fabricated description).
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert third.human_review_required is True
    assert engine.case.adverse_event is None
    assert "adverse_event" in engine.missing_information()


def test_unclear_product_replies_park_review_without_scope_stop():
    engine = AEStateEngine.create_case("AE-product-review-bound")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("kuch gadbad ho rahi hai")
    # Step 2 answered, so the unclear replies below are answers to the
    # medication question (Step 3).
    workflow.process_turn("I have a rash")
    second = workflow.process_turn("shayad")
    assert engine.current_state is not WorkflowState.HUMAN_REVIEW
    assert "didn't catch the product name" in second.response_text

    third = workflow.process_turn("kya?")

    # An unclear answer is bounded to human review with the product left
    # empty; it is NOT treated as a confirmed out-of-scope medicine.
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert third.human_review_required is True
    assert engine.case.product is None
    assert "configured for Lantus SoloStar only" not in third.response_text
    assert "product" in engine.missing_information()


# ---- question-shaped text is never an adverse event -------------------------


def test_fresh_mi_question_is_not_an_ae_report():
    provider = AnswerLLMProvider()
    result = provider.interpret(
        "What are the common adverse reactions listed for Lantus SoloStar "
        "in its product information?",
        {},
    )
    assert "adverse_event" not in result.extracted_fields
    assert result.intent != "ae_report"


def test_fresh_product_free_question_is_not_captured():
    result = AnswerLLMProvider().interpret(
        "What are the possible side effects of this injection?", {}
    )
    assert "adverse_event" not in result.extracted_fields
    assert result.intent != "ae_report"


def test_mi_question_is_not_stored_when_adverse_event_pending():
    engine = AEStateEngine.create_case("AE-question-guard-pending")
    engine.update("product", "Lantus SoloStar")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    workflow.process_turn("What are the possible side effects of this injection?")

    assert engine.case.adverse_event is None
    # The workflow simply re-asks for the actual description.
    assert "adverse_event" in engine.missing_information()


def test_repeat_request_is_not_stored_as_adverse_event():
    engine = AEStateEngine.create_case("AE-question-guard-repeat")
    engine.update("product", "Lantus SoloStar")
    workflow = ConversationalAEWorkflow(engine, AnswerLLMProvider())

    result = workflow.process_turn("Can you repeat that?")

    assert engine.case.adverse_event is None
    assert result.response_text
    assert "adverse_event" in engine.missing_information()


def test_genuine_event_statement_is_still_captured_verbatim():
    statement = (
        "I took Lantus yesterday and since this morning I have a "
        "severe headache and vomiting."
    )
    result = AnswerLLMProvider().interpret(statement, {})
    assert result.intent == "ae_report"
    assert result.extracted_fields["adverse_event"] == statement


def test_date_led_statement_is_not_treated_as_a_question():
    statement = "May 12 I felt a rash after the injection"
    result = AnswerLLMProvider().interpret(statement, {})
    assert result.extracted_fields.get("adverse_event") == statement
    assert result.intent == "ae_report"


# ---- Lantus SoloStar alias recognition across the ten locales ----------------


@pytest.mark.parametrize(
    "text",
    [
        # English ASR variants (existing behaviour, pinned).
        "I took Lantus",
        "The lantern pen",
        "SoloStar injection",
        # Hindi: the existing Devanagari aliases and the observed STT
        # variants ("लैंटर" alone, "लैंडर सोलो स्टार" via the sub-brand).
        "मुझे लैंटस लिया था",
        "लैंटर",
        "लैंडर सोलो स्टार",
        # Bengali
        "ল্যান্টাস নিয়েছিলাম",
        "সোলো স্টার",
        # Gujarati
        "મેં લાન્ટસ લીધું",
        "લેન્ટસ",
        "સોલો સ્ટાર",
        # Kannada
        "ನಾನು ಲ್ಯಾಂಟಸ್ ತೆಗೆದುಕೊಂಡೆ",
        "ಸೋಲೋ ಸ್ಟಾರ್",
        # Malayalam: the observed browser-test transcript stem.
        "ലാൻഡസ്",
        "ലാൻറസ്",
        "ലാൻടസ്",
        "ലാന്ടസ്",
        "ലാന്ദസ്",
        "ലാറ്റസ്",
        "സോളോ സ്റ്റാർ",
        "സോളോസ്റ്റാ",
        "സോലോ സ്റ്റാർ",
        "സോലോസ്റ്റാ",
        "ഞാനെടുത്ത മരുന്നിൻ്റെ പേര് ആണ് ലാൻടസ് സോലോ സ്റ്റാർ",
        # Marathi
        "लँटस",
        # Punjabi
        "ਲੈਂਟਸ",
        "ਸੋਲੋ ਸਟਾਰ",
        # Tamil
        "லான்டஸ்",
        "லானடஸ்",
        "சோலோ ஸ்டார்",
        # Telugu
        "లాంటస్",
        "సోలో స్టార్",
        # English: observed phonetic and spacing variants.
        "Landus Solo Star",
        "ZoloStar",
        "Zolo Star",
    ],
)
def test_lantus_aliases_across_all_ten_locales_canonicalize(text):
    assert _normalize_product(text) == "Lantus SoloStar", text


@pytest.mark.parametrize(
    "text",
    [
        "crocin",
        "paracetamol",
        "unknown tablet",
        "solar star",
        # A garbled ASR word is never aliased, alone or in context.
        "ഞാൻ എടുത്ത മരുന്നാണ് കൊതുക",
        "കൊതുക",
    ],
)
def test_unrelated_or_garbled_medicines_never_canonicalize_to_lantus(text):
    assert _normalize_product(text) is None, text


# ---- unclear Indic-script replies are clarified, never captured as product ---


# One unclear, non-medication reply per supported language (synthetic text).
UNCLEAR_MEDICATION_REPLIES = [
    "എഴുമല്ലേ പിന്നെ അങ്ങനെ ലഭിക്കാന് കഴിയുന്നു",
    "കൊതുക വന്നു പോയി",
    "ಅದು ಹೀಗೆ ಇರುತ್ತದೆ",
    "समझ नहीं आया कैसे",
    "এই যে",
    "அப்படித்தான்",
    "అలాగే ఉంది",
    "એમ જ થયું",
    "ਉਸ ਤਰ੍ਹਾਂ",
]


@pytest.mark.parametrize("text", UNCLEAR_MEDICATION_REPLIES)
def test_unclear_indic_medication_replies_are_never_product_names(text):
    provider = AnswerLLMProvider()
    result = provider.interpret(
        text,
        {
            "next_field": ["product", "Which medicine were you using?"],
            "collected": {"adverse_event": "headache"},
        },
    )
    assert "product" not in result.extracted_fields, text


@pytest.mark.parametrize(
    "text",
    [
        "അറിയില്ല",
        "தெரியல",
        "తెలియదు",
        "ಗೊತ್ತಿಲ್ಲ",
        "माहीत नाही",
        "ખબર નથી",
        "জানি না",
        "ਪਤਾ ਨਹੀਂ",
    ],
)
def test_dont_know_replies_in_new_locales_map_to_unknown_product(text):
    provider = AnswerLLMProvider()
    result = provider.interpret(
        text,
        {
            "next_field": ["product", "Which medicine were you using?"],
            "collected": {"adverse_event": "headache"},
        },
    )
    assert (
        result.extracted_fields.get("product") == "Unknown (to be confirmed)"
    ), text


def test_english_dont_know_product_reply_still_maps_to_unknown():
    provider = AnswerLLMProvider()
    result = provider.interpret(
        "I don't know",
        {
            "next_field": ["product", "Which medicine were you using?"],
            "collected": {"adverse_event": "headache"},
        },
    )
    assert result.extracted_fields.get("product") == "Unknown (to be confirmed)"


# ---- spoken DOB parsing in the other supported languages ----------------------

from app.providers.answer import _extract_dob_date  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        # The observed Malayalam forms.
        "ഇരുപത് മേയ് രണ്ടായിരം",
        "ഇരുപത് മെയ് രണ്ടായിരം",
        # One clear spoken date per remaining locale (20 May 2000).
        "বিশ মে দুই হাজার",
        "વીસ મે બે હજાર",
        "ಇಪ್ಪತ್ತು ಮೇ ಎರಡು ಸಾವಿರ",
        "वीस मे दोन हजार",
        "ਵੀਹ ਮਈ ਦੋ ਹਜ਼ਾਰ",
        "இருபது மே இரண்டு ஆயிரம்",
        "ఇరవై మే రెండు వెయ్యి",
        # English spoken form (existing behaviour, pinned).
        "Twenty May Two Thousand",
    ],
)
def test_spoken_dob_parses_to_20_may_2000_across_locales(text):
    parsed, verbatim = _extract_dob_date(text)
    assert parsed is not None, text
    assert normalized_dob_in(text) == "20-May-2000", text
    assert verbatim is not None, text


@pytest.mark.parametrize(
    "text",
    [
        "ഇരുപത് മേയ് രണ്ടായിരത്തി അഞ്ച്",
        "বিশ মে দুই হাজার পাঁচ",
        "વીસ મે બે હજાર પાંચ",
        "ಇಪ್ಪತ್ತು ಮೇ ಎರಡು ಸಾವಿರದ ಐದು",
        "वीस मे दोन हजार पाच",
        "ਵੀਹ ਮਈ ਦੋ ਹਜ਼ਾਰ ਪੰਜ",
        "இருபது மே இரண்டு ஆயிரத்து ஐந்து",
        "ఇరవై మే రెండు వెయ్యి ఐదు",
    ],
)
def test_spoken_dob_year_unit_forms_parse_to_20_may_2005(text):
    parsed, _ = _extract_dob_date(text)
    assert parsed is not None, text
    assert normalized_dob_in(text) == "20-May-2005", text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # STT commonly transcribes spoken numbers as digits inside an
        # otherwise spoken date; every locale must accept the mixed forms.
        ("ഇരുപത് മേയ് 2006", "20-May-2006"),
        ("20 മേയ് രണ്ടായിരത്തി ആറ്", "20-May-2006"),
        ("20 മേയ് 2006", "20-May-2006"),
        ("पच्चीस मई 2004", "25-May-2004"),
        ("25 मई दो हज़ार चार", "25-May-2004"),
        ("বিশ মে 2000", "20-May-2000"),
        ("વીસ મે 2000", "20-May-2000"),
        ("ಇಪ್ಪತ್ತು ಮೇ 2000", "20-May-2000"),
        ("वीस मे 2000", "20-May-2000"),
        ("ਵੀਹ ਮਈ 2000", "20-May-2000"),
        ("இருபது மே 2000", "20-May-2000"),
        ("ఇరవై మే 2000", "20-May-2000"),
    ],
)
def test_mixed_digit_and_spoken_dob_parses_across_locales(text, expected):
    assert normalized_dob_in(text) == expected, text


@pytest.mark.parametrize(
    "text",
    [
        "മേയ് 2006",
        "ഇരുപത് മേയ്",
        "ഇരുപത് മേയ് 117",
        "20 മേയ് 20061",
        "पच्चीस में 2004",
        "पच्चीस में दो हजार",
    ],
)
def test_mixed_form_incomplete_dob_is_never_guessed(text):
    assert normalized_dob_in(text) is None, text


@pytest.mark.parametrize(
    "month",
    ["മേ", "മേയ്", "മെയ്"],
)
def test_malayalam_may_spellings_parse(month):
    # The observed browser defect: the STT-short spelling മേ was missing
    # while മെയ് and മേയ് were accepted.
    assert normalized_dob_in(f"ഇരുപത് {month} രണ്ടായിരത്തി ആറ്") == "20-May-2006"


@pytest.mark.parametrize(
    ("day_text", "expected"),
    [
        # Spaced compound day forms: ASR often emits the compound with a
        # space; without table entries, "ഇരുപത്തി രണ്ട്" silently matched the
        # unit word alone as day 2 (02-May-2000).
        ("ഇരുപത്തി ഒന്ന്", "21-May-2000"),
        ("ഇരുപത്തി രണ്ട്", "22-May-2000"),
        ("ഇരുപത്തി മൂന്ന്", "23-May-2000"),
        ("ഇരുപത്തി നാല്", "24-May-2000"),
        ("ഇരുപത്തി അഞ്ച്", "25-May-2000"),
        ("ഇരുപത്തി ആറ്", "26-May-2000"),
        ("ഇരുപത്തി ഏഴ്", "27-May-2000"),
        ("ഇരുപത്തി എട്ട്", "28-May-2000"),
        ("ഇരുപത്തി ഒമ്പത്", "29-May-2000"),
        ("മുപ്പത്തി ഒന്ന്", "31-May-2000"),
    ],
)
def test_malayalam_spaced_compound_days_parse(day_text, expected):
    assert normalized_dob_in(f"{day_text} മേയ് രണ്ടായിരം") == expected


def test_malayalam_single_day_forms_unchanged():
    # The single-token compound and the unit-word-alone forms keep their
    # established values.
    assert normalized_dob_in("ഇരുപത്തിരണ്ട് മേയ് രണ്ടായിരം") == "22-May-2000"
    assert normalized_dob_in("രണ്ട് മേയ് രണ്ടായിരം") == "02-May-2000"
    assert normalized_dob_in("മുപ്പത് മേയ് രണ്ടായിരം") == "30-May-2000"


def test_transliterated_english_dob_stays_unsupported():
    # Documented gap: English number words transliterated into Malayalam
    # script (ട്വന്റി/ടു/തൗസൻഡ്) remain unsupported until their mappings are
    # verified by a Malayalam speaker; nothing is guessed at.
    assert normalized_dob_in("ട്വന്റി ടു മേ ടു തൗസൻഡ് ആറ്") is None


@pytest.mark.parametrize(
    "month",
    ["മേ", "മേയ്", "മെയ്"],
)
def test_combined_malayalam_may_spellings_name_and_dob(month):
    text = f"എൻ്റെ പേരാണ് രാജു പി കെ ജന്മതിഥി ഇരുപത് {month} രണ്ടായിരത്തി ആറ്"
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["patient_identifier", "What is your name?"]}
    )
    assert result.extracted_fields.get("patient_age") == "20-May-2006", text
    assert result.extracted_fields.get("patient_identifier") == "രാജു പി കെ", text


def test_every_locale_month_token_parses_with_digit_day_and_year():
    # Table-driven completeness: every month token in every locale table
    # (canonical and alias spellings alike) must parse in the complete date
    # shape, so a missing month alias can never hide again.
    from datetime import date as _date

    from app.providers.answer import _SPOKEN_DATE_TABLES, format_report_date

    for table in _SPOKEN_DATE_TABLES.values():
        for token, number in table["months"].items():
            text = f"1 {token} 2000"
            assert normalized_dob_in(text) == format_report_date(
                _date(2000, number, 1)
            ), text


def test_every_hindi_month_token_parses_in_its_required_shape():
    from datetime import date as _date

    from app.providers.answer import (
        _HINDI_MONTHS,
        _HINDI_MONTH_VARIANTS,
        format_report_date,
    )

    for token, number in _HINDI_MONTHS.items():
        if token in _HINDI_MONTH_VARIANTS:
            # Variant months keep their conservative shape: an exact year
            # unit word is required; a bare digit year is never guessed.
            text = f"1 {token} दो हजार चार"
            expected = format_report_date(_date(2004, number, 1))
        else:
            text = f"1 {token} 2000"
            expected = format_report_date(_date(2000, number, 1))
        assert normalized_dob_in(text) == expected, text


def test_every_english_month_token_parses():
    from datetime import date as _date

    from app.providers.answer import _MONTH_NAMES, format_report_date

    for token, number in _MONTH_NAMES.items():
        text = f"1 {token} 2000"
        assert normalized_dob_in(text) == format_report_date(
            _date(2000, number, 1)
        ), text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ഇരുപത് മാർച്ച രണ്ടായിരത്തി ആറ്", "20-Mar-2006"),
        ("ഇരുപത് ഓഗസ്റ്റ രണ്ടായിരം", "20-Aug-2000"),
        ("বিশ জানুয়ারী দুই হাজার", "20-Jan-2000"),
        ("বিশ ফেব্রুয়ারী দুই হাজার", "20-Feb-2000"),
        ("ಇಪ್ಪತ್ತು ಫೆಬ್ರುವರಿ ಎರಡು ಸಾವಿರ", "20-Feb-2000"),
        ("ಇಪ್ಪತ್ತು ಮಾರ್ಚಿ ಎರಡು ಸಾವಿರ", "20-Mar-2000"),
        ("ਵੀਹ ਸਿਤੰਬਰ ਦੋ ਹਜ਼ਾਰ", "20-Sep-2000"),
        ("ਵੀਹ ਅਗਸਟ ਦੋ ਹਜ਼ਾਰ", "20-Aug-2000"),
        ("இருபது ஆகஸ்டு இரண்டு ஆயிரம்", "20-Aug-2000"),
        ("ఇరవై ఆగష్టు రెండు వెయ్యి", "20-Aug-2000"),
        ("ఇరవై మార్చ్ రెండు వెయ్యి", "20-Mar-2000"),
    ],
)
def test_new_month_aliases_parse_in_full_spoken_dates(text, expected):
    assert normalized_dob_in(text) == expected, text


@pytest.mark.parametrize(
    "text",
    [
        "32 ജനുവരി 2000",
        "0 ജനുവരി 2000",
        "ഇരുപത് മേ",
        "മേ രണ്ടായിരത്തി ആറ്",
        "30 ഫെബ്രുവരി 2000",
        "ഇരുപത് മായ് രണ്ടായിരം",
    ],
)
def test_invalid_month_day_combinations_are_never_guessed(text):
    assert normalized_dob_in(text) is None, text


@pytest.mark.parametrize(
    ("text", "expected_name", "expected_dob"),
    [
        # The observed Malayalam contamination: the ASR variant ജന്മതിയി of
        # the ജന്മതിഥി label (with and without the പിന്നെ connector) used to
        # stay appended to the name. With no date spoken yet there is no DOB
        # to extract - and none is ever invented.
        (
            "എൻ്റെ പേരാണ് രാജു പി കെ പിന്നെ ജന്മതിയി",
            "രാജു പി കെ",
            None,
        ),
        (
            "എൻ്റെ പേരാണ് രാജു പി കെ ജന്മതിയി",
            "രാജു പി കെ",
            None,
        ),
        # The remaining locales' established DOB labels (the same vocabulary
        # the workflow's which-field correction already uses) with a valid
        # spoken date after them.
        ("রাহুল দাস জন্ম তারিখ বিশ মে দুই হাজার", "রাহুল দাস", "20-May-2000"),
        ("રમેશ પટેલ જન્મ તારીખ વીસ મે બે હજાર", "રમેશ પટેલ", "20-May-2000"),
        (
            "ರವಿ ಕುಮಾರ್ ಜನ್ಮ ದಿನಾಂಕ ಇಪ್ಪತ್ತು ಮೇ ಎರಡು ಸಾವಿರ",
            "ರವಿ ಕುಮಾರ್",
            "20-May-2000",
        ),
        ("अमोल शर्मा जन्मतारीख वीस मे दोन हजार", "अमोल शर्मा", "20-May-2000"),
        (
            "ਰਾਜੇਸ਼ ਕੁਮਾਰ ਜਨਮ ਤਾਰੀਖ ਵੀਹ ਮਈ ਦੋ ਹਜ਼ਾਰ",
            "ਰਾਜੇਸ਼ ਕੁਮਾਰ",
            "20-May-2000",
        ),
        (
            "ராஜு குமார் பிறந்த தேதி இருபது மே இரண்டு ஆயிரம்",
            "ராஜு குமார்",
            "20-May-2000",
        ),
        (
            "రాజు రెడ్డి పుట్టిన తేదీ ఇరవై మే రెండు వెయ్యి",
            "రాజు రెడ్డి",
            "20-May-2000",
        ),
    ],
)
def test_combined_name_and_dob_label_never_contaminates_the_name(
    text, expected_name, expected_dob
):
    result = AnswerLLMProvider().interpret(
        text, {"next_field": ["patient_identifier", "What is your name?"]}
    )
    assert result.extracted_fields.get("patient_identifier") == expected_name, text
    if expected_dob is None:
        assert "patient_age" not in result.extracted_fields, text
    else:
        assert normalized_dob_in(text) == expected_dob, text
        assert result.extracted_fields.get("patient_age") == expected_dob, text


def test_name_without_a_dob_label_stays_untouched():
    # A plain multiword name (no DOB label present) is stored exactly as
    # given, initials included.
    result = AnswerLLMProvider().interpret(
        "എൻ്റെ പേരാണ് രാജു പി കെ",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "രാജു പി കെ"


@pytest.mark.parametrize(
    "text",
    [
        "ഇരുപത് മേയ്",
        "twenty may",
        "ഇരുപത്",
        "മേയ് രണ്ടായിരം",
    ],
)
def test_incomplete_spoken_dob_is_never_guessed(text):
    parsed, _ = _extract_dob_date(text)
    assert parsed is None, text


def test_combined_malayalam_name_and_dob_extract():
    from app.providers.answer import _extract_dob_with_year


def test_combined_malayalam_name_and_dob_extract():
    from app.providers.answer import _extract_dob_with_year

    result = AnswerLLMProvider().interpret(
        "എന്റെ പേരാണ് രാജു വർഗീസ് ജന്മദിനം ഇരുപത് മെയ് രണ്ടായിരം",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )
    assert result.extracted_fields.get("patient_age") == "20-May-2000"
    assert result.extracted_fields.get("patient_identifier") == "രാജു വർഗീസ്"
    dob_value, _year = _extract_dob_with_year(
        "എന്റെ പേരാണ് രാജു വർഗീസ് ജന്മദിനം ഇരുപത് മെയ് രണ്ടായിരം"
    )
    # `_extract_dob_with_year` returns the verbatim span (not the normalized
    # date) — that span is what later name-splitting removes.
    assert dob_value == "ഇരുപത് മെയ് രണ്ടായിരം"


def test_combined_english_name_and_spoken_dob_extract():
    # The observed failure: the greedy spoken-DOB day group swallowed the
    # lead-in word ("… is twenty May two thousand" first matched with day
    # "is twenty"), so the combined answer parsed neither value. The search
    # now retries overlapping candidates, and the name cleanup then removes
    # the DOB phrase ("and my date of birth is twenty May two thousand").
    for name in ("Raju", "Raju PK"):
        text = f"My name is {name} and my date of birth is twenty May two thousand"
        parsed, verbatim = _extract_dob_date(text)
        assert parsed is not None, text
        assert normalized_dob_in(text) == "20-May-2000", text
        assert verbatim == "twenty May two thousand", text
        result = AnswerLLMProvider().interpret(
            text, {"next_field": ["patient_identifier", "What is your name?"]}
        )
        assert result.extracted_fields.get("patient_age") == "20-May-2000", text
        assert result.extracted_fields.get("patient_identifier") == name, text

# ---- mixed alphabets and digits in phone capture (reported live failure) -----


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        # Spoken words mixed with literal digits: every digit survives in
        # order, including repeats.
        ("nine 2 2 four four one nine one zero three", "9224419103"),
        ("nine two two 4 4 one nine one zero three", "9224419103"),
        # All-literal spaced digits, including a leading zero.
        ("9 2 2 4 4 1 9 1 0 3", "9224419103"),
        ("0 9 8 8 6 0 9 4 3 2 1", "09886094321"),
        # Native-script digit glyphs (Devanagari ०-९).
        ("८२२४४१८१०३", "8224418103"),
    ],
)
def test_mixed_digits_and_digit_words_extract_phone(transcript, expected):
    from app.providers.answer import _extract_spoken_phone

    assert _extract_spoken_phone(transcript) == expected


def test_mixed_script_name_wrappers_are_stripped():
    # "माझं नाव Raju Patil आहे" -> "Raju Patil": the Marathi possessive
    # lead and the trailing copula are never part of the name, while every
    # Latin and native-script letter of the name itself is preserved.
    result = AnswerLLMProvider().interpret(
        "माझं नाव Raju Patil आहे",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "Raju Patil"


def test_mixed_marathi_name_and_dob_extract_into_separate_fields():
    # The reported browser sentence: "माझं नाव राजू के आणि माझा जन्मतारीख
    # वीस जुलै दोन हजार" — the DOB-label tail (with its "आणि माझा"
    # conjunction and possessive lead) is never part of the name, so the
    # two fields extract separately and cleanly.
    result = AnswerLLMProvider().interpret(
        "माझं नाव राजू के "
        "आणि माझा जन्मतारीख "
        "वीस जुलै दोन हजार",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "राजू के"
    assert result.extracted_fields.get("patient_age") == "20-Jul-2000"


# ---- Marathi transliterated phone + month-first Marathi name/DOB --------------


def test_devanagari_asr_spelling_variants_extract_phone():
    # The reported Marathi transcript: "नाईन टू टू फोर फोर वन नाईन झीरो
    # झीरो वन" — the ि/ी matra spelling of nine and the झ spelling of
    # zero must map exactly as their established variants do.
    from app.providers.answer import _extract_spoken_phone

    assert _extract_spoken_phone(
        "नाईन टू टू फोर फोर "
        "वन नाईन झीरो झीरो वन"
    ) == "9224419001"
    assert _extract_spoken_phone(
        "nine two two four four one nine zero zero one"
    ) == "9224419001"


def test_incomplete_transliterated_number_is_never_converted():
    from app.providers.answer import _extract_spoken_phone

    # Three spoken digits: below the established minimum, nothing is
    # invented and nothing is returned.
    assert _extract_spoken_phone(
        "नाईन टू टू"
    ) is None


def test_marathi_name_lead_variants_and_month_first_dob():
    # The reported browser sentence: the masculine-possessive "माझा नाव"
    # lead must strip exactly like "माझं/माझे नाव", and the Marathi
    # month-first DOB order ("<month> <day> <year>") must yield the same
    # normalized date as the day-first order.
    from app.providers.answer import _clean_name_value

    assert _clean_name_value(
        "माझा नाव राजू के"
    ) == "राजू के"

    month_first = "माझा नाव राजू के "
    month_first += "आणि माझा जन्मतारीख "
    month_first += "जुलै वीस दोन हजार"
    result = AnswerLLMProvider().interpret(
        month_first, {"next_field": ["patient_identifier", "name and dob?"]}
    )
    assert result.extracted_fields.get("patient_identifier") == "राजू के"
    assert result.extracted_fields.get("patient_age") == "20-Jul-2000"

    # A name that merely contains a conjunction-shaped word is never
    # blindly truncated: without a DOB label the whole value is the name.
    full_name = AnswerLLMProvider().interpret(
        "राजू आणि सुनीता",
        {"next_field": ["patient_identifier", "What is your name?"]},
    )
    assert (
        full_name.extracted_fields.get("patient_identifier")
        == "राजू आणि सुनीता"
    )


# ---- phone length plausibility (reported 11-digit acceptance) -----------------


def test_marathi_eleven_digit_dictation_is_not_captured():
    # The reported transcript: "नऊ दोन दोन चार चार एक नऊ शून्य शून्य एक
    # शून्य" → an 11-digit run (a doubled digit or an extra word). The raw
    # run is still extracted faithfully, but it is never captured as a
    # candidate: the workflow asks for the number again (digit by digit,
    # bounded) instead of offering an implausible number for confirmation.
    from app.providers.answer import _extract_spoken_phone

    transcript = (
        "नऊ दोन दोन चार चार "
        "एक नऊ शून्य शून्य "
        "एक शून्य"
    )
    assert _extract_spoken_phone(transcript) == "92244190010"  # faithful run

    engine = AEStateEngine.create_case("AE-phone-11-gate")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju P K")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    result = workflow.process_turn(transcript)

    assert engine.case.contact_phone is None
    assert engine.case.confirmation_candidate == {}
    assert result.next_question is not None
    assert result.next_question[0] == "contact_phone"


@pytest.mark.parametrize(
    ("run", "plausible"),
    [
        ("9224419001", True),      # 10 digits
        ("922441900", True),       # 9 digits
        ("09886094321", True),     # 11 digits with a leading zero (STD form)
        ("92244190010", False),    # 11 digits without a leading zero
        ("922441", False),         # below the established minimum
    ],
)
def test_phone_length_plausibility(run, plausible):
    from app.providers.answer import _is_plausible_phone

    assert _is_plausible_phone(run) is plausible


def test_implausible_pending_phone_never_commits_on_confirmation():
    # An explicit confirmation of a read-back is not successful number
    # validation: a directly-seeded implausible candidate cannot be
    # committed by a bare "yes" — the number is asked for again instead.
    engine = AEStateEngine.create_case("AE-phone-11-commit-gate")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju P K")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    engine.case.pending_confirmation = "contact_phone"
    engine.case.confirmation_candidate = {"contact_phone": "92244190010"}
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )

    result = workflow.process_turn("yes")

    assert engine.case.contact_phone is None
    assert engine.case.pending_confirmation == "contact_phone"
    assert result.completed is False


def test_leading_zero_phone_confirms_and_commits():
    # The STD-prefixed form stays plausible end to end, and the committed
    # value preserves the leading zero exactly.
    engine = AEStateEngine.create_case("AE-phone-std")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    engine.update("patient_identifier", "Raju P K")
    engine.update("patient_age", "25")
    engine.update("consent_to_contact", True)
    engine.update("preferred_contact_method", "phone")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    readback = workflow.process_turn("शून्य नऊ दोन दोन चार चार एक नऊ शून्य शून्य")
    assert engine.case.confirmation_candidate["contact_phone"] == "0922441900"

    committed = workflow.process_turn("yes")

    assert engine.case.contact_phone == "0922441900"
    assert "00" in readback.response_text  # the tail is spoken for review


# ---- ambiguous Hindi date attempt keeps the name clean (reported failure) ----


def test_failed_hindi_date_attempt_never_contaminates_the_name():
    # The reported live failure: "राजू शर्मा बीस में दो हजार" — the month
    # word was misrecognised ("में" for "मई"), so the date cannot parse.
    # The day-word + year-lead tail is a failed date attempt: it is
    # stripped from the name candidate, and the DOB stays unset for a
    # bounded clarification — nothing is invented.
    result = AnswerLLMProvider().interpret(
        "राजू शर्मा बीस में दो हजार",
        {"next_field": ["patient_identifier", "name and dob?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "राजू शर्मा"
    assert "patient_age" not in result.extracted_fields

    # The DOB retry of the same ambiguous phrase stays unparseable: no
    # date is invented from it.
    retry = AnswerLLMProvider().interpret(
        "बीस में दो हजार",
        {"next_field": ["patient_age", "What is your date of birth?"]},
    )
    assert "patient_age" not in retry.extracted_fields


def test_clear_hindi_name_and_dob_still_extract_together():
    # The clean spelling (मई = May) parses normally: both fields extract
    # separately from the same sentence.
    result = AnswerLLMProvider().interpret(
        "राजू शर्मा बीस मई दो हजार",
        {"next_field": ["patient_identifier", "name and dob?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "राजू शर्मा"
    assert result.extracted_fields.get("patient_age") == "20-May-2000"


def test_workflow_keeps_name_and_reasks_dob_after_a_failed_date():
    # Workflow level: the ambiguous answer keeps the held name candidate
    # clean and the pending DOB question is re-asked for clarification.
    from app.workflow.ae_engine import AEStateEngine
    from app.workflow.service.conversational import (
        ConversationalAEWorkflow,
        VOICE_CONFIRMATION_FIELDS,
    )

    engine = AEStateEngine.create_case("AE-ambiguous-dob")
    engine.update("product", "Lantus SoloStar")
    engine.update("adverse_event", "rash")
    engine.update("reporter_role", "self")
    workflow = ConversationalAEWorkflow(
        engine, AnswerLLMProvider(), confirm_fields=VOICE_CONFIRMATION_FIELDS
    )
    workflow.process_turn(
        "राजू शर्मा बीस में दो हजार"
    )
    assert engine.case.patient_identifier is None  # held for the read-back
    assert engine.case.confirmation_candidate["patient_identifier"] == (
        "राजू शर्मा"
    )
    assert engine.case.patient_age is None


# ---- English name capitalization (structured-field normalization) ------------


def test_latin_name_words_are_capitalized_in_the_structured_field():
    # The reported display issue: a lower-case spoken English name is
    # capitalized per word in the STRUCTURED field (the confirmation
    # candidate, the read-back and the report show the capitalized form);
    # the provider layer keeps the cleaned raw wording and the raw
    # transcript is never touched.
    result = AnswerLLMProvider().interpret(
        "raju sharma", {"next_field": ["patient_identifier", "What is your name?"]}
    )
    assert result.extracted_fields.get("patient_identifier") == "raju sharma"

    engine = AEStateEngine.create_case("AE-titlecase-name")
    engine.update("patient_identifier", "raju sharma")
    assert engine.case.patient_identifier == "Raju Sharma"

    engine2 = AEStateEngine.create_case("AE-titlecase-confirm")
    engine2.begin_confirmation("patient_identifier", "raju sharma")
    assert engine2.case.confirmation_candidate["patient_identifier"] == "Raju Sharma"


def test_internal_capitalization_and_native_script_are_preserved():
    from app.providers.answer import _clean_name_value

    # Meaningful internal capitalization is never rewritten.
    assert _clean_name_value("Eric McDonald") == "Eric McDonald"
    assert _clean_name_value("McDonald") == "McDonald"
    # The engine's structured value keeps internal caps and non-Latin
    # scripts byte-for-byte (no case mapping exists for them).
    engine = AEStateEngine.create_case("AE-titlecase-mixed")
    engine.update("patient_identifier", "Eric McDonald")
    assert engine.case.patient_identifier == "Eric McDonald"
    engine2 = AEStateEngine.create_case("AE-titlecase-deva")
    native = "राजू शर्मा"
    engine2.update("patient_identifier", native)
    assert engine2.case.patient_identifier == native


def test_reported_nukta_spelling_name_preserved_dob_unset():
    # The exact reported turn-4 transcript (the nukta spelling हज़ार): the
    # name candidate stays राजू करम and no date is invented.
    result = AnswerLLMProvider().interpret(
        "राजू करम बीस मे "
        "दो हज़ार",
        {"next_field": ["patient_identifier", "name and dob?"]},
    )
    assert result.extracted_fields.get("patient_identifier") == "राजू करम"
    assert "patient_age" not in result.extracted_fields


def test_failed_dob_attempt_shape_detector():
    from app.providers.answer import _ends_with_failed_date_attempt

    # The recognizable unparseable shape (day word + में + year lead).
    assert _ends_with_failed_date_attempt(
        "बीस में दो हजार"
    ) is True
    # A parseable date is not a failed attempt.
    assert _ends_with_failed_date_attempt(
        "बीस मई दो हजार"
    ) is False
    # An unrelated name/utterance never matches.
    assert _ends_with_failed_date_attempt("राजू शर्मा") is False


def test_hindi_at_rate_spelling_extracts_email():
    # The reported Hindi transcript: the at-rate cue without the candrabindu
    # ("एट द रेट") must extract exactly like the Marathi spelling.
    from app.providers.answer import _normalize_spoken_email

    hindi = "एस आर ई ई एच एट द रेट जीमेल डॉट कॉम"
    marathi = "एस आर ई ई एच अॅट द रेट जीमेल डॉट कॉम"
    assert _normalize_spoken_email(hindi) == "sreeh@gmail.com"
    assert _normalize_spoken_email(marathi) == "sreeh@gmail.com"
    # The Malayalam and Latin spellings are covered by the existing
    # dictation-variant tests (test_all_three_sreehari_dictation_variants_parse).
    assert _normalize_spoken_email("S R E E H at the rate gmail dot com") == "sreeh@gmail.com"
