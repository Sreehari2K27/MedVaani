"""Tests for automatic spoken-language mirroring.

Unit tests cover offline language identification and translation; endpoint
tests cover session language persistence, the STT language switch, the Timbre
language/voice selection, and the English fallback for languages without a
translation catalog.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.language import (
    detect_language,
    normalize_language,
    translate,
    translation_available,
    voice_for,
)
from app.language.catalog import DEFAULT_LANGUAGE
from app.providers.base import LLMResult
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    ConversationalAEWorkflow,
    EVENT_DESCRIPTION_QUESTION,
    INTRODUCTION_MESSAGE,
)

HINDI_PRODUCT_QUESTION = "आपको क्या लगता है कि किस दवा या उत्पाद से यह हुआ हो सकता है?"


class _NeutralProvider:
    def interpret(self, message, context=None):
        return LLMResult(intent="unknown", extracted_fields={})


# ---- identification ----------------------------------------------------------


def test_english_latin_is_english():
    assert detect_language("I have had a headache since yesterday.") == "en-IN"


def test_devanagari_is_hindi():
    assert detect_language("मुझे कल से सिर में दर्द है।") == "hi-IN"


def test_devanagari_marathi_markers_resolve_to_marathi():
    assert detect_language("मला कालपासून डोके दुखत आहे आणि ताप आहे.") == "mr-IN"


def test_other_indic_scripts_map_to_their_language():
    assert detect_language("எனக்கு தலைவலி உள்ளது") == "ta-IN"
    assert detect_language("నాకు తలనొప్పి ఉంది") == "te-IN"
    assert detect_language("ನನಗೆ ತಲೆನೋವು ಇದೆ") == "kn-IN"
    assert detect_language("എനിക്ക് തലവേദനയുണ്ട്") == "ml-IN"
    assert detect_language("আমার মাথা ব্যথা করছে") == "bn-IN"


def test_romanized_hindi_is_detected():
    assert detect_language("mujhe kal se bahut dard ho raha hai") == "hi-IN"


def test_provider_language_wins_over_transcript():
    assert detect_language("anything at all", "ta-IN") == "ta-IN"


def test_unidentifiable_text_returns_none():
    assert detect_language("") is None
    assert detect_language("    ") is None
    assert detect_language("12345") is None


def test_romanized_hindi_conversational_words_are_detected():
    assert detect_language("haan") == "hi-IN"
    assert detect_language("gambhir") == "hi-IN"
    assert detect_language("sahmat") == "hi-IN"


def test_short_markerless_fragments_keep_the_current_language():
    # Names and short yes/no replies carry no language evidence: they must
    # not flip an established session (e.g. Hindi) back to English.
    assert detect_language("myself") is None
    assert detect_language("Demo Patient") is None
    assert detect_language("yes that is fine") is None
    assert detect_language("I am 45 years old") is None
    # Full English sentences still detect as English.
    assert detect_language("I have had a headache since yesterday.") == "en-IN"


def test_normalize_and_voice_mapping():
    assert normalize_language("hi") == "hi-IN"
    assert normalize_language("hi-IN") == "hi-IN"
    assert normalize_language("xx-XX") is None
    assert voice_for("hi-IN") == "Nalini"
    assert voice_for("hi-en") == "Kaveri"
    assert voice_for("xx-XX") == "Kaveri"


# ---- translation -------------------------------------------------------------


def test_hindi_question_translates():
    result = translate(
        "Which medicine or product do you think may have caused this?", "hi-IN"
    )
    assert result.translated is True
    assert result.language == "hi-IN"
    assert result.text == HINDI_PRODUCT_QUESTION


def test_readback_keeps_structured_value_untranslated():
    result = translate("I heard Rahul Sharma. Is that correct?", "hi-IN")
    assert result.translated is True
    assert "Rahul Sharma" in result.text
    assert "क्या यह सही है?" in result.text


def test_unknown_sentence_falls_back_to_english_wholesale():
    text = (
        "This is a product information request. "
        "It will be routed to the MI/RAG layer (not implemented yet)."
    )
    result = translate(text, "hi-IN")
    assert result.translated is False
    assert result.language == DEFAULT_LANGUAGE
    assert result.text == text


def test_english_is_passthrough():
    result = translate("Take care.", "en-IN")
    assert result.translated is False
    assert result.text == "Take care."


def test_language_without_catalog_falls_back():
    result = translate("Take care.", "ta-IN")
    assert result.translated is False
    assert result.language == DEFAULT_LANGUAGE
    assert translation_available("hi-IN") is True
    assert translation_available("ta-IN") is False


def test_urgent_safety_message_translates():
    engine = AEStateEngine.create_case("AE-lang-urgent")
    workflow = ConversationalAEWorkflow(engine, _NeutralProvider())
    result = workflow.process_turn("I am having severe difficulty breathing")
    assert result.safety_status == "urgent_life_threatening"
    spoken = translate(result.response_text, "hi-IN")
    assert spoken.translated is True
    assert spoken.text != result.response_text


# ---- endpoint mirroring ------------------------------------------------------


class _ScriptSTT:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)
        self.languages: list[str] = []

    def transcribe(self, audio_path, language_code="en-IN"):
        self.languages.append(language_code)
        return SimpleNamespace(text=self._transcripts.pop(0))


class _RecordingTTS:
    def __init__(self):
        self.calls: list[dict] = []

    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        self.calls.append({"text": text, "language": language, "voice": voice})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake")
        return Path(output_path)


def _client(stt, tts):
    router = create_voice_conversation_router(
        stt=stt,
        tts=tts,
        session_store=SessionStore(),
        llm_provider=_NeutralProvider(),
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, session_id, language="en-IN"):
    resp = client.post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"fake-audio", "audio/wav")},
        data={"session_id": session_id, "language": language},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_hindi_conversation_mirrors_language_and_voice():
    stt = _ScriptSTT(["मुझे कल से सिर में दर्द है।", "अब थोड़ा ठीक हूँ।"])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    first = _post(client, "hindi-1")
    # The first reply is the approved opening (Step 1 introduction + the
    # event-description invitation); UI text stays English while the spoken
    # text is Hindi and audio uses the Hindi voice.
    assert first["response_text"].startswith(INTRODUCTION_MESSAGE)
    assert first["response_text"].endswith(
        "Please tell me what happened, in your own words."
    )
    assert first["next_question"] == ["adverse_event", EVENT_DESCRIPTION_QUESTION]
    assert first["conversation_language"] == "hi-IN"
    assert first["language_translated"] is True
    assert first["spoken_response_text"] != first["response_text"]
    assert first["voice"] == "Nalini"
    assert tts.calls[-1]["language"] == "hi-IN"
    assert tts.calls[-1]["voice"] == "Nalini"
    # First turn uses the hint; the session language then drives STT.
    assert stt.languages[0] == "en-IN"

    _post(client, "hindi-1")
    assert stt.languages[1] == "hi-IN"


def test_english_conversation_stays_english():
    stt = _ScriptSTT(["I need to report something."])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    data = _post(client, "english-1")
    assert data["conversation_language"] == "en-IN"
    assert data["language_translated"] is False
    assert data["spoken_response_text"] == data["response_text"]
    assert data["voice"] is None
    assert tts.calls[-1]["language"] == "en-IN"
    assert tts.calls[-1]["voice"] is None


def test_unsupported_translation_language_falls_back_to_english_audio():
    stt = _ScriptSTT(["எனக்கு தலைவலி உள்ளது", "இன்னும் உள்ளது"])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    first = _post(client, "tamil-1")
    assert first["conversation_language"] == "ta-IN"
    assert first["language_translated"] is False
    assert first["spoken_response_text"] == first["response_text"]
    assert tts.calls[-1]["language"] == "en-IN"
    assert tts.calls[-1]["voice"] is None

    _post(client, "tamil-1")
    # The transcript language still drives STT for the next turn.
    assert stt.languages[1] == "ta-IN"


def test_language_switches_when_a_different_language_is_detected():
    stt = _ScriptSTT(["எனக்கு தலைவலி உள்ளது", "Now I am speaking in English."])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    assert _post(client, "switch-1")["conversation_language"] == "ta-IN"
    assert _post(client, "switch-1")["conversation_language"] == "en-IN"


def test_english_start_then_hindi_turn_speaks_reply_in_hindi():
    # English first turn, then Hindi: the session must switch and every spoken
    # reply afterwards (including the short product clarification, whose
    # sentences are newly catalogued) must be Hindi audio, not English.
    stt = _ScriptSTT(
        [
            "I need to report a side effect",
            "mujhe bahut dard ho raha hai",
            "nahi theek hua",
        ]
    )
    tts = _RecordingTTS()
    client = _client(stt, tts)

    first = _post(client, "switch-hi-2")
    assert first["conversation_language"] == "en-IN"
    assert first["language_translated"] is False
    assert tts.calls[-1]["language"] == "en-IN"

    second = _post(client, "switch-hi-2")
    assert second["conversation_language"] == "hi-IN"
    assert second["language_translated"] is True
    assert second["spoken_response_text"] != second["response_text"]
    assert tts.calls[-1]["language"] == "hi-IN"
    assert tts.calls[-1]["voice"] == "Nalini"

    # The switch sticks: STT and the next clarification reply stay Hindi.
    third = _post(client, "switch-hi-2")
    assert stt.languages[2] == "hi-IN"
    assert third["conversation_language"] == "hi-IN"
    assert third["language_translated"] is True
    assert tts.calls[-1]["language"] == "hi-IN"
    assert tts.calls[-1]["voice"] == "Nalini"


def test_romanized_hindi_turn_mirrors_as_hindi():
    stt = _ScriptSTT(["mujhe kal se bahut dard ho raha hai"])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    data = _post(client, "roman-1")
    assert data["conversation_language"] == "hi-IN"
    assert data["language_translated"] is True
    assert tts.calls[-1]["language"] == "hi-IN"


def test_short_replies_do_not_revert_an_established_hindi_session():
    stt = _ScriptSTT(
        ["मुझे कल से सिर में दर्द है।", "haan", "myself", "yes that is fine"]
    )
    tts = _RecordingTTS()
    client = _client(stt, tts)

    assert _post(client, "keep-hi-1")["conversation_language"] == "hi-IN"
    for _ in range(3):
        data = _post(client, "keep-hi-1")
        assert data["conversation_language"] == "hi-IN"
        assert data["language_translated"] is True
    # STT keeps being driven by the session language, not the en-IN hint.
    assert stt.languages == ["en-IN", "hi-IN", "hi-IN", "hi-IN"]


def test_explicit_language_hint_is_used_for_stt_and_kept_for_unidentifiable_turns():
    stt = _ScriptSTT(["12345"])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    data = _post(client, "explicit-hi", language="hi-IN")

    assert stt.languages == ["hi-IN"]
    assert data["conversation_language"] == "hi-IN"


def test_confirmation_review_notice_translates_to_hindi():
    from app.workflow.service.conversational import CONFIRMATION_REVIEW_NOTICE

    result = translate(CONFIRMATION_REVIEW_NOTICE, "hi-IN")

    # All-or-nothing: every sentence of the notice must be covered.
    assert result.translated is True
    assert result.language == "hi-IN"
    assert "Thank you for your patience" not in result.text


def test_confirmation_review_notice_is_honest_about_missing_details():
    from app.workflow.service.conversational import CONFIRMATION_REVIEW_NOTICE

    lower = CONFIRMATION_REVIEW_NOTICE.lower()
    # Must not claim everything was saved, that the report is complete, or
    # that anything was submitted.
    assert "saved everything" not in lower
    assert "everything you've told me" not in lower
    assert "complete" not in lower
    assert "submit" not in lower
    # It explains why review is needed and that a human will review.
    assert "could not confirm" in lower
    assert "reviewed by a member of our team" in lower


def test_hindi_confirmation_review_notice_matches_the_honest_wording():
    from app.workflow.service.conversational import CONFIRMATION_REVIEW_NOTICE

    result = translate(CONFIRMATION_REVIEW_NOTICE, "hi-IN")

    assert result.translated is True
    assert result.language == "hi-IN"
    # No English saved-everything claim leaks through, and the Hindi wording
    # states that details could not be confirmed and need review.
    assert "saved everything" not in result.text
    assert "Thank you for your patience" not in result.text
    assert "पुष्टि" in result.text
    assert "समीक्षा" in result.text


def test_first_turn_hindi_hint_keeps_stt_and_tts_in_hindi():
    # Clean Hindi session: the client's locale hint is hi-IN from the very
    # first turn, so the English model is never used for transcription.
    stt = _ScriptSTT(["मुझे कल से सिर में दर्द है।", "अब थोड़ा ठीक हूँ।"])
    tts = _RecordingTTS()
    client = _client(stt, tts)

    first = _post(client, "hindi-clean-1", language="hi-IN")
    assert first["conversation_language"] == "hi-IN"
    assert first["language_translated"] is True
    # The first reply is the approved opening (Step 1 introduction + the
    # event-description invitation), spoken in Hindi with the Hindi voice.
    assert first["response_text"].startswith(INTRODUCTION_MESSAGE)
    assert first["response_text"].endswith(
        "Please tell me what happened, in your own words."
    )
    assert first["spoken_response_text"] != first["response_text"]
    assert first["voice"] == "Nalini"
    assert tts.calls[-1]["language"] == "hi-IN"
    assert tts.calls[-1]["voice"] == "Nalini"

    second = _post(client, "hindi-clean-1", language="hi-IN")
    assert second["conversation_language"] == "hi-IN"
    # Every STT call used the Hindi language code; en-IN never appears.
    assert stt.languages == ["hi-IN", "hi-IN"]
    assert tts.calls[-1]["language"] == "hi-IN"
