"""Focused tests for the AE email-capture, verification and consent integrity.

Follow-up consent, email availability and email verification are three
independent states: an email reply can never change consent, a rejected or
unprovided email never blocks the optional step, and an unconfirmed address
is never reported as verified. All data is synthetic; all providers are
offline stubs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.voice_conversation import create_voice_conversation_router
from app.language import detect_language, translate
from app.providers.answer import AnswerLLMProvider
from app.session import SessionStore
from app.workflow.ae_engine import AEStateEngine
from app.workflow.service.conversational import (
    EMAIL_OFFER_QUESTION,
    EMAIL_REQUEST_QUESTION,
    SEVERITY_QUESTION,
    VOICE_CONFIRMATION_FIELDS,
    ConversationalAEWorkflow,
)


def _workflow(case_id: str):
    engine = AEStateEngine.create_case(case_id)
    workflow = ConversationalAEWorkflow(
        engine,
        AnswerLLMProvider(),
        confirm_fields=VOICE_CONFIRMATION_FIELDS,
    )
    return engine, workflow


def _to_email_offer(case_id: str):
    """Workflow with consent granted and the Step 8 email offer pending."""
    engine, workflow = _workflow(case_id)
    workflow.process_turn("I have a headache after taking Lantus")
    workflow.process_turn("yes")  # Step 4
    workflow.process_turn("myself")
    workflow.process_turn("Alpha Jones, 05-01-1990")
    workflow.process_turn("yes")
    workflow.process_turn("9886094321")
    workflow.process_turn("yes")
    consented = workflow.process_turn("Yes, I consent")
    assert engine.case.consent_to_contact is True
    assert consented.next_question is not None
    assert consented.next_question[0] == "contact_email"
    return engine, workflow


def _email_to_readback(workflow):
    offered = workflow.process_turn("yes")
    assert offered.next_question is not None
    assert offered.next_question[1] == EMAIL_REQUEST_QUESTION
    return workflow.process_turn("alpha@example.com")


MALAYALAM_EMAIL_DICTATION = (
    "ആർ എ ജെ യു പി കെ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
)


def test_malayalam_dictated_email_is_read_back_and_committed_after_yes():
    # The observed browser dictation, spoken with Malayalam letter-name
    # spellings ("R A J U P K at the rate Gmail dot com").
    engine, workflow = _to_email_offer("em-ml-dictation")

    dictated = workflow.process_turn(MALAYALAM_EMAIL_DICTATION)

    assert engine.case.contact_email is None  # held for the read-back
    assert "rajupk@gmail.com" in dictated.next_question[1]

    committed = workflow.process_turn("അതെ")
    assert engine.case.contact_email == "rajupk@gmail.com"
    assert engine.case.consent_to_contact is True


def test_malayalam_negation_leads_the_corrected_address():
    # "ഇല്ല ആർ … കോം" is a correction carrying the dictated address, never
    # a decline of the optional email (this is the observed second turn).
    engine, workflow = _to_email_offer("em-ml-negation")

    dictated = workflow.process_turn("ഇല്ല " + MALAYALAM_EMAIL_DICTATION)

    assert engine.case.contact_email_declined is False
    assert engine.case.contact_email is None
    assert "rajupk@gmail.com" in dictated.next_question[1]

    workflow.process_turn("ശരിയാണ്")
    assert engine.case.contact_email == "rajupk@gmail.com"


def test_plain_malayalam_no_still_declines_the_email_offer():
    engine, workflow = _to_email_offer("em-ml-decline")
    declined = workflow.process_turn("ഇല്ല")
    assert engine.case.contact_email_declined is True
    assert engine.case.contact_email is None
    assert declined.next_question is not None
    assert declined.next_question[0] == "seriousness"


def test_malayalam_dictation_without_an_at_cue_is_never_accepted():
    # Incomplete/malformed dictation: no "@" cue means no address is
    # assembled and nothing is stored or declined.
    engine, workflow = _to_email_offer("em-ml-malformed")
    result = workflow.process_turn(
        "ആർ എ ജെ യു പി കെ ജിമെയിൽ ഡോട്ട് കോം"
    )
    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation is None
    assert result.next_question is not None


SREEHARI_LOCAL = "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ"


def test_all_three_sreehari_dictation_variants_parse():
    # The observed transcript's three variants: the full spelling, the
    # shorter (letter-missing) retry, and the split "ജി മെയിൽ" form. Each
    # yields exactly what its letters spell - nothing is invented, so the
    # letter-missing retry is NOT the same address as the full one.
    from app.providers.answer import _normalize_spoken_email

    assert _normalize_spoken_email(SREEHARI_LOCAL + " അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം") == "sreehari@gmail.com"
    assert _normalize_spoken_email("എസ് ആർ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം") == "srehari@gmail.com"
    assert _normalize_spoken_email(SREEHARI_LOCAL + " അറ്റ് ദ റേറ്റ് ജി മെയിൽ ഡോട്ട് കോം") == "sreehari@gmail.com"


def test_re_dictation_during_email_readback_updates_the_candidate():
    # Retry-resilient correction: redictating the address while its read-back
    # is pending replaces the candidate (mirrors the corrected-phone
    # pattern); the explicit confirmation of the new candidate is still
    # required before anything is committed.
    engine, workflow = _to_email_offer("em-ml-redictate")

    first = workflow.process_turn(SREEHARI_LOCAL + " അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം")
    assert engine.case.contact_email is None
    assert "sreehari@gmail.com" in first.next_question[1]

    re_dictated = workflow.process_turn(
        "ഇല്ല " + SREEHARI_LOCAL + " അറ്റ് ദ റേറ്റ് ജി മെയിൽ ഡോട്ട് കോം"
    )
    assert engine.case.contact_email is None  # still unconfirmed
    assert "sreehari@gmail.com" in re_dictated.next_question[1]

    committed = workflow.process_turn("അതെ")
    assert engine.case.contact_email == "sreehari@gmail.com"
    assert engine.case.consent_to_contact is True


# ---- repeated letters and rejected-candidate replacement (reported defects) ---


def test_latin_letter_name_spelling_preserves_repeated_letters():
    # "S R E E H A R I" spells sreehari: repeated letter names are preserved
    # in order, in both spaced and comma-separated dictation forms.
    from app.providers.answer import _normalize_spoken_email

    assert (
        _normalize_spoken_email("S R E E H A R I at the rate gmail dot com")
        == "sreehari@gmail.com"
    )
    assert (
        _normalize_spoken_email("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com")
        == "sreehari@gmail.com"
    )


def test_reported_malayalam_spelling_keeps_both_e_letters():
    # The reported browser transcript: both ഇ (E) tokens are retained, so the
    # full spelling is sreehari@gmail.com. When the transcription itself
    # omits one E (the observed letter-missing retry), nothing is invented:
    # the address spells exactly what was heard (srehari@gmail.com).
    from app.providers.answer import _normalize_spoken_email

    full = "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    missing = "എസ് ആർ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"

    assert _normalize_spoken_email(full) == "sreehari@gmail.com"
    assert _normalize_spoken_email(missing) == "srehari@gmail.com"


def test_rejected_wrong_candidate_is_replaced_and_only_new_confirmed_commits():
    # A letter-missing first spelling is read back, rejected, and re-spelled
    # in full: the rejected candidate is never committed, the replacement is
    # read back accurately, and only the explicitly confirmed NEW candidate
    # is committed.
    engine, workflow = _to_email_offer("em-ml-reject-replace")

    first = workflow.process_turn(
        "എസ് ആർ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    )
    assert engine.case.contact_email is None  # candidate only, unconfirmed
    assert "srehari@gmail.com" in first.next_question[1]

    workflow.process_turn("ഇല്ല")  # rejection: the wrong candidate is discarded

    replaced = workflow.process_turn(
        "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    )
    assert engine.case.contact_email is None  # still unconfirmed
    assert "sreehari@gmail.com" in replaced.next_question[1]
    assert "srehari" not in replaced.next_question[1]

    committed = workflow.process_turn("അതെ")

    assert engine.case.contact_email == "sreehari@gmail.com"
    assert engine.case.contact_email != "srehari@gmail.com"
    assert engine.case.consent_to_contact is True


# ---- localized email read-back (reported mixed-language defect) ---------------


def _malayalam_readback(case_id: str, dictation: str):
    engine, workflow = _to_email_offer(case_id)
    workflow.process_turn("അതെ")  # accept the offer: the request is asked
    readback = workflow.process_turn(dictation)
    assert readback.next_question is not None
    assert readback.next_question[0] == "confirmation"
    return engine, workflow, readback


def test_malayalam_email_readback_is_fully_malayalam():
    # The email read-back must be fully localized: no English template
    # fragment ("your email address as") may survive inside the Malayalam
    # sentence; only the address itself stays Latin.
    _, _, readback = _malayalam_readback(
        "em-ml-readback", "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    )
    assert "I heard your email address as sreehari@gmail.com." in readback.response_text

    localized = translate(readback.response_text, "ml-IN")

    assert localized.translated is True
    assert "your email address" not in localized.text
    assert "sreehari@gmail.com" in localized.text
    assert "അത് ശരിയാണോ?" in localized.text


def test_malayalam_email_readback_stays_malayalam_after_rejection_and_reentry():
    # The same localization applies to the second read-back after a rejected
    # candidate is re-entered.
    engine, workflow, first = _malayalam_readback(
        "em-ml-readback-retry",
        "എസ് ആർ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം",
    )
    localized_first = translate(first.response_text, "ml-IN")
    assert localized_first.translated is True
    assert "your email address" not in localized_first.text

    workflow.process_turn("ഇല്ല")
    replaced = workflow.process_turn(
        "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    )

    localized_second = translate(replaced.response_text, "ml-IN")

    assert localized_second.translated is True
    assert "your email address" not in localized_second.text
    assert "sreehari@gmail.com" in localized_second.text


def test_english_email_readback_is_unchanged_passthrough():
    engine, workflow = _to_email_offer("em-en-readback")
    readback = _email_to_readback(workflow)
    assert "I heard your email address as alpha@example.com." in readback.response_text

    passthrough = translate(readback.response_text, "en-IN")

    assert passthrough.translated is False
    assert passthrough.text == readback.response_text


def test_spelled_email_content_is_never_language_evidence():
    # English letter names, "@" cues and domain words inside a dictated
    # address are field content: they never switch the session language
    # (endpoint-level stability is covered by the language regression tests).
    assert (
        detect_language("S R E E H A R I at the rate gmail dot com") is None
    )
    assert (
        detect_language("എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം")
        == "ml-IN"
    )


def test_repeated_letters_in_other_positions_are_preserved():
    # Multiplicity is preserved wherever the repeat falls: a doubled letter
    # at the end (TESS) and in the middle (FOOT) keep both letters, and the
    # read-back speaks the same value without collapsing it.
    from app.providers.answer import _normalize_spoken_email

    assert _normalize_spoken_email("T E S S at the rate gmail dot com") == (
        "tess@gmail.com"
    )
    assert _normalize_spoken_email("F O O T at the rate example dot com") == (
        "foot@example.com"
    )


def test_double_e_raw_transcripts_are_preserved_exactly():
    # The observed live sequence: when the transcript carries both E tokens
    # the address is captured exactly ("S R E E H" -> sreeh); when it
    # carries only one, the address spells exactly what was heard and no
    # letter is invented. The loss, when it happens, is upstream of the
    # application (STT/articulation) - never the parser.
    from app.providers.answer import _normalize_spoken_email

    assert _normalize_spoken_email("S R E E H at the rate gmail dot com") == (
        "sreeh@gmail.com"
    )
    assert _normalize_spoken_email("S R E H at the rate gmail dot com") == (
        "sreh@gmail.com"
    )
    assert _normalize_spoken_email("S R E E H @ GMAIL.COM") == "sreeh@gmail.com"


def test_double_e_address_survives_readback_and_snapshot():
    # End to end: the exact double-E transcript survives extraction, is
    # read back byte-for-byte as the candidate, and only its explicit
    # confirmation stores it in the report snapshot.
    engine, workflow = _to_email_offer("em-double-e")

    offered_yes = workflow.process_turn("yes")
    assert offered_yes.next_question is not None

    readback = workflow.process_turn("S R E E H at the rate gmail dot com")
    assert engine.case.contact_email is None
    assert "sreeh@gmail.com" in readback.next_question[1]

    committed = workflow.process_turn("yes")

    assert engine.case.contact_email == "sreeh@gmail.com"
    assert workflow._collected_snapshot()["contact_email"] == "sreeh@gmail.com"
    # Read-back and committed snapshot agree exactly.
    assert "sreeh@gmail.com" in readback.next_question[1]


# ---- Malayalam-script "yes" transliterations (reported live failure) ---------


def test_malayalam_script_yes_confirms_the_exact_read_back():
    # The browser STT transliterates spoken "yes" into Malayalam script
    # ("യെസ്"); it must confirm the pending read-back and commit exactly
    # the displayed candidate - never a guessed or reconstructed address.
    # The observed live transcript variants are anchored as well.
    from app.providers.answer import _normalize_spoken_email

    assert _normalize_spoken_email("S R E H at the rate gmail dot com") == (
        "sreh@gmail.com"
    )
    assert _normalize_spoken_email("S R E E H I at the rate gmail dot com") == (
        "sreehi@gmail.com"
    )

    engine, workflow = _to_email_offer("em-ml-script-yes")
    readback = _email_to_readback(workflow)
    assert "alpha@example.com" in readback.next_question[1]

    confirmed = workflow.process_turn("യെസ്")

    assert engine.case.contact_email == "alpha@example.com"
    assert workflow._collected_snapshot()["contact_email"] == "alpha@example.com"
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_compound_malayalam_script_yes_also_confirms():
    # "യെസ് ശരിയാണ്" ("yes, that's correct") is an affirmation led by the
    # transliteration: its lead is shed and the confirmation goes through.
    engine, workflow = _to_email_offer("em-ml-script-yes-compound")
    readback = _email_to_readback(workflow)
    assert "alpha@example.com" in readback.next_question[1]

    confirmed = workflow.process_turn("യെസ് ശരിയാണ്")

    assert engine.case.contact_email == "alpha@example.com"
    assert workflow._collected_snapshot()["contact_email"] == "alpha@example.com"


def test_empty_transcript_at_email_confirmation_keeps_the_read_back():
    # "No speech detected" at the email confirmation: an empty STT capture
    # changes nothing (the pending read-back is re-asked, nothing is
    # committed, no attempt is consumed) and the conversation stays
    # recoverable for the reporter's next attempt.
    store = SessionStore()
    client = _voice_client(
        _QueueSTT(EMAIL_JOURNEY + ["S R E E H I at the rate gmail dot com", ""]),
        store,
    )
    session_id = "em-empty-confirm"

    for _ in range(len(EMAIL_JOURNEY) + 1):
        ok = _post(client, session_id)
        assert ok.status_code == 200

    session = store.get_or_create(session_id)
    assert session.ae_engine.case.pending_confirmation == "contact_email"
    assert session.ae_engine.case.contact_email is None

    empty = _post(client, session_id)  # the reported empty attempt
    assert empty.status_code == 200
    body = empty.json()
    assert body["reason"] == "empty_transcript"
    assert body["next_question"] is not None
    assert body["next_question"][0] == "confirmation"
    final = store.get_or_create(session_id)
    assert final.ae_engine.case.contact_email is None
    assert final.ae_engine.case.pending_confirmation == "contact_email"


def test_compact_domain_after_the_at_cue_keeps_repeated_letters():
    # The compact symbol form ("@ GMAIL.COM" / "at gmail.com") is accepted,
    # and repeated letters still survive exactly as dictated; a missing
    # letter is never invented.
    from app.providers.answer import _normalize_spoken_email

    assert _normalize_spoken_email("S R E E H A R I @ GMAIL.COM") == (
        "sreehari@gmail.com"
    )
    assert _normalize_spoken_email("s r e e h a r i at gmail.com") == (
        "sreehari@gmail.com"
    )
    assert _normalize_spoken_email("S R E H A R I @ GMAIL.COM") == (
        "srehari@gmail.com"
    )


def test_compact_form_is_read_back_before_any_confirmation():
    engine, workflow = _to_email_offer("em-compact-readback")

    offered_yes = workflow.process_turn("yes")
    assert offered_yes.next_question is not None

    readback = workflow.process_turn("S R E E H A R I @ GMAIL.COM")

    assert engine.case.contact_email is None  # held for the read-back
    assert readback.next_question is not None
    assert readback.next_question[0] == "confirmation"
    assert "sreehari@gmail.com" in readback.next_question[1]


def test_latin_rejected_candidate_stores_only_the_confirmed_address():
    # The reported Latin sequence: a one-E transcript reads back srehari,
    # is rejected, the same spelling reads back the same value, a bare
    # affirmative cannot confirm it, and only the full spelling commits -
    # with the read-back, committed value and report snapshot agreeing.
    engine, workflow = _to_email_offer("em-latin-reject-flow")
    missing = "S R E H A R I at the rate gmail dot com"
    full = "S R E E H A R I at the rate gmail dot com"

    first = workflow.process_turn(missing)
    assert "srehari@gmail.com" in first.next_question[1]
    assert engine.case.contact_email is None

    workflow.process_turn("no")

    again = workflow.process_turn(missing)
    assert "srehari@gmail.com" in again.next_question[1]  # unchanged candidate

    blocked = workflow.process_turn("yes")
    assert engine.case.contact_email is None  # rejected value never commits
    assert engine.case.pending_confirmation == "contact_email"
    assert "The address still reads as srehari@gmail.com." in blocked.response_text
    assert "spell it out again" in blocked.response_text

    fresh = workflow.process_turn(full)
    assert "sreehari@gmail.com" in fresh.next_question[1]  # the NEW candidate

    committed = workflow.process_turn("yes")

    assert engine.case.contact_email == "sreehari@gmail.com"
    assert engine.case.contact_email != "srehari@gmail.com"
    # The report snapshot stores the latest explicitly confirmed candidate;
    # read-back, committed value and snapshot agree.
    assert workflow._collected_snapshot()["contact_email"] == "sreehari@gmail.com"
    assert engine.case.confirmation_candidate == {}


def test_truncated_respelling_after_rejection_needs_explicit_completeness():
    # The reported sequence: the first spelling is read back as
    # srehari@gmail.com (the transcription already dropped one letter) and
    # is rejected; the re-spelling is shorter and is a strict prefix of the
    # rejected address (the transcription dropped more). A bare affirmative
    # must NOT commit it: the workflow names the value, asks whether it is
    # complete, and only the reporter's explicit resolution (a further
    # affirmative after that question) commits anything.
    engine, workflow = _to_email_offer("em-truncated-prefix")
    first = "S R E H A R I at the rate gmail dot com"
    short = "S R E H at the rate gmail dot com"

    workflow.process_turn(first)
    workflow.process_turn("no")  # rejection of srehari@gmail.com
    assert workflow._rejected_email_candidate == "srehari@gmail.com"

    again = workflow.process_turn(short)
    assert "sreh@gmail.com" in again.next_question[1]
    assert engine.case.contact_email is None

    blocked = workflow.process_turn("yes")
    # The first affirmative cannot commit the possibly truncated value.
    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    assert "The address reads as sreh@gmail.com." in blocked.response_text
    assert "If that is the complete address, say yes" in blocked.response_text

    # The explicit resolution after the question commits the exact value
    # that was read back - the rejected value never reappears.
    resolved = workflow.process_turn("yes")

    assert engine.case.contact_email == "sreh@gmail.com"
    assert engine.case.contact_email != "srehari@gmail.com"
    assert workflow._collected_snapshot()["contact_email"] == "sreh@gmail.com"
    assert resolved.next_question is not None  # the interview continues
    assert workflow._email_pending_completeness is None


def test_full_respelling_after_the_completeness_question_commits_normally():
    # After the completeness question the reporter may spell the complete
    # address instead: the informed re-entry reads back and commits without
    # a dead end (the unchanged-value guard is suspended once the
    # completeness question has been raised).
    engine, workflow = _to_email_offer("em-truncated-then-full")
    first = "S R E H A R I at the rate gmail dot com"
    short = "S R E H at the rate gmail dot com"
    full = "S R E E H A R I at the rate gmail dot com"

    workflow.process_turn(first)
    workflow.process_turn("no")
    workflow.process_turn(short)
    workflow.process_turn("yes")  # raises the completeness question
    assert engine.case.contact_email is None

    respelled = workflow.process_turn(full)
    assert "sreehari@gmail.com" in respelled.next_question[1]

    committed = workflow.process_turn("yes")

    assert engine.case.contact_email == "sreehari@gmail.com"
    assert workflow._collected_snapshot()["contact_email"] == "sreehari@gmail.com"
    assert workflow._email_pending_completeness is None


def test_short_address_without_rejection_commits_normally():
    # The completeness check is rejection-scoped: a genuinely short address
    # with no prior rejection commits on its own explicit confirmation -
    # short local parts are never blanket-rejected.
    engine, workflow = _to_email_offer("em-short-no-rejection")

    offered_yes = workflow.process_turn("yes")
    assert offered_yes.next_question is not None

    readback = workflow.process_turn("S R E H at the rate gmail dot com")
    assert "sreh@gmail.com" in readback.next_question[1]

    committed = workflow.process_turn("yes")

    assert engine.case.contact_email == "sreh@gmail.com"
    assert workflow._collected_snapshot()["contact_email"] == "sreh@gmail.com"
    # Read-back and committed snapshot agree exactly.
    assert "sreh@gmail.com" in readback.next_question[1]


def test_rejected_candidate_is_never_confirmed_unchanged_and_report_stores_new_address():
    # The reported data-integrity sequence, end to end: the (transcription-
    # shortened) first spelling is read back as srehari@gmail.com and
    # rejected; the same spelling again re-reads the same value; a bare
    # "ശരിയാണ്" can NEVER confirm the unchanged rejected candidate - the
    # workflow names the value and asks for a fresh spelling. Only the
    # complete re-spelling produces a new candidate, and only its explicit
    # confirmation stores sreehari@gmail.com in the report.
    engine, workflow = _to_email_offer("em-ml-reject-unchanged")
    missing_e = "എസ് ആർ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"
    full = "എസ് ആർ ഇ ഇ എച്ച് എ ആർ ഐ അറ്റ് ദ റേറ്റ് ജിമെയിൽ ഡോട്ട് കോം"

    first = workflow.process_turn(missing_e)
    assert "srehari@gmail.com" in first.next_question[1]
    assert engine.case.contact_email is None

    workflow.process_turn("അല്ല")  # the reported rejection phrase

    again = workflow.process_turn(missing_e)
    assert "srehari@gmail.com" in again.next_question[1]  # unchanged candidate

    blocked = workflow.process_turn("ശരിയാണ്")
    # The short affirmative cannot confirm the known rejected candidate.
    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    assert blocked.next_question is not None
    assert blocked.next_question[0] == "contact_email"  # fresh spelling asked
    assert "The address still reads as srehari@gmail.com." in blocked.response_text
    assert "spell it out again" in blocked.response_text

    # And it stays blocked while the value stays unchanged.
    again2 = workflow.process_turn(missing_e)
    assert "srehari@gmail.com" in again2.next_question[1]
    blocked2 = workflow.process_turn("ശരിയാണ്")
    assert engine.case.contact_email is None
    assert "The address still reads as srehari@gmail.com." in blocked2.response_text

    # The complete spelling is a NEW candidate: it reads back accurately and
    # only its explicit confirmation commits it.
    fresh = workflow.process_turn(full)
    assert engine.case.contact_email is None
    assert "sreehari@gmail.com" in fresh.next_question[1]

    committed = workflow.process_turn("ശരിയാണ്")

    assert engine.case.contact_email == "sreehari@gmail.com"
    assert engine.case.contact_email != "srehari@gmail.com"
    assert engine.case.consent_to_contact is True
    # The report snapshot stores the newly confirmed candidate, never the
    # rejected or stale value.
    assert workflow._collected_snapshot()["contact_email"] == "sreehari@gmail.com"


# ---- 1-4. confirmation, rejection and correction ----------------------------


def test_consent_yes_valid_email_and_confirmation_keeps_both():
    engine, workflow = _to_email_offer("em-confirm")

    readback = _email_to_readback(workflow)
    assert "I heard your email address as alpha@example.com." in (
        readback.response_text
    )
    # Unconfirmed: the candidate is never reported as the address.
    assert engine.case.contact_email is None

    confirmed = workflow.process_turn("yes")

    assert engine.case.consent_to_contact is True
    assert engine.case.contact_email == "alpha@example.com"
    assert engine.case.contact_email_declined is False
    assert confirmed.next_question is not None
    assert confirmed.next_question[0] == "seriousness"


def test_incorrect_email_phrase_rejects_the_candidate_and_asks_for_a_fix():
    engine, workflow = _to_email_offer("em-incorrect")
    _email_to_readback(workflow)

    rejected = workflow.process_turn("the email address is incorrect")

    # The rejected candidate is discarded, never kept as verified.
    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    # The corrected address is asked for.
    assert rejected.next_question is not None
    assert rejected.next_question[1] == EMAIL_REQUEST_QUESTION
    # Consent is untouched by an email correction.
    assert engine.case.consent_to_contact is True


def test_plain_no_to_the_email_readback_never_changes_consent():
    engine, workflow = _to_email_offer("em-no")
    _email_to_readback(workflow)

    rejected = workflow.process_turn("no")

    assert engine.case.contact_email is None
    assert rejected.next_question is not None
    assert rejected.next_question[1] == EMAIL_REQUEST_QUESTION
    assert engine.case.consent_to_contact is True


def test_corrected_email_updates_only_the_email_field():
    engine, workflow = _to_email_offer("em-corrected")
    _email_to_readback(workflow)
    workflow.process_turn("the email address is incorrect")

    fixed = workflow.process_turn("beta@example.com")
    assert "I heard your email address as beta@example.com." in fixed.response_text

    workflow.process_turn("yes")
    assert engine.case.contact_email == "beta@example.com"
    # Identity and contact details are untouched.
    assert engine.case.confirmation_candidate == {}
    assert engine.case.patient_age == "05-Jan-1990"
    assert engine.case.contact_phone == "9886094321"
    assert engine.case.consent_to_contact is True


# ---- 5. optional email skipped ----------------------------------------------


def test_no_email_address_is_recorded_as_unavailable_and_interview_continues():
    engine, workflow = _to_email_offer("em-unavailable")

    result = workflow.process_turn("I don't have an email address")

    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.consent_to_contact is True
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


def test_declining_at_the_request_turn_also_skips_safely():
    engine, workflow = _to_email_offer("em-decline-request")
    workflow.process_turn("yes")

    result = workflow.process_turn("no, I would rather not share it")

    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.consent_to_contact is True
    assert result.next_question is not None
    assert result.next_question[0] == "seriousness"


# ---- 8. bounded ambiguity ---------------------------------------------------


def test_ambiguous_email_replies_hit_the_retry_limit_and_continue():
    engine, workflow = _to_email_offer("em-bound")
    workflow.process_turn("yes")  # email requested

    workflow.process_turn("hmm")
    workflow.process_turn("not sure")
    final = workflow.process_turn("maybe")

    # The optional email is left unprovided and the interview continues:
    # no loop, no human-review park for an optional field.
    assert engine.case.contact_email is None
    assert engine.case.contact_email_declined is True
    assert engine.case.pending_confirmation is None
    assert engine.current_state.value != "HUMAN_REVIEW"
    assert final.next_question is not None
    assert final.next_question[0] == "seriousness"
    assert engine.case.consent_to_contact is True


# ---- 9-10. consent integrity ------------------------------------------------


def test_explicit_consent_answers_set_the_field_only_at_the_consent_turn():
    engine, workflow = _workflow("em-consent-yes")
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 05-01-1990",
        "yes",
        "9886094321",
        "yes",
    ):
        workflow.process_turn(message)
    assert engine.case.consent_to_contact is None

    workflow.process_turn("Yes, I consent")
    assert engine.case.consent_to_contact is True


def test_consent_no_is_never_flipped_by_a_later_email_answer():
    engine, workflow = _workflow("em-consent-no")
    for message in (
        "I have a headache after taking Lantus",
        "yes",
        "myself",
        "Alpha Beta, 05-01-1990",
        "yes",
        "9886094321",
        "yes",
    ):
        workflow.process_turn(message)
    workflow.process_turn("no thanks")

    assert engine.case.consent_to_contact is False

    # A later email-shaped answer cannot withdraw or flip the consent.
    workflow.process_turn("you can email me at alpha@example.com")
    assert engine.case.consent_to_contact is False


# ---- 12. report integrity ---------------------------------------------------


def test_final_report_distinguishes_consent_from_email_status():
    # Confirmed email: consent and the address are separate report fields.
    engine, workflow = _to_email_offer("em-report-final")
    _email_to_readback(workflow)
    workflow.process_turn("yes")
    contact = engine.to_dict()["contact"]
    assert contact["consent_to_contact"] is True
    assert contact["email"] == "alpha@example.com"
    assert contact["email_declined"] is False

    # Declined/unavailable email: consent stays, the email stays unprovided.
    engine3, workflow3 = _to_email_offer("em-report-declined")
    workflow3.process_turn("I don't have an email address")
    contact3 = engine3.to_dict()["contact"]
    assert contact3["consent_to_contact"] is True
    assert contact3["email"] is None
    assert contact3["email_declined"] is True


# ---- 6-7. endpoint-level: empty transcript and provider failure -------------


class _QueueSTT:
    def __init__(self, transcripts: list[str], fail_status: int | None = None):
        self._transcripts = list(transcripts)
        self._fail_status = fail_status

    def transcribe(self, audio_path, language_code="en-IN"):
        from app.services.prisma import PrismaAPIError

        if self._transcripts:
            return SimpleNamespace(text=self._transcripts.pop(0))
        raise PrismaAPIError(
            "Gnani STT returned an empty transcript.", status_code=self._fail_status or 200
        )


class _WavTTS:
    def synthesize_to_file(self, text, output_path, language="en-IN", voice=None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF" + b"\x00" * 40)
        return path


def _voice_client(stt, store):
    router = create_voice_conversation_router(
        stt=stt, tts=_WavTTS(), session_store=store, llm_provider=AnswerLLMProvider()
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


EMAIL_JOURNEY = [
    "I have a headache after taking Lantus",
    "yes",
    "myself",
    "Alpha Jones, 05-01-1990",
    "yes",
    "9886094321",
    "yes",
    "Yes, I consent",
    "yes",  # email requested; the next turn would carry the address
]


def test_empty_transcript_during_email_capture_retries_without_state_change():
    from app.config import get_gnani_api_key

    store = SessionStore()
    client = _voice_client(_QueueSTT(EMAIL_JOURNEY), store)
    session_id = "em-empty-stt"

    for _ in range(len(EMAIL_JOURNEY)):
        ok = _post(client, session_id)
        assert ok.status_code == 200

    session = store.get_or_create(session_id)
    before_state = session.ae_engine.current_state.value
    before_consent = session.ae_engine.case.consent_to_contact

    empty = _post(client, session_id)  # STT returns an empty transcript
    assert empty.status_code == 200
    body = empty.json()
    assert body["reason"] == "empty_transcript"
    assert "couldn't catch the email address" in body["response_text"]
    # Nothing advanced and nothing was treated as "no" or a rejection.
    case = store.get_or_create(session_id).ae_engine.case
    assert store.get_or_create(session_id).ae_engine.current_state.value == before_state
    assert case.consent_to_contact is before_consent
    assert case.contact_email_declined is False
    assert body["next_question"][0] == "contact_email"
    key = get_gnani_api_key()
    assert not key or key not in body["response_text"]


def test_provider_failure_preserves_the_captured_email_and_consent():
    from app.services.prisma import PrismaAPIError

    store = SessionStore()
    stt = _QueueSTT(EMAIL_JOURNEY + ["alpha@example.com", "yes"], fail_status=403)
    client = _voice_client(stt, store)
    session_id = "em-provider-fail"

    for _ in range(len(EMAIL_JOURNEY) + 2):
        ok = _post(client, session_id)
        assert ok.status_code == 200

    case = store.get_or_create(session_id).ae_engine.case
    assert case.contact_email == "alpha@example.com"
    assert case.consent_to_contact is True

    failed = _post(client, session_id)  # STT now fails with 403
    assert failed.status_code == 500

    after = store.get_or_create(session_id).ae_engine.case
    assert after.contact_email == "alpha@example.com"
    assert after.consent_to_contact is True
    assert after.contact_email_declined is False


# ---- 11. text and voice consistency -----------------------------------------


def test_text_endpoint_drives_the_same_email_flow():
    from app.main import create_app

    client = TestClient(create_app())
    session_id = "em-text"

    def turn(message: str) -> dict:
        resp = client.post(
            "/text/conversation",
            data={"session_id": session_id, "message": message},
        )
        assert resp.status_code == 200
        return resp.json()

    turn("I have a headache after taking Lantus")
    turn("yes")
    turn("myself")
    turn("Alpha Jones, 05-01-1990")
    turn("yes")
    turn("9886094321")
    turn("yes")
    consented = turn("Yes, I consent")
    assert consented["next_question"][0] == "contact_email"

    offered = turn("yes")
    assert offered["next_question"][1] == EMAIL_REQUEST_QUESTION

    readback = turn("alpha@example.com")
    # Text conversations use the text-appropriate confirmation wording.
    assert "I understood your email address as alpha@example.com." in (
        readback["response_text"]
    )
    assert "I heard" not in readback["response_text"]

    confirmed = turn("yes")
    collected = confirmed["ae"]["collected_information"]
    assert collected["consent_to_contact"] is True
    assert collected["contact_email"] == "alpha@example.com"
    assert collected["contact_email_declined"] is False
    assert confirmed["next_question"][0] == "seriousness"


# ---- spelled/dictated email normalization ----------------------------------


def _norm(text: str):
    from app.providers.answer import _normalize_spoken_email

    return _normalize_spoken_email(text)


def test_individually_spelled_letters_normalize():
    assert (
        _norm("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com")
        == "sreehari@gmail.com"
    )


def test_spoken_digits_and_mixed_alphanumeric_normalize():
    assert (
        _norm("A, B, C, one, two, three, at the rate, Gmail, dot, com")
        == "abc123@gmail.com"
    )


def test_repeated_letters_are_preserved():
    assert _norm("A, A, B, C, at the rate, Gmail, dot, com") == "aabc@gmail.com"
    # The repeated "E, E" in "sreehari" is kept, never collapsed.
    assert _norm("S, R, E, E, H, A, R, I, at the rate, Gmail, dot, com") == (
        "sreehari@gmail.com"
    )


def test_spoken_at_dot_underscore_and_hyphen_normalize():
    assert _norm("srihari at gmail dot com") == "srihari@gmail.com"
    assert (
        _norm("john dot doe underscore work at the rate gmail dot com")
        == "john.doe_work@gmail.com"
    )
    assert (
        _norm("my alias hyphen work at the rate gmail dot com")
        == "myalias-work@gmail.com"
    )


def test_ordinary_speech_never_becomes_an_email():
    # Evidence of an address (an @ cue and a dot cue ending in a real domain
    # label) is required; ordinary sentences are rejected.
    assert _norm("look at this dot on my arm") is None
    assert _norm("I am not sure") is None
    assert _norm("the email address is incorrect") is None


# ---- corrections and confirmation state -------------------------------------


def test_correction_phrase_replaces_the_candidate_with_the_new_address():
    engine, workflow = _to_email_offer("em-spell-correct")

    # Initial (misheard) candidate captured from a spoken address.
    first = workflow.process_turn("shrihari at gmail dot com")
    assert "I heard your email address as shrihari@gmail.com." in first.response_text

    # The spelled correction replaces it and is read back in normalized form.
    second = workflow.process_turn("it should be srihari at the rate gmail dot com")
    assert "I heard your email address as srihari@gmail.com." in second.response_text
    # The superseded candidate is never re-read.
    assert "shrihari@gmail.com" not in second.response_text

    workflow.process_turn("yes")
    # Only the latest candidate can be confirmed and included as verified.
    assert engine.case.contact_email == "srihari@gmail.com"


def test_no_the_email_should_be_enters_correction_handling():
    engine, workflow = _to_email_offer("em-should-be")
    _email_to_readback(workflow)

    corrected = workflow.process_turn(
        "No, the email should be alpha dot beta at the rate gmail dot com"
    )

    assert "I heard your email address as alpha.beta@gmail.com." in (
        corrected.response_text
    )
    assert engine.case.contact_email is None  # candidate only, unverified

    workflow.process_turn("yes")
    assert engine.case.contact_email == "alpha.beta@gmail.com"


def test_unclear_confirmation_never_verifies_the_candidate():
    engine, workflow = _to_email_offer("em-unclear")
    _email_to_readback(workflow)

    unclear = workflow.process_turn("hmm")

    assert engine.case.contact_email is None
    assert engine.case.pending_confirmation == "contact_email"
    assert unclear.next_question is not None
    assert unclear.next_question[0] == "confirmation"
    # The interview stays in the email workflow, never the severity question.
    assert "severe or life-threatening" not in unclear.response_text


def test_email_pending_states_never_ask_the_severity_question():
    engine, workflow = _to_email_offer("em-no-severity-leak")

    for reply in ("yes", "alpha beta at gmail dot com", "hmm", "no"):
        result = workflow.process_turn(reply)
        assert "severe or life-threatening" not in result.response_text
        assert result.next_question is not None
        assert result.next_question[0] in ("contact_email", "confirmation")
        # Follow-up consent is unchanged by every email turn.
        assert engine.case.consent_to_contact is True

    # Only after the email is declined/confirmed does severity follow.
    assert engine.case.contact_email_declined or engine.case.pending_confirmation


def test_no_duplicate_email_question_in_one_response():
    engine, workflow = _to_email_offer("em-single-prompt")

    offered = workflow.process_turn("yes")
    assert offered.response_text.count(EMAIL_REQUEST_QUESTION) == 1

    captured = workflow.process_turn("alpha beta at gmail dot com")
    assert captured.response_text.count(EMAIL_REQUEST_QUESTION) <= 1
    assert captured.response_text.count("spell out your email address") <= 1


def test_voice_flow_keeps_the_heard_email_confirmation_wording():
    # The audio/voice path (no text_mode flag) keeps the spoken "heard"
    # wording; only text conversations use "understood".
    engine, workflow = _to_email_offer("em-voice-heard")
    offered = workflow.process_turn("yes")
    assert offered.next_question is not None

    readback = workflow.process_turn("alpha@example.com")
    assert "I heard your email address as alpha@example.com." in (
        readback.response_text
    )
    assert "I understood" not in readback.response_text

    # An explicit text-mode turn uses the text wording instead.
    engine2, workflow2 = _to_email_offer("em-text-understood")
    offered2 = workflow2.process_turn("yes")
    assert offered2.next_question is not None
    readback2 = workflow2.process_turn(
        "alpha@example.com", context={"text_mode": True}
    )
    assert "I understood your email address as alpha@example.com." in (
        readback2.response_text
    )
    assert "I heard" not in readback2.response_text
