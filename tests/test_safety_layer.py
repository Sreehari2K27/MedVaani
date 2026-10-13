"""Focused unit tests for the deterministic safety layer."""
from __future__ import annotations

from app.safety import SafetyLayer, SafetySignal
from app.workflow.ae_engine import AEStateEngine, WorkflowState


def test_urgent_breathing_difficulty():
    engine = AEStateEngine.create_case("AE-safety-001-demo")
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Mild headache (demo)")

    result = SafetyLayer.classify_input("I have severe difficulty breathing")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True
    assert result.escalate_to_human_review is True
    assert result.mark_high_priority is True
    assert result.preserve_case_info is True
    assert "emergency" in result.message.lower() or "immediate" in result.message.lower()

    SafetyLayer.apply_safety_to_case(result, engine)
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.priority == "high"
    assert engine.case.product == "Lantus (fictional)"  # preserved


def test_urgent_unconsciousness():
    engine = AEStateEngine.create_case("AE-safety-002-demo")
    result = SafetyLayer.classify_input("He is unconscious")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.escalate_to_human_review is True
    assert result.mark_high_priority is True


def test_urgent_severe_chest_pain():
    engine = AEStateEngine.create_case("AE-safety-003-demo")
    result = SafetyLayer.classify_input("I have severe chest pain")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING


def test_urgent_severe_bleeding():
    engine = AEStateEngine.create_case("AE-safety-004-demo")
    result = SafetyLayer.classify_input("I have severe bleeding")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING


def test_urgent_severe_abdominal_stomach_pain():
    for text in (
        "I have severe stomach pain",
        "I have severe abdominal pain",
        "severe belly pain after the injection",
        "the abdominal pain that severe started just now",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.mark_high_priority is True, text


def test_mild_stomach_discomfort_is_not_urgent():
    for text in (
        "I have mild stomach pain",
        "some stomach discomfort after injecting",
        "a little belly pain but it is fine",
        "indigestion and stomach upset",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.mark_high_priority is False, text


def test_urgent_explicit_death_statements():
    for text in (
        "I died",
        "the patient died",
        "she died last night",
        "my father passed away after taking the medicine",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.escalate_to_human_review is True, text
        assert result.mark_high_priority is True, text


def test_negated_death_and_severe_abdominal_statements_do_not_escalate():
    for text in (
        "I did not die",
        "he didn't pass away",
        "no one died",
        "I don't have severe stomach pain",
        "I do not have severe abdominal pain",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.mark_high_priority is False, text


def test_urgent_severe_abdominal_pain_escalates_workflow_and_preserves_event_description():
    from app.providers.mock import MockLLMProvider
    from app.workflow.service.conversational import ConversationalAEWorkflow

    engine = AEStateEngine.create_case("AE-safety-abdominal-demo")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    first = wf.process_turn("I took Lantus and have a headache.")
    assert first.safety_status == "ok"
    original = engine.case.adverse_event
    assert original

    urgent = wf.process_turn("Now I am having severe abdominal pain")
    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    assert original in engine.case.adverse_event


def test_normal_ae_reporting():
    engine = AEStateEngine.create_case("AE-safety-005-demo")
    result = SafetyLayer.classify_input("I developed a headache after taking Lantus.")
    assert result.signal == SafetySignal.NORMAL_AE_REPORTING
    assert result.should_interrupt_ae is False
    assert result.escalate_to_human_review is False
    assert result.mark_high_priority is False


def test_personalized_dose_request():
    engine = AEStateEngine.create_case("AE-safety-006-demo")
    result = SafetyLayer.classify_input("Should I increase my Lantus dose?")
    assert result.signal == SafetySignal.PERSONALIZED_TREATMENT_REQUEST
    assert "healthcare professional" in result.message.lower()
    assert result.should_interrupt_ae is False


def test_medication_stop_request():
    engine = AEStateEngine.create_case("AE-safety-007-demo")
    result = SafetyLayer.classify_input("Should I stop taking Lantus?")
    assert result.signal == SafetySignal.PERSONALIZED_TREATMENT_REQUEST
    assert "healthcare professional" in result.message.lower()


def test_product_information_request():
    engine = AEStateEngine.create_case("AE-safety-008-demo")
    result = SafetyLayer.classify_input("What is Lantus used for?")
    assert result.signal == SafetySignal.PRODUCT_INFORMATION_REQUEST
    assert result.should_interrupt_ae is False
    assert result.escalate_to_human_review is False


def test_unknown_ambiguous_input():
    engine = AEStateEngine.create_case("AE-safety-009-demo")
    result = SafetyLayer.classify_input("blah blah something random")
    assert result.signal == SafetySignal.UNKNOWN_AMBIGUOUS
    assert "clarify" in result.message.lower()


def test_preservation_of_case_information():
    engine = AEStateEngine.create_case("AE-safety-010-demo")
    engine.update("product", "Lantus (fictional)")
    engine.update("adverse_event", "Rash (demo)")
    engine.update("patient_identifier", "PT-001")
    original_product = engine.case.product
    original_ae = engine.case.adverse_event
    original_id = engine.case.patient_identifier

    result = SafetyLayer.classify_input("I have severe chest pain")
    assert result.preserve_case_info is True
    SafetyLayer.apply_safety_to_case(result, engine)

    assert engine.case.product == original_product
    assert engine.case.adverse_event == original_ae
    assert engine.case.patient_identifier == original_id
    assert engine.case.priority == "high"


def test_high_priority_human_review_behavior():
    engine = AEStateEngine.create_case("AE-safety-011-demo")
    result = SafetyLayer.classify_input("severe bleeding")
    SafetyLayer.apply_safety_to_case(result, engine)
    assert engine.current_state is WorkflowState.HUMAN_REVIEW
    assert engine.case.priority == "high"


def test_urgent_life_threatening_statement_still_escalates():
    result = SafetyLayer.classify_input("I am experiencing a life-threatening reaction")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True
    assert result.escalate_to_human_review is True
    assert result.mark_high_priority is True


def test_urgent_breathing_statement_still_escalates():
    result = SafetyLayer.classify_input("I am having severe difficulty breathing")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.mark_high_priority is True


def test_negated_life_threatening_statement_does_not_escalate():
    result = SafetyLayer.classify_input("I am not experiencing a life-threatening reaction")
    assert result.signal != SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is False
    assert result.escalate_to_human_review is False
    assert result.mark_high_priority is False


def test_negated_breathing_statement_does_not_escalate():
    result = SafetyLayer.classify_input("I am not having severe difficulty breathing")
    assert result.signal != SafetySignal.URGENT_LIFE_THREATENING
    assert result.escalate_to_human_review is False


def test_urgent_statement_after_contrast_still_escalates_despite_earlier_negation():
    result = SafetyLayer.classify_input(
        "I am not sure about this, but I am having severe difficulty breathing"
    )
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.mark_high_priority is True


def test_negated_seriousness_answer_is_normal_ae_reporting():
    result = SafetyLayer.classify_input(
        "I did not go to the hospital and I am not experiencing a life-threatening reaction"
    )
    assert result.signal == SafetySignal.NORMAL_AE_REPORTING
    assert result.should_interrupt_ae is False
    assert result.mark_high_priority is False


# ---- urgent vocabulary regression (heart attack / stroke / seizure / ...) ----


def test_urgent_heart_attack():
    result = SafetyLayer.classify_input(
        "I am having a severe heart attack after taking Lantus"
    )
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True
    assert result.escalate_to_human_review is True
    assert result.mark_high_priority is True
    assert "emergency" in result.message.lower() or "immediate" in result.message.lower()


def test_urgent_cardiac_arrest():
    result = SafetyLayer.classify_input("He went into cardiac arrest")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.escalate_to_human_review is True
    assert result.mark_high_priority is True


def test_urgent_stroke():
    result = SafetyLayer.classify_input("She is having a stroke")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.mark_high_priority is True


def test_urgent_seizure():
    result = SafetyLayer.classify_input("He had a seizure after the injection")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.escalate_to_human_review is True


def test_urgent_fainting_with_unresponsiveness():
    result = SafetyLayer.classify_input("She fainted and was unresponsive")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True
    assert result.mark_high_priority is True


def test_urgent_anaphylaxis_and_throat_or_facial_swelling():
    for text in (
        "I think this is anaphylaxis",
        "my throat is swelling",
        "my throat is closing",
        "I have swelling of my lips",
        "lip swelling",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text


def test_urgent_overdose():
    result = SafetyLayer.classify_input("I took an overdose by mistake")
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is True
    assert result.mark_high_priority is True


# ---- negation must suppress the new urgent phrases ----


def test_negated_heart_attack_does_not_escalate():
    result = SafetyLayer.classify_input("I did not have a heart attack")
    assert result.signal != SafetySignal.URGENT_LIFE_THREATENING
    assert result.should_interrupt_ae is False
    assert result.escalate_to_human_review is False
    assert result.mark_high_priority is False


def test_negated_stroke_and_seizure_do_not_escalate():
    stroke = SafetyLayer.classify_input("He did not have a stroke")
    assert stroke.signal != SafetySignal.URGENT_LIFE_THREATENING
    assert stroke.mark_high_priority is False

    seizure = SafetyLayer.classify_input("She has not had a seizure")
    assert seizure.signal != SafetySignal.URGENT_LIFE_THREATENING
    assert seizure.mark_high_priority is False


def test_negated_heart_attack_history_but_current_urgent_symptom_escalates():
    result = SafetyLayer.classify_input(
        "I did not have a heart attack, but now I have severe chest pain"
    )
    assert result.signal == SafetySignal.URGENT_LIFE_THREATENING
    assert result.mark_high_priority is True


def test_urgent_heart_attack_escalates_workflow_and_preserves_event_description():
    from app.providers.mock import MockLLMProvider
    from app.workflow.service.conversational import ConversationalAEWorkflow

    engine = AEStateEngine.create_case("AE-safety-heart-attack-demo")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    first = wf.process_turn("I took Lantus and have a headache.")
    assert first.safety_status == "ok"
    original = engine.case.adverse_event
    assert original  # original event description captured

    urgent = wf.process_turn("Now I am having a severe heart attack")
    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    assert urgent.current_state == "HUMAN_REVIEW"
    assert (
        "emergency" in urgent.response_text.lower()
        or "immediate" in urgent.response_text.lower()
    )
    # The original description is preserved; the urgent statement is appended.
    assert original in engine.case.adverse_event
    assert "severe heart attack" in engine.case.adverse_event


# ---- Hindi urgent-symptom vocabulary -----------------------------------------


def test_urgent_hindi_heart_attack_phrases():
    for text in (
        "मुझे दिल का दौरा हुआ",
        "हृदय आघात के लक्षण हैं",
        "हार्ट अटैक आया",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.escalate_to_human_review is True, text
        assert result.mark_high_priority is True, text
        assert result.preserve_case_info is True, text


def test_urgent_hindi_stroke_seizure_phrases():
    for text in ("उसे स्ट्रोक हुआ", "पक्षाघात हुआ", "मुझे मिर्गी का दौरा पड़ा"):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.mark_high_priority is True, text


def test_urgent_hindi_unconscious_allergic_overdose_phrases():
    for text in (
        "वह बेहोश है",
        "मरीज अचेत है",
        "एनाफिलैक्सिस हो रहा है",
        "गंभीर एलर्जी हो गई",
        "ओवरडोज हो गया",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.escalate_to_human_review is True, text
        assert result.mark_high_priority is True, text


def test_urgent_romanized_hindi_phrases():
    for text in (
        "mujhe dil ka daura hua",
        "use mirgi pada",
        "woh behosh ho gaya",
        "use gambhir allergy hua",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.mark_high_priority is True, text


def test_negated_hindi_urgent_phrases_do_not_escalate():
    # Hindi negation follows the phrase ("...नहीं हुआ", "... nahi hua"), which
    # the clause-prefix window cannot see; the after-match check handles it.
    for text in (
        "उसे दिल का दौरा नहीं हुआ",
        "मुझे मिर्गी का दौरा नहीं पड़ा",
        "वह बेहोश नहीं है",
        "उसे स्ट्रोक नहीं हुआ",
        "ओवरडोज नहीं हुआ",
        "mujhe dil ka daura nahi hua",
        "use mirgi nahi hua",
        "woh behosh nahi hai",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.escalate_to_human_review is False, text
        assert result.mark_high_priority is False, text


def test_preceding_hindi_negation_does_not_escalate():
    # Pre-posed negation reuses the existing clause-prefix window.
    for text in (
        "मुझे नहीं लगता कि दिल का दौरा हुआ",
        "mujhe nahi lagta ki dil ka daura hua",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.mark_high_priority is False, text


def test_urgent_hindi_severe_stomach_pain_phrases():
    for text in (
        "मुझे बहुत पेट दर्द हो रहा है",
        "गंभीर पेट दर्द",
        "तेज पेट दर्द",
        "पेट में बहुत दर्द",
        "bahut pet dard kar raha hai",
        "bohot pet dard",
        "tez pet dard",
        "pet mein bahut dard",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.escalate_to_human_review is True, text
        assert result.mark_high_priority is True, text


def test_mild_hindi_stomach_pain_is_not_urgent():
    # Severity is mandatory: bare or mild stomach-pain mentions never escalate.
    for text in (
        "मुझे पेट दर्द है",
        "पेट में हल्का दर्द",
        "pet dard",
        "halka pet dard",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.mark_high_priority is False, text


def test_negated_hindi_stomach_pain_does_not_escalate():
    # Negation reuses the existing Hindi after-match window ("... नहीं है",
    # "... nahi hai") plus the clause-prefix window for pre-posed denial.
    for text in (
        "पेट दर्द नहीं है",
        "pet dard nahi hai",
        "mere ko bahut pet dard nahi hai",
        "मुझे बहुत पेट दर्द नहीं हो रहा",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.mark_high_priority is False, text


def test_urgent_hindi_stomach_pain_escalates_workflow_and_preserves_event_description():
    from app.providers.mock import MockLLMProvider
    from app.workflow.service.conversational import ConversationalAEWorkflow

    engine = AEStateEngine.create_case("AE-safety-hindi-abdominal-demo")
    wf = ConversationalAEWorkflow(engine, MockLLMProvider())
    first = wf.process_turn("I took Lantus and have a headache.")
    assert first.safety_status == "ok"
    original = engine.case.adverse_event
    assert original

    urgent = wf.process_turn(
        "kal davai lene ke baad mujhe aaj bahut pet dard kar raha hai"
    )
    assert urgent.safety_status == "urgent_life_threatening"
    assert urgent.human_review_required is True
    assert urgent.priority == "high"
    assert urgent.current_state == "HUMAN_REVIEW"
    # The original description is preserved; the urgent statement is appended.
    assert original in engine.case.adverse_event
    assert "bahut pet dard" in engine.case.adverse_event


def test_urgent_hindi_death_phrases():
    for text in (
        "वह मर गया",
        "रोगी की मृत्यु हो गई",
        "mar gaya",
        "woh mar gaya",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal == SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is True, text
        assert result.escalate_to_human_review is True, text
        assert result.mark_high_priority is True, text


def test_negated_hindi_death_phrases_do_not_escalate():
    for text in (
        "मर गया नहीं",
        "mar gaya nahi hai",
        "nahi mar gaya",
    ):
        result = SafetyLayer.classify_input(text)
        assert result.signal != SafetySignal.URGENT_LIFE_THREATENING, text
        assert result.should_interrupt_ae is False, text
        assert result.mark_high_priority is False, text
