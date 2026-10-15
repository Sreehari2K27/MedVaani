"""Focused tests for MI answer generation."""
from __future__ import annotations

import re

from app.mi.answer import MIAnswerService, MIAnswerResult
from app.mi.retrieval import LantusRetriever
from app.providers import MockLLMProvider


def test_supported_product_information_question():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=2)
    assert isinstance(result, MIAnswerResult)
    assert result.intent == "product_info"


def test_answer_generation_from_retrieved_context():
    service = MIAnswerService(retriever=LantusRetriever(), llm_provider=MockLLMProvider())
    result = service.answer("What is Lantus used for?", top_k=2)
    assert result.answer
    assert len(result.answer) > 0


def test_source_page_metadata_preservation():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=3)
    assert result.source_page_numbers is not None
    assert len(result.source_page_numbers) > 0
    assert all(p >= 1 for p in result.source_page_numbers)
    assert result.retrieved_chunk_ids is not None


def test_unsupported_question_when_retrieval_insufficient():
    service = MIAnswerService()
    # Use a very specific query unlikely to match
    result = service.answer("nonexistentmedicationconceptxyz123", top_k=2)
    assert result.unsupported is True
    assert "sufficient information is not available" in result.answer.lower()


def test_grounding_instruction_supplied():
    service = MIAnswerService()
    result = service.answer("storage conditions", top_k=2)
    assert result.answer
    assert result.intent == "product_info"


def test_deterministic_mock_behavior():
    service1 = MIAnswerService(llm_provider=MockLLMProvider())
    service2 = MIAnswerService(llm_provider=MockLLMProvider())
    res1 = service1.answer("What is Lantus used for?", top_k=1)
    res2 = service2.answer("What is Lantus used for?", top_k=1)
    assert res1.answer == res2.answer


def test_empty_question_handling():
    service = MIAnswerService()
    result = service.answer("", top_k=2)
    assert result.answer
    assert result.unsupported or result.grounded is False


def test_top_k_behavior():
    service = MIAnswerService()
    result = service.answer("indication", top_k=1)
    assert len(result.source_chunks) <= 1
    result2 = service.answer("indication", top_k=5)
    assert len(result2.source_chunks) <= 5


def test_storage_answer_is_extractive_and_cited():
    service = MIAnswerService()
    result = service.answer("How should Lantus be stored?", top_k=3)
    assert result.answer
    assert "package insert" in result.answer
    assert "page 11" in result.answer
    assert "Information relevant to your question" not in result.answer
    assert result.grounded is True
    assert result.unsupported is False


def test_refrigeration_answer_uses_storage_section():
    service = MIAnswerService()
    result = service.answer("Does Lantus need refrigeration?", top_k=3)
    assert "page 11" in result.answer
    assert "refriger" in result.answer.lower()


def test_side_effects_answer_uses_adverse_reactions_section():
    service = MIAnswerService()
    result = service.answer("What are the side effects of Lantus?", top_k=3)
    assert "package insert" in result.answer
    assert any(p == 6 for p in result.source_page_numbers)
    assert result.unsupported is False


def test_indication_answer_is_deterministic_and_grounded():
    service = MIAnswerService()
    result = service.answer("What is Lantus used for?", top_k=3)
    assert result.answer
    assert result.grounded is True
    assert result.unsupported is False
    assert result.source_page_numbers


def test_used_for_question_returns_indication_passage_not_packaging():
    """The 'used for' question must be answered from the indication section."""
    service = MIAnswerService()
    result = service.answer(
        "What is Lantus SoloStar used for?", top_k=3
    )
    lowered = result.answer.lower()
    # The indication statement itself, as a clean sentence (no label glue).
    assert "for the treatment of diabetes mellitus" in lowered
    assert "atc code" not in lowered
    # Not packaging, pen-appearance, or shelf-life content.
    assert "appearance" not in lowered
    assert "shelf life" not in lowered
    assert "instruction leaflet" not in lowered
    # The citation matches the supporting source page.
    assert "package insert" in result.answer
    assert "page 1" in result.answer
    assert 1 in result.source_page_numbers
    assert result.unsupported is False


def test_adverse_reactions_question_returns_relevant_passage():
    """The adverse-reactions question must quote the adverse-reactions
    content (not the CIOMS frequency legend that shares the heading)."""
    service = MIAnswerService()
    result = service.answer(
        "What are the common adverse reactions listed for Lantus SoloStar "
        "in its product information?",
        top_k=3,
    )
    assert result.unsupported is False
    lowered = result.answer.lower()
    assert "hypoglycemia" in lowered
    assert "most frequent adverse reaction" in lowered
    assert "cioms frequency rating" not in lowered
    assert "package insert" in result.answer
    assert 6 in result.source_page_numbers
    assert "page 6" in result.answer
    assert result.grounded is True


def test_side_effects_question_returns_adverse_section_not_unrelated_text():
    """The side-effects question answers from the Adverse Reactions section
    only: full labeled groups with source frequencies, one accurate citation,
    and no unrelated administration passage."""
    service = MIAnswerService()
    result = service.answer("What are side effects of Lantus?", top_k=3)
    assert result.unsupported is False
    lowered = result.answer.lower()

    # Source facts, verbatim numbers included.
    assert "hypoglycemia" in lowered
    assert "lipohypertrophy was observed in 1 to 2 %" in lowered
    assert "injection site reactions were observed in 3 to 4 %" in lowered

    # Each source group stays a distinct bullet, in source order (the key
    # hypoglycemia risk first) - never merged, never reordered.
    for group in (
        "• hypoglycemia:",
        "• eyes:",
        "• skin and subcutaneous tissue disorders:",
        "• injection site and allergic reactions:",
        "• other reactions:",
    ):
        assert group in lowered
    assert lowered.index("• hypoglycemia:") < lowered.index("• eyes:")
    assert lowered.index("• eyes:") < lowered.index("• skin and subcutaneous")

    # No frequency legend (it defines the scale, it is not a reaction) and
    # no passage from the unrelated Administration section.
    assert "cioms" not in lowered
    assert "blood glucose monitoring is recommended" not in lowered

    # Exactly one citation, on its own line, matching the source section.
    assert lowered.count("(source:") == 1
    assert (
        '\n(source: lantus package insert, page 6, section "adverse reactions").'
        in lowered
    )
    assert 6 in result.source_page_numbers

    # Fact preservation: every number in the answer exists in the source.
    source = next(
        c["text"]
        for c in service.retriever.chunks
        if c["chunk_id"].startswith("chunk_6")
    )
    answer_body = result.answer.split("\n(Source:")[0]
    for number in re.findall(r"\d+", answer_body):
        assert number in source, f"number {number} not present in the source"


# ---- SoloStar Instruction Leaflet pen-use answers ---------------------------


def _pen_answer(question: str) -> str:
    service = MIAnswerService(
        llm_provider=_ProviderStub(exc=RuntimeError("offline"))
    )
    result = service.answer(question, top_k=3)
    assert result.unsupported is False
    return result.answer


def test_pen_step_check_insulin():
    answer = _pen_answer(
        "How do I check the insulin before using my Lantus SoloStar pen?"
    )
    assert "Step 1. Check the insulin" in answer
    assert "A. Check the label on your SoloStar" in answer
    assert "B. Take off the pen cap." in answer
    assert "C. Check the appearance of your insulin." in answer
    assert "cloudy, colored or has particles" in answer
    assert answer.strip().endswith(
        "(Source: Lantus package insert, page 12)."
    )
    # Device instructions only: no clinical/adverse content mixed in.
    assert "hypoglycemia" not in answer.lower()


def test_pen_step_attach_needle_across_page_boundary():
    answer = _pen_answer("How do I attach a new needle to the SoloStar pen?")
    assert "Step 2. Attach the needle" in answer
    assert "Always use a new sterile needle for each injection." in answer
    # p13 continuation is included, in source order, after the p12 lines.
    assert "Wipe the Rubber Seal with alcohol." in answer
    assert "A. Remove the protective seal from a new needle." in answer
    assert "B. Line up the needle with the pen" in answer
    assert "screw or push on" in answer
    # The material warning is never omitted.
    assert "damage the rubber seal and cause leakage or break the needle" in answer
    assert answer.index("Always use a new sterile needle") < answer.index(
        "Wipe the Rubber Seal"
    )
    assert answer.strip().endswith(
        "(Source: Lantus package insert, pages 12, 13)."
    )


def test_pen_step_safety_test_complete_and_scoped():
    answer = _pen_answer("How do I perform the safety test before injecting?")
    assert "Step 3. Perform a safety test" in answer
    assert "A. Select a dose of 2 units by turning the dosage selector." in answer
    assert "C. Hold the pen with the needle pointing upwards." in answer
    assert "D. Tap the insulin reservoir" in answer
    assert "E. Press the injection button all the way in." in answer
    # Failure-path warnings retained.
    assert "repeat the safety test two more times" in answer
    assert "the needle may be blocked. Change the needle" in answer
    # Pages 13-14 only; Step 5 injection steps are a different question.
    assert answer.strip().endswith(
        "(Source: Lantus package insert, pages 13, 14)."
    )
    assert "count to 10" not in answer


def test_pen_step_select_dose_and_inject_in_sequence():
    answer = _pen_answer(
        "How do I select the dose and inject with the SoloStar pen?"
    )
    assert "Step 4. Select the dose" in answer
    assert "Step 5. Inject the dose" in answer
    assert answer.index("Step 4.") < answer.index("Step 5.")
    assert "from a minimum of 1 unit to a maximum of 80 units" in answer
    assert "Check that the dose window shows" in answer
    # The source's clinician qualification is retained verbatim.
    assert "Use the injection method as instructed by your healthcare professional." in answer
    assert "Slowly count to 10" in answer
    assert answer.strip().endswith(
        "(Source: Lantus package insert, pages 14, 15)."
    )


def test_pen_step_remove_and_discard_needle():
    answer = _pen_answer(
        "How do I remove and dispose of the needle after injection?"
    )
    assert "Step 6. Remove and discard the needle" in answer
    assert "store Solostar\u00ae without a needle attached" in answer
    assert "A. Put the outer needle cap back on the needle" in answer
    assert "never replace the inner needle cap" in answer
    assert "Dispose of the needle safely" in answer
    assert answer.strip().endswith(
        "(Source: Lantus package insert, page 15)."
    )
    # "after injection" is context: the dose/injection steps are not pulled in.
    assert "count to 10" not in answer


def test_pen_general_how_to_use_overview_is_sequential():
    answer = _pen_answer("How do I use the Lantus SoloStar pen?")
    headings = [
        line for line in answer.split("\n") if line.startswith("Step ")
    ]
    assert len(headings) == 6
    assert [h.split(".")[0] for h in headings] == [
        "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6",
    ]
    # Each overview line is source-supported (heading + first instruction).
    assert "instructed by your healthcare professional" in answer
    assert "Always remove the needle after each injection" in answer
    assert answer.strip().endswith(
        "(Source: Lantus package insert, pages 12, 13, 14, 15)."
    )


def test_lettered_substeps_survive_sentence_splitting():
    from app.mi.answer.service import _SENTENCE_SPLIT

    parts = _SENTENCE_SPLIT.split(
        "A. Take off the pen cap. B. Check the appearance of your insulin."
    )
    assert parts == [
        "A. Take off the pen cap.",
        "B. Check the appearance of your insulin.",
    ]
    # Normal sentence boundaries still split (word endings are not markers).
    prose = _SENTENCE_SPLIT.split(
        "Do not allow the insulin to freeze, discard if frozen. Opened 10 mL vials."
    )
    assert len(prose) == 2
    # Abbreviations remain protected.
    kept = _SENTENCE_SPLIT.split(
        "store it (e.g. in a refrigerator) and protected from light."
    )
    assert len(kept) == 1


def test_mechanism_question_answered_from_mode_of_action():
    """'How does it work?' must get the pharmacodynamics content, not a
    refusal, not the product cover, and not a generic fragment."""
    service = MIAnswerService()
    result = service.answer("How does Lantus SoloStar work?", top_k=3)
    lowered = result.answer.lower()
    assert result.unsupported is False
    assert "sufficient information is not available" not in lowered
    assert "micro-precipitates" in lowered
    assert "prolonged duration of action" in lowered
    # Not the product cover / pen packaging.
    assert "prefilled pen" not in lowered
    assert "package insert" in result.answer
    assert 7 in result.source_page_numbers
    assert "page 7" in result.answer


def test_mechanism_answer_still_safe_when_provider_refuses():
    service = MIAnswerService(llm_provider=_ProviderStub(
        answer="Sufficient information is not available in the provided product reference."
    ))
    result = service.answer("How does Lantus SoloStar work?", top_k=3)
    assert result.unsupported is False
    assert "micro-precipitates" in result.answer.lower()
    assert "package insert" in result.answer


def test_storage_table_rows_rendered_as_faithful_sentences():
    """The flattened table becomes grammatical bullets that preserve column
    relationships (unopened vs in-use) and every source number."""
    service = MIAnswerService()
    result = service.answer(
        "What are the storage durations for unopened and in-use Lantus products?",
        top_k=3,
    )
    assert result.unsupported is False
    answer = result.answer
    lowered = answer.lower()
    # Structure: intro + one bullet per product row, citation on its own line.
    assert "summarized below." in answer
    assert answer.count("•") == 3
    assert "\n(Source: Lantus package insert, page 11" in answer
    # Column relationships preserved per row.
    assert "10mL vial: not in use (unopened), until the expiration date" in answer
    assert "3ml cartridge inserted into pen (solostar®)" in lowered
    assert "in use (opened), 28 days, refrigerated or room temperature." in lowered
    assert "in use (opened), 28 days, room temperature only (do not refrigerate)." in lowered
    # Raw flattened cells are never dumped.
    assert "not in use (unopened) not in use in use (opened)" not in lowered
    # Fact preservation: every number in the answer exists in the source chunk.
    source = next(
        c["text"]
        for c in service.retriever.chunks
        if c["chunk_id"].startswith("chunk_11")
    )

    answer_without_citation = answer.split("\n(Source:")[0]
    for number in re.findall(r"\d+", answer_without_citation):
        assert number in source, f"number {number} not present in the source"


def test_refusal_when_evidence_is_missing():
    """With no retrieved evidence the fixed insufficient message is used."""
    service = MIAnswerService()
    result = service.answer("nonexistentmedicationconceptxyz123", top_k=2)
    assert result.unsupported is True
    assert "sufficient information is not available" in result.answer.lower()


# ---- provider (e.g. Mistral) acceptance gates -------------------------------


class _ProviderStub:
    """Stub LLM provider for MI: returns one answer or raises."""

    def __init__(self, answer=None, exc=None, intent="product_info") -> None:
        self.answer = answer
        self.exc = exc
        self.intent = intent
        self.calls = 0

    def interpret(self, message, context=None):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        from app.providers.base import LLMResult

        return LLMResult(intent=self.intent, extracted_fields={"answer": self.answer})


def _storage_sentence():
    retriever = LantusRetriever()
    chunks = retriever.retrieve("storage conditions", k=1)
    assert chunks, "storage chunks must exist for this fixture"
    text = chunks[0].text.strip()
    sentence = text.split(". ")[0]
    if not sentence.endswith("."):
        sentence = f"{sentence}."
    return retriever, chunks, sentence


def test_fully_grounded_provider_answer_accepted_with_citation():
    retriever, chunks, sentence = _storage_sentence()
    stub = _ProviderStub(answer=sentence)  # grounded but no citation yet
    service = MIAnswerService(retriever=retriever, llm_provider=stub)
    result = service.answer("How should Lantus be stored?", top_k=1)
    assert stub.calls == 1
    page = chunks[0].page_number
    assert result.answer == f"{sentence}\n(Source: Lantus package insert, page {page})."
    assert result.unsupported is False


def test_provider_answer_with_citation_kept_verbatim():
    retriever, _chunks, sentence = _storage_sentence()
    cited = f"{sentence} (Source: Lantus package insert, page 11)."
    stub = _ProviderStub(answer=cited)
    service = MIAnswerService(retriever=retriever, llm_provider=stub)
    result = service.answer("How should Lantus be stored?", top_k=1)
    assert stub.calls == 1
    assert result.answer == cited
    # No doubled citation appended.
    assert result.answer.count("(Source:") == 1


def test_extra_unsupported_claim_rejects_whole_provider_answer():
    retriever, _chunks, sentence = _storage_sentence()
    tainted = f"{sentence} It is also a proven cure for influenza."
    stub = _ProviderStub(answer=tainted)
    service = MIAnswerService(retriever=retriever, llm_provider=stub)
    result = service.answer("How should Lantus be stored?", top_k=1)
    assert stub.calls == 1
    # The unsupported sentence poisons the answer: extractive fallback used.
    assert result.answer != tainted
    assert "cure for influenza" not in result.answer
    assert "package insert" in result.answer


def test_provider_refusal_with_evidence_falls_back_to_extractive_quote():
    retriever = LantusRetriever()
    stub = _ProviderStub(
        answer="Sufficient information is not available in the provided product reference."
    )
    service = MIAnswerService(retriever=retriever, llm_provider=stub)
    result = service.answer("What is Lantus used for?", top_k=2)
    assert stub.calls == 1
    # Retrieval DID return evidence: the grounded extractive quote answers
    # from the actual source instead of surfacing a generic refusal.
    assert "sufficient information is not available" not in result.answer.lower()
    assert "for the treatment of diabetes mellitus" in result.answer.lower()
    assert "package insert" in result.answer
    assert result.unsupported is False


def test_provider_failure_falls_back_to_extractive_answer():
    stub = _ProviderStub(exc=RuntimeError("mistral down"))
    service = MIAnswerService(llm_provider=stub)
    result = service.answer("How should Lantus be stored?", top_k=3)
    assert stub.calls == 1
    assert result.answer
    assert "package insert" in result.answer
    assert result.unsupported is False


def test_non_product_info_provider_intent_not_used():
    retriever, _chunks, sentence = _storage_sentence()
    stub = _ProviderStub(answer=sentence, intent="unknown")
    service = MIAnswerService(retriever=retriever, llm_provider=stub)
    result = service.answer("How should Lantus be stored?", top_k=1)
    assert result.answer != sentence
    assert "package insert" in result.answer


def test_guardrail_blocks_before_any_provider_call():
    from app.mi.guardrails import MIGroundedService

    stub = _ProviderStub(answer="should not be used")
    guarded = MIGroundedService(answer_service=MIAnswerService(llm_provider=stub))
    result = guarded.handle("How many units should I take tonight?")
    assert result.allowed is False
    assert result.status == "blocked"
    assert stub.calls == 0
