"""Grounded MI answer generation.

Answering is extractive by default: when the LLM provider does not supply a
grounded answer, MIAnswerService quotes the most relevant passage from the top
retrieved chunk and cites the package insert page. This keeps answers
deterministic and source-grounded without adding an LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

from app.mi.retrieval import LantusRetriever, RetrievedChunk
from app.mi.retrieval.retriever import normalize_token, tokenize_text
from app.providers.base import LLMProvider


@dataclass
class MIAnswerResult:
    answer: str
    intent: str = "product_info"
    source_chunks: List[Dict[str, Any]] = None
    source_page_numbers: List[int] = None
    grounded: bool = True
    unsupported: bool = False
    retrieved_chunk_ids: List[str] = None

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []
        if self.source_page_numbers is None:
            self.source_page_numbers = []
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MAX_PASSAGE_CHARS = 400
_GROUNDED_THRESHOLD = 0.9

_PROVIDER_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "has", "have", "had", "who", "whom", "which", "their", "them",
        "they", "his", "her", "its", "not", "may", "can", "any", "all", "also",
        "into", "than", "then", "when", "what", "your", "you", "will", "been",
        "being", "there", "these", "those", "does", "would", "should", "such",
        "other", "some", "each", "both", "many", "much", "more", "only", "one",
        "two", "how", "about",
    }
)


def _provider_words(answer: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{4,}", answer.lower()) if w not in _PROVIDER_STOPWORDS
    }


def _is_grounded_in(answer: str, chunks: List[RetrievedChunk]) -> bool:
    """Provider answers are only trusted when their content words appear in
    the retrieved context (i.e. the answer could have come from the citation)."""
    words = _provider_words(answer)
    if not words:
        return False
    context = " ".join(r.text.lower() for r in chunks)
    present = sum(1 for w in words if w in context)
    return present / len(words) >= _GROUNDED_THRESHOLD


def _best_passage(chunk: RetrievedChunk, query_terms: list[str]) -> str:
    """Pick the sentence in a chunk that best matches the query terms."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(chunk.text) if s.strip()]
    if not sentences:
        return " ".join(chunk.text.split())[:_MAX_PASSAGE_CHARS]

    def score(sentence: str) -> int:
        lowered = sentence.lower()
        return sum(1 for term in query_terms if term in lowered)

    best = max(sentences, key=score)
    if score(best) == 0 and len(sentences) > 1:
        best = sentences[0]
    # Don't start a quote with the section heading itself.
    if chunk.section_heading and chunk.section_heading.strip():
        heading = chunk.section_heading.strip()
        if best.lstrip().lower().startswith(heading.lower()):
            best = best.lstrip()[len(heading):].lstrip()
    if not best:
        best = sentences[0]
    if len(best) > _MAX_PASSAGE_CHARS:
        best = " ".join(best.split())[:_MAX_PASSAGE_CHARS]
    if not best.endswith((".", "!", "?")):
        best = f"{best}."
    return best


def _query_terms(question: str) -> list[str]:
    words = tokenize_text(question)
    # Keep full-length terms only; prefer matched suffix-sim tokens.
    return words


def _build_grounded_answer(chunks: List[RetrievedChunk], question: str) -> str:
    top = chunks[0]
    terms = _query_terms(question)
    passages = []
    for chunk in chunks[:2]:
        passage = _best_passage(chunk, terms)
        location = f"page {chunk.page_number}"
        if chunk.section_heading and chunk.section_heading.strip():
            location += f", section \"{chunk.section_heading.strip()}\""
        passages.append(f"{passage} (Source: Lantus package insert, {location}).")
        if len(" ".join(passages)) >= 240:
            break
    return " ".join(passages)


class MIAnswerService:
    def __init__(self, retriever: LantusRetriever = None, llm_provider: LLMProvider = None) -> None:
        from app.providers import MockLLMProvider

        self.retriever = retriever or LantusRetriever()
        self.llm_provider = llm_provider or MockLLMProvider()

    def answer(self, question: str, top_k: int = 3) -> MIAnswerResult:
        if not question or not question.strip():
            return MIAnswerResult(
                answer="No question provided.",
                intent="product_info",
                grounded=False,
                unsupported=True,
            )

        question = question.strip()
        retrieved = self.retriever.retrieve(question, k=top_k)

        # Build context for LLM
        context_parts = []
        for r in retrieved:
            context_parts.append(f"=== Page {r.page_number} ===\n{r.text}")

        context_text = "\n\n".join(context_parts) if context_parts else ""

        grounding_instruction = (
            "Answer ONLY using the supplied reference content. "
            "Do NOT invent facts. Do NOT use outside medical knowledge. "
            "If the supplied reference does not contain enough information, "
            "explicitly say that sufficient information is not available in the provided product reference."
        )

        llm_context = {
            "question": question,
            "context": context_text,
            "grounding_instruction": grounding_instruction,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "text": r.text,
                }
                for r in retrieved
            ],
        }

        llm_result = self.llm_provider.interpret(
            f"grounding_instruction: {grounding_instruction}\n\nQuestion: {question}\n\nContext: {context_text}",
            llm_context,
        )

        provider_answer = (llm_result.extracted_fields or {}).get("answer")

        answer = None
        if provider_answer and retrieved and llm_result.intent == "product_info":
            if _is_grounded_in(provider_answer, retrieved):
                answer = provider_answer
        if not answer and retrieved:
            # Deterministic extractive answer, cited to the package insert.
            answer = _build_grounded_answer(retrieved, question)
        if not answer:
            answer = "Sufficient information is not available in the provided product reference."

        unsupported = not retrieved or "sufficient information is not available" in answer.lower()

        return MIAnswerResult(
            answer=answer,
            intent="product_info",
            source_chunks=[
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "section_heading": r.section_heading,
                    "text": r.text,
                }
                for r in retrieved
            ],
            source_page_numbers=[r.page_number for r in retrieved],
            grounded=bool(retrieved),
            unsupported=unsupported,
            retrieved_chunk_ids=[r.chunk_id for r in retrieved],
        )
