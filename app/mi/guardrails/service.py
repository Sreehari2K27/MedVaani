"""MI service with guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.mi.guardrails.guardrail import MIGuardrail, MIGuardrailResult
from app.mi.answer.service import MIAnswerService, MIAnswerResult


@dataclass
class MIGroundedResult:
    status: str
    allowed: bool
    category: str
    answer: str
    source_chunks: List[Dict[str, Any]] = None
    source_page_numbers: List[int] = None
    retrieved_chunk_ids: List[str] = None
    reason: Optional[str] = None
    grounded: bool = False
    supported: bool = False

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []
        if self.source_page_numbers is None:
            self.source_page_numbers = []
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []


MIGHybridResult = MIGroundedResult


class MIGroundedService:
    def __init__(self, answer_service: MIAnswerService = None) -> None:
        self.answer_service = answer_service or MIAnswerService()

    def handle(self, question: str, top_k: int = 3) -> MIGHybridResult:
        guardrail = MIGuardrail.check(question)
        if not guardrail.allowed:
            return MIGHybridResult(
                status="blocked",
                allowed=False,
                category=guardrail.category,
                answer=guardrail.response_text,
                reason=guardrail.reason,
            )

        ans = self.answer_service.answer(question, top_k=top_k)
        return MIGHybridResult(
            status="allowed",
            allowed=True,
            category=guardrail.category,
            answer=ans.answer,
            source_chunks=ans.source_chunks,
            source_page_numbers=ans.source_page_numbers,
            retrieved_chunk_ids=ans.retrieved_chunk_ids,
            grounded=ans.grounded,
            supported=not ans.unsupported,
        )


class MIGHybridService(MIGroundedService):
    pass
