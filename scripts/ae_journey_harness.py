"""Offline development harness for the complete multi-turn AE reporting journey.

Feeds a scripted conversation sequentially through the existing
``AEStateEngine`` and ``ConversationalAEWorkflow``. No external service is
contacted: no database, no speech service, no hosted model provider, and no
network call of any kind. The only injected dependency is a deterministic
stand-in for the LLM provider, which is the outside dependency the
conversational workflow already expects via the ``LLMProvider`` protocol.

The state machine itself is reused, never reimplemented: required-field
tracking, state advancement, validation, escalation and completion all stay
inside ``app.workflow.ae_engine``. Answer parsing is shared with the production
``app.providers.answer`` module so the harness and the endpoint never drift.

Run:
    python scripts/ae_journey_harness.py
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.mock import MockLLMProvider  # noqa: E402
from app.workflow.ae_engine import AEStateEngine, WorkflowState  # noqa: E402
from app.workflow.service.conversational import (  # noqa: E402
    ConversationalAEWorkflow,
    WorkflowTurnResult,
)
from app.providers.answer import (  # noqa: E402
    _BASE_INTERPRETED,
    _parse_answer,
    _parse_for_pending_field,
)


INITIAL_REPORT = (
    "I took Lantus yesterday and today I have severe headache and vomiting. "
    "I want to report this adverse event."
)
ANSWER_PATIENT_IDENTIFICATION = "Demo Patient 001, 45 years old"
ANSWER_REPORTER_ROLE = "I am the patient"
ANSWER_CONTACT_METHOD = "You can reach me by phone"
ANSWER_CONTACT_PHONE = "9999990000"
ANSWER_CONTACT_EMAIL = "demo.patient@example.com"
ANSWER_CONSENT = "Yes, I consent"
ANSWER_ONSET = "It started this morning"
ANSWER_SERIOUSNESS = (
    "I did not go to the hospital and I am not experiencing a life-threatening reaction"
)
ANSWER_OUTCOME = "The headache and vomiting are still ongoing"
URGENT_STATEMENT = "I am having severe difficulty breathing"

# Normal journey: every required field supplied, so the production workflow
# reaches CASE_COMPLETED. Consent is captured before any contact detail.
SCRIPTED_JOURNEY = [
    INITIAL_REPORT,
    ANSWER_REPORTER_ROLE,
    ANSWER_PATIENT_IDENTIFICATION,
    ANSWER_CONSENT,
    ANSWER_CONTACT_METHOD,
    ANSWER_CONTACT_PHONE,
    ANSWER_ONSET,
    ANSWER_SERIOUSNESS,
    ANSWER_OUTCOME,
]

# Urgent journey: a genuinely urgent statement arriving mid-interview.
URGENT_JOURNEY = [
    INITIAL_REPORT,
    ANSWER_REPORTER_ROLE,
    ANSWER_PATIENT_IDENTIFICATION,
    ANSWER_CONTACT_PHONE,
    URGENT_STATEMENT,
    ANSWER_OUTCOME,
]


class ScriptedAnswerLLMProvider:
    """Deterministic, fully offline stand-in for the model provider.

    Delegates intent detection and initial report extraction to the existing
    ``MockLLMProvider``, then answers the workflow's pending question for
    plain reply messages using the shared ``app.providers.answer`` extractors.
    Implements the same ``interpret`` surface the conversational workflow
    already consumes.
    """

    def __init__(self, engine: Optional[AEStateEngine] = None) -> None:
        self.engine = engine
        self._base = MockLLMProvider()

    def interpret(self, message: str, context: Optional[Dict[str, Any]] = None) -> Any:
        base = self._base.interpret(message, context)
        if base.intent in _BASE_INTERPRETED:
            return base

        fields = _parse_answer(message)
        if not fields and self.engine is not None:
            pending = self.engine.next_field()
            if pending:
                fields = _parse_for_pending_field(pending[0], message)

        if not fields:
            return base

        merged = dict(base.extracted_fields or {})
        merged.update(fields)
        base.extracted_fields = merged
        if not base.intent:
            base.intent = "ae_answer"
        return base


@dataclass
class TurnRecord:
    """One printed/inspected turn of the scripted journey."""

    index: int
    user_input: str
    intent: str
    current_state: str
    extracted_information: Dict[str, Any]
    next_question: Optional[List[str]]
    priority: str
    human_review_required: bool
    completed: bool
    safety_status: str
    missing_information: List[str] = field(default_factory=list)
    response_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        bar = "-" * 72
        return "\n".join(
            [
                bar,
                f"TURN {self.index}",
                bar,
                f"  user input            : {self.user_input}",
                f"  detected intent       : {self.intent}",
                f"  current state         : {self.current_state}",
                f"  extracted information : {self.extracted_information}",
                f"  next question         : {self.next_question}",
                f"  priority              : {self.priority}",
                f"  human_review_required : {self.human_review_required}",
                f"  completed             : {self.completed}",
                f"  safety status         : {self.safety_status}",
                f"  missing information   : {self.missing_information}",
                f"  assistant response    : {self.response_text}",
            ]
        )


class AEJourneyHarness:
    """Runs a scripted conversation through the production AE workflow."""

    def __init__(self, case_id: str = "AE-harness-demo") -> None:
        self.engine = AEStateEngine.create_case(case_id)
        self.provider = ScriptedAnswerLLMProvider(self.engine)
        self.workflow = ConversationalAEWorkflow(self.engine, self.provider)
        self.turns: List[TurnRecord] = []

    def run_turn(self, user_input: str) -> TurnRecord:
        result: WorkflowTurnResult = self.workflow.process_turn(user_input)
        record = TurnRecord(
            index=len(self.turns) + 1,
            user_input=user_input,
            intent=result.intent,
            current_state=result.current_state,
            extracted_information=dict(result.extracted_information),
            next_question=result.next_question,
            priority=result.priority,
            human_review_required=result.human_review_required,
            completed=result.completed,
            safety_status=result.safety_status,
            missing_information=list(result.missing_information),
            response_text=result.response_text,
        )
        self.turns.append(record)
        return record

    def run_journey(self, messages: List[str]) -> List[TurnRecord]:
        return [self.run_turn(message) for message in messages]


def build_harness(case_id: str = "AE-harness-demo") -> AEJourneyHarness:
    return AEJourneyHarness(case_id)


def render_summary(engine: AEStateEngine) -> str:
    case = engine.case
    return "\n".join(
        [
            "=" * 72,
            "JOURNEY SUMMARY",
            "=" * 72,
            f"  case id              : {case.case_id}",
            f"  final state          : {case.status.value}",
            f"  priority             : {case.priority}",
            f"  completed            : {case.status is WorkflowState.CASE_COMPLETED}",
            f"  human review reason  : {case.human_review_reason}",
            f"  missing information  : {engine.missing_information()}",
            f"  next question        : {engine.next_field()}",
            f"  collected            : {engine.to_dict()['reporter'], engine.to_dict()['contact']}",
            "=" * 72,
        ]
    )


def main() -> int:
    harness = build_harness()
    print("AE MULTI-TURN REPORTING JOURNEY (offline harness, no external services)")
    for message in SCRIPTED_JOURNEY:
        print(harness.run_turn(message).render())
    print(render_summary(harness.engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())