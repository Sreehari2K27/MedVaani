"""In-memory session store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from app.workflow.ae_engine import AEStateEngine


@dataclass
class Session:
    session_id: str
    ae_engine: AEStateEngine = field(default_factory=lambda: AEStateEngine.create_case())
    context: Dict[str, Any] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                ae_engine=AEStateEngine.create_case(),
            )
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def clear(self) -> None:
        self._sessions.clear()
