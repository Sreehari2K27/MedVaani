"""Deployment-serving gates: health, SPA static serving and route precedence.

The built React frontend (frontend/dist) is served as static files by the
same FastAPI process that owns every API route and the /audio mount. These
tests lock the ordering contract: the SPA catch-all mount is registered last
and must never shadow /health, any API route, or /audio. Everything here is
offline: no Gnani, Timbre, EVon or LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.prisma import TranscriptionResult

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def make_client() -> TestClient:
    stt = MagicMock()
    stt.transcribe.return_value = TranscriptionResult(text="unused")
    tts = MagicMock()
    tts.synthesize_to_file.return_value = Path("output") / "unused.wav"
    return TestClient(create_app(prisma=stt, timbre=tts))


def test_required_frontend_dist_exists() -> None:
    # The committed dist/ is what production serves; the deployment build
    # depends on it, so its presence and shape are part of the contract.
    assert FRONTEND_DIST.is_dir()
    assert (FRONTEND_DIST / "index.html").is_file()


def test_health_endpoint_reports_ok() -> None:
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_served_root_returns_built_index_html() -> None:
    response = make_client().get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="app">' in response.text
    assert "/assets/index-" in response.text


def test_voice_conversation_route_precedes_spa() -> None:
    # An empty upload fails fast with the API's own 400 before any STT call:
    # the SPA mount never answers for API paths.
    response = make_client().post(
        "/voice/conversation",
        files={"audio_file": ("turn.wav", b"", "audio/wav")},
        data={"session_id": "static-serve-test"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty audio file"


def test_mi_chat_route_precedes_spa() -> None:
    # The API's own validation fires first (no MI service is constructed and
    # no LLM call happens): the SPA mount never answers for /mi/chat.
    response = make_client().post("/mi/chat", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "A question is required."


def test_audio_route_precedes_spa(tmp_path: Path) -> None:
    marker = Path("output") / "static_serving_probe.wav"
    marker.write_bytes(b"RIFF static serving probe")
    try:
        response = make_client().get("/audio/static_serving_probe.wav")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.content == b"RIFF static serving probe"
    finally:
        marker.unlink(missing_ok=True)
