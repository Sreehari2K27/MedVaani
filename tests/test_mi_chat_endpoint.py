"""Focused tests for the dedicated MI chat endpoints (mocked, offline).

No external API calls: the STT client and the MI answer provider are always
injected stubs, and the grounding path uses the real offline retriever over
the local chunk index only.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.mi_chat import create_mi_chat_router
from app.mi.answer import MIAnswerService
from app.mi.guardrails import MIGroundedService
from app.mi.retrieval import LantusRetriever
from app.services.prisma import PrismaAPIError, PrismaValidationError

CHAT_KEYS = {
    "answer",
    "status",
    "category",
    "grounded",
    "supported",
    "source_page_numbers",
    "retrieved_chunk_ids",
    "urgent",
    "suggest_ae_report",
}


class _FailingProvider:
    """Provider whose failure forces the deterministic extractive answer."""

    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, message, context=None):
        self.calls += 1
        raise RuntimeError("mistral down")


class _CountingProvider:
    """Provider that records calls and returns an unusable canned answer."""

    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, message, context=None):
        self.calls += 1
        from app.providers.base import LLMResult

        return LLMResult(
            intent="product_info",
            extracted_fields={"answer": "A canned answer with no grounding."},
        )


class _MiServiceStub:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = 0

    def handle(self, question: str, top_k: int = 3):
        self.calls += 1
        return self.result


class FakeSTT:
    def __init__(self, text=None, exc=None) -> None:
        self.text = text
        self.exc = exc
        self.calls = []

    def transcribe(self, audio_path, language_code="en-IN", output_format="verbatim"):
        self.calls.append({"path": str(audio_path), "language": language_code})
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(text=self.text)


def _client(stt=None, mi_service=None) -> TestClient:
    app = FastAPI()
    app.include_router(create_mi_chat_router(stt=stt, mi_service=mi_service))
    return TestClient(app)


def _grounded_service(provider=None) -> MIGroundedService:
    return MIGroundedService(
        answer_service=MIAnswerService(
            retriever=LantusRetriever(),
            llm_provider=provider or _FailingProvider(),
        )
    )


def _post_chat(client: TestClient, question: str):
    return client.post("/mi/chat", json={"question": question})


# ---- POST /mi/chat ----------------------------------------------------------


def test_grounded_product_question_with_citation_and_metadata():
    client = _client(mi_service=_grounded_service())
    response = _post_chat(client, "How should Lantus be stored?")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "allowed"
    assert data["category"] == "product_info"
    assert "package insert" in data["answer"]
    assert "page 11" in data["answer"]
    assert data["grounded"] is True
    assert data["supported"] is True
    assert 11 in data["source_page_numbers"]
    assert data["retrieved_chunk_ids"]
    assert data["urgent"] is False
    assert data["suggest_ae_report"] is False


def test_response_schema_is_exact_and_text_only():
    client = _client(mi_service=_grounded_service())
    data = _post_chat(client, "What is Lantus used for?").json()
    # The response model exposes only the documented safe fields: no AE
    # session/case keys and no audio/TTS keys can ever appear.
    assert set(data.keys()) == CHAT_KEYS


def test_source_structurally_isolated_from_ae_and_tts():
    source = Path("app/api/mi_chat.py").read_text(encoding="utf-8")
    assert "SessionStore" not in source
    assert "AEStateEngine" not in source
    assert "TimbreTTS" not in source
    assert "synthesize" not in source
    assert "audio_path" not in source


def test_personalized_dosing_is_blocked_without_retrieval():
    provider = _CountingProvider()
    client = _client(mi_service=_grounded_service(provider))
    response = _post_chat(client, "How many units should I take tonight?")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "blocked"
    assert "dosing" in data["answer"].lower()
    assert data["grounded"] is False
    assert data["source_page_numbers"] == []
    # The guardrail blocked before any answer generation: the provider was
    # never consulted and no retrieval passage was quoted.
    assert provider.calls == 0


def test_urgent_symptoms_take_precedence_over_retrieval():
    stub = _MiServiceStub()
    client = _client(mi_service=stub)
    response = _post_chat(
        client, "I am having severe difficulty breathing right now"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgent"] is True
    assert data["status"] == "urgent"
    assert "medical attention" in data["answer"].lower()
    assert data["grounded"] is False
    assert data["source_page_numbers"] == []
    # Routine product-info handling never ran.
    assert stub.calls == 0


def test_ae_report_intent_offers_ae_workflow_without_creating_a_case():
    client = _client(mi_service=_grounded_service())
    response = _post_chat(
        client, "I want to report a side effect after taking Lantus"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["suggest_ae_report"] is True
    assert set(data.keys()) == CHAT_KEYS  # no case/session fields exist
    assert data["answer"]  # safe bounded MI reply, never a report receipt


def test_provider_failure_falls_back_to_extractive_answer():
    client = _client(mi_service=_grounded_service(_FailingProvider()))
    response = _post_chat(client, "Does Lantus need refrigeration?")
    assert response.status_code == 200
    data = response.json()
    assert data["supported"] is True
    assert "package insert" in data["answer"]
    assert "refriger" in data["answer"].lower()


def test_empty_question_is_rejected():
    client = _client(mi_service=_MiServiceStub())
    for question in ("", "   "):
        response = _post_chat(client, question)
        assert response.status_code == 400
        assert "question" in response.json()["detail"].lower()


def test_missing_question_field_treated_as_empty():
    client = _client(mi_service=_MiServiceStub())
    response = client.post("/mi/chat", json={})
    assert response.status_code == 400
    assert "question" in response.json()["detail"].lower()


def test_mechanism_question_reaches_answer_generation_end_to_end():
    """"How does Lantus SoloStar work?" passes the guardrail gate and returns
    the grounded mode-of-action answer from page 7 - never the out-of-scope
    fallback message."""
    client = _client(mi_service=_grounded_service())
    response = _post_chat(client, "How does Lantus SoloStar work?")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == CHAT_KEYS
    assert data["status"] == "allowed"
    assert data["category"] == "product_info"
    assert data["grounded"] is True
    assert data["supported"] is True
    assert data["urgent"] is False
    assert data["suggest_ae_report"] is False
    assert "micro-precipitates" in data["answer"]
    assert "prolonged duration of action" in data["answer"]
    assert "page 7" in data["answer"]
    assert 7 in data["source_page_numbers"]
    assert "I can only answer general product information" not in data["answer"]


def test_pen_use_question_reaches_step_builder_end_to_end():
    """"How do I attach a new needle?" passes the guardrail gate and returns
    the source-grounded leaflet steps with their cross-page citation."""
    client = _client(mi_service=_grounded_service())
    response = _post_chat(
        client, "How do I attach a new needle to the SoloStar pen?"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "allowed"
    assert data["category"] == "product_info"
    assert data["grounded"] is True
    assert "Step 2. Attach the needle" in data["answer"]
    assert "Wipe the Rubber Seal with alcohol." in data["answer"]
    assert "pages 12, 13" in data["answer"]
    assert "I can only answer general product information" not in data["answer"]


# ---- POST /mi/transcribe ----------------------------------------------------


def _post_transcribe(client: TestClient, content: bytes = b"RIFFdata", language="en-IN"):
    return client.post(
        "/mi/transcribe",
        files={"audio_file": ("recording.wav", content, "audio/wav")},
        data={"language": language},
    )


def test_transcribe_success_returns_transcript_only():
    stt = FakeSTT(text="What is Lantus used for?")
    client = _client(stt=stt)
    response = _post_transcribe(client)
    assert response.status_code == 200
    data = response.json()
    assert data == {"transcript": "What is Lantus used for?"}
    assert stt.calls and stt.calls[0]["language"] == "en-IN"


def test_transcribe_rejects_empty_audio():
    client = _client(stt=FakeSTT(text="never reached"))
    response = _post_transcribe(client, content=b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_rejects_empty_transcript():
    client = _client(stt=FakeSTT(text="   "))
    response = _post_transcribe(client)
    assert response.status_code == 400


def test_transcribe_maps_upstream_empty_transcript_to_400():
    stt = FakeSTT(
        exc=PrismaAPIError(
            "Gnani STT returned an empty transcript.", status_code=200
        )
    )
    client = _client(stt=stt)
    response = _post_transcribe(client)
    assert response.status_code == 400
    assert "speech" in response.json()["detail"].lower()


def test_transcribe_maps_unsupported_language_to_400():
    stt = FakeSTT(
        exc=PrismaValidationError("Unsupported language_code 'xx-XX'.")
    )
    client = _client(stt=stt)
    response = _post_transcribe(client, language="xx-XX")
    assert response.status_code == 400
    assert "language" in response.json()["detail"].lower()


def test_transcribe_maps_upstream_failure_and_rate_limit():
    rate_limited = _client(
        stt=FakeSTT(exc=PrismaAPIError("slow down", status_code=429))
    )
    assert _post_transcribe(rate_limited).status_code == 429

    broken = _client(
        stt=FakeSTT(exc=PrismaAPIError("boom", status_code=500))
    )
    assert _post_transcribe(broken).status_code == 502

    unauthorized = _client(
        stt=FakeSTT(exc=PrismaAPIError("denied", status_code=403))
    )
    assert _post_transcribe(unauthorized).status_code == 500
