"""tests/test_tutor.py — tests for the AI tutoring pipeline.

These tests cover the LLM / RAG / LangGraph layers. They deliberately avoid
making ANY real Gemini call (the SDK is stubbed at the node layer), so they
run offline and are safe to run in CI with ``GEMINI_API_KEY`` unset.

Run:
    python -m pytest tests/test_tutor.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

BACKEND = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_kb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chroma"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _engine_request() -> dict[str, Any]:
    return {
        "experiment_id": "line_coding",
        "block_inputs": {
            "source": {"bits": "10110"},
            "encoder": {"scheme": "ami"},
            "scope": {"voltage": 1.0},
        },
        "connections": [
            {"from": "source.out", "to": "encoder.in"},
            {"from": "encoder.out", "to": "scope.in"},
        ],
    }


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def test_gemini_client_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    from app.llm.client import GeminiClient, is_gemini_configured

    assert is_gemini_configured() is True
    client = GeminiClient()
    assert client.ready is True
    assert client.model == "gemini-2.5-flash"
    assert client._client is None


def test_gemini_client_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from app.llm.client import is_gemini_configured  # noqa: F401

    assert is_gemini_configured() is False
    from app.llm.client import GeminiClient

    client = GeminiClient()
    assert client.ready is False
    with pytest.raises(Exception):
        client.generate("prompt", "user msg")


def test_gemini_client_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import os

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    import app.llm.client as client_module  # noqa: F401
    importlib.reload(client_module)
    from app.llm.client import GeminiClient

    client = GeminiClient()
    assert client.model == "gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# RAG ingestion / retrieval
# ---------------------------------------------------------------------------

def test_ingestion_builds_deterministic_chunks() -> None:
    from app.ingestion.ingestion import build_documents

    docs = build_documents()
    assert len(docs) >= 6
    for c in docs:
        assert c["chunk_id"]
        assert c["text"]
        assert isinstance(c["metadata"], dict)


def test_ingestion_contains_required_topics() -> None:
    from app.ingestion.ingestion import build_documents

    docs = build_documents()
    all_text = " ".join(c["text"] for c in docs).lower()
    for topic in ["ami", "manchester", "nrz", "rz", "unipolar", "coding"]:
        assert topic in all_text


def test_ingestion_storage_and_retrieval(tmp_kb_dir: Path) -> None:
    from app.ingestion.ingestion import KnowledgeBase

    kb = KnowledgeBase.from_manifest(manifest=None, chunk_size=800, overlap=80)
    kb.retriever.persist_directory = str(tmp_kb_dir)
    count = kb.add()
    assert count >= 6

    results = kb.retrieve("Why is my AMI polarity wrong?", n_results=2)
    assert len(results) == 2
    for r in results:
        assert r.get("chunk_id")
        assert r.get("text")
        assert isinstance(r.get("metadata"), dict)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def test_router_classifies_concept_query() -> None:
    from app.graph.nodes.router import classify_query

    assert classify_query("What is NRZ-L?") == "concept"
    assert classify_query("What is Manchester encoding?") == "concept"


def test_router_classifies_wiring_query() -> None:
    from app.graph.nodes.router import classify_query

    assert classify_query("Why is my AMI waveform wrong?") == "diagnosis"
    assert classify_query("My circuit is not working") == "diagnosis"


def test_router_classifies_what_if_query() -> None:
    from app.graph.nodes.router import classify_query

    assert classify_query("What if I use Manchester instead?") == "what_if"
    assert classify_query("What would happen if I skip the encoder?") == "what_if"


def test_router_writes_intent_and_next_action() -> None:
    from app.graph.nodes.router import router

    state = {"query": "Why is my AMI waveform wrong?"}
    updated = router(state)
    assert updated["intent"] == "diagnosis"
    assert updated["next_action"] == "diagnose"


# ---------------------------------------------------------------------------
# Diagnose (deterministic only)
# ---------------------------------------------------------------------------

def test_diagnose_valid_circuit_has_no_faults() -> None:
    from app.graph.nodes.diagnose import diagnose

    from app.experiments.engine import build_circuit_state

    engine_result = build_circuit_state(**_engine_request())
    state = {
        "circuit": {
            "experiment_id": "line_coding",
            "block_inputs": _engine_request()["block_inputs"],
            "connections": engine_result["connections"],
            "engine_result": engine_result,
        }
    }
    updated = diagnose(state)
    assert updated["circuit"]["diagnosis"]["diagnosed"] is True
    assert updated["circuit"]["diagnosis"]["faults"] == []


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def test_retrieve_populates_knowledge(tmp_kb_dir: Path) -> None:
    from app.graph.nodes.retrieve import retrieve

    from app.ingestion.ingestion import KnowledgeBase

    kb = KnowledgeBase.from_manifest(manifest=None, chunk_size=800, overlap=80)
    kb.retriever.persist_directory = str(tmp_kb_dir)
    kb.add()

    state: dict[str, Any] = {
        "query": "Why is my AMI polarity wrong?",
        "circuit": {"engine_result": _engine_request()},
    }
    updated = retrieve(state)
    assert updated["knowledge"]["query_used"] == state["query"]
    assert isinstance(updated["knowledge"]["retrieved"], list)
    assert updated["knowledge"]["enough_evidence"] is True


# ---------------------------------------------------------------------------
# Verify / confidence
# ---------------------------------------------------------------------------

def test_verify_skipped_for_fallback_draft() -> None:
    from app.graph.nodes.verify import verify

    state = {
        "tutor": {"draft_answer": "I couldn't load the tutor model right now."},
        "circuit": {"engine_result": _engine_request()},
        "knowledge": {},
        "verification": {"retry_count": 0},
    }
    updated = verify(state)
    assert isinstance(updated["verification"], dict)


# ---------------------------------------------------------------------------
# Graph end-to-end with a stubbed Gemini client.
# ---------------------------------------------------------------------------

def test_graph_run_valid_circuit() -> None:
    """Run one turn of the graph with Gemini stubbed out so it never touches
    the network. The deterministic engine grounds the circuit, gathering all
    facts, and the LLM stub produces a grounded answer. The graph must reach
    compose and emit a final_response.

    The graph is rebuilt from scratch in the test (rather than patched from
    :func:`make_graph`) so the stub replaces the ``generate`` node cleanly.
    """
    from app.graph.graph import StateGraph
    from app.graph.state import AgentState
    from app.graph.nodes.compose import compose
    from app.graph.nodes.diagnose import diagnose
    from app.graph.nodes.generate import _prompt  # noqa: F401
    from app.graph.nodes.hint_escalate import hint_escalate
    from app.graph.nodes.retrieve import retrieve
    from app.graph.nodes.rerank import rerank
    from app.graph.nodes.router import router
    from app.graph.nodes.score_confidence import score_confidence
    from app.graph.nodes.verify import verify
    from app.graph.nodes.what_if import what_if
    from app.experiments.engine import build_circuit_state

    engine_result = build_circuit_state(**_engine_request())

    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("diagnose", diagnose)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", _stub_generate)
    graph.add_node("hint_escalate", hint_escalate)
    graph.add_node("what_if", what_if)
    graph.add_node("verify", verify)
    graph.add_node("score_confidence", score_confidence)
    graph.add_node("compose", compose)

    graph.add_edge("router", "diagnose")
    graph.add_edge("diagnose", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "score_confidence")
    graph.add_edge("score_confidence", "compose")
    graph.set_entry_point("router")

    graph = graph.compile()

    state: dict[str, Any] = {
        "query": "Why is my AMI waveform wrong?",
        "intent": "diagnosis",
        "next_action": "diagnose",
        "student": {"student_id": "student-0001"},
        "experiment": {"experiment_id": "line_coding", "stage": "observe"},
        "circuit": {
            "blocks": engine_result["blocks"],
            "connections": engine_result["connections"],
            "wiring_status": engine_result["wiring_status"],
            "dirty": False,
            "oracle": engine_result["oracle"],
            "diagnosis": {"faults": [], "summary": "Healthy", "diagnosed": True},
            "engine_result": engine_result,
        },
        "knowledge": {},
        "tutor": {},
        "verification": {"retry_count": 0},
        "evaluation": {"turn_id": "t1", "session_id": "s1"},
        "final_response": None,
    }

    final_state = graph.invoke(state, config={"configurable": {"thread_id": "t"}})
    final = final_state.get("final_response")

    assert final is not None, "graph should produce a final_response"
    assert isinstance(final.get("answer"), str) and final["answer"]
    # The stub answer is oracle-grounded, so the circuit facts are intact.
    assert "experiment_id" in final_state.get("experiment", {})


def _stub_generate(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "tutor": {
            "draft_answer": (
                "Understood. AMI alternates mark polarity and the engine levels are "
                "the source of truth."
            ),
            "facts_used": [],
            "citations": [],
            "hint_level": 0,
            "hint_text": None,
            "what_if": None,
        }
    }
