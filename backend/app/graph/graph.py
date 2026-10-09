"""app/graph/graph.py — the DigiWave X LangGraph.

Builds a :class:`StateGraph` whose state is the canonical :class:`AgentState`
from :mod:`app.graph.state`. The graph is strictly sequential: one node at a
time, conditional edges never branch in parallel, so last-writer-wins inside
each state section is safe.

The router is the only node that sets the routing decision. The verify step
has a conditional edge (``verify_gate``) that loops back to generate when
the verification failed and the retry budget (bounded by :data:`MAX_RETRIES`
= 2) has not been exhausted. Otherwise execution flows straight to
score_confidence and compose.

The graph accepts a *config* keyword on entry that carries request-scoped
metadata (turn_id, etc.) and returns a copy of the final state plus the
synthetic ``final_response`` on the output channel.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState, MAX_RETRIES
from app.graph.nodes.compose import compose
from app.graph.nodes.diagnose import diagnose
from app.graph.nodes.generate import generate
from app.graph.nodes.hint_escalate import hint_escalate
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.rerank import rerank
from app.graph.nodes.router import router
from app.graph.nodes.score_confidence import score_confidence
from app.graph.nodes.verify import verify
from app.graph.nodes.what_if import what_if

log = logging.getLogger(__name__)


def _is_fallback_draft(draft: str | None) -> bool:
    """Return True when the LLM could not produce a real answer.

    The generate node marks these drafts (its fallback text begins
    "I couldn't load the tutor model..."). Verification and the retry loop
    would otherwise loop forever on a draft that contains no circuit
    claims, so we short-circuit straight to composition.
    """
    if not draft:
        return True
    return draft.startswith("I couldn't load the tutor model")


def _verify_gate(state: AgentState) -> str:
    """Conditional edge for the verify step.

    Return the next node name. Only when verification failed, the draft is
    not a fallback, AND the retry budget is not exhausted do we loop back to
    generate.  Otherwise the pipeline advances to score_confidence.
    """
    verification = state.get("verification") or {}
    result = verification.get("verification") or {}
    retry_count = verification.get("retry_count", 0)
    draft = (state.get("tutor") or {}).get("draft_answer")
    if _is_fallback_draft(draft):
        return "score_confidence"
    if not result.get("passed", True) and retry_count < MAX_RETRIES:
        return "generate"
    return "score_confidence"


def _run_node(name: str):
    """Decorator-free factory: returns a LangGraph node function for *name*.

    Small helper so the graph file reads as a mapping rather than ten
    duplicate decorated functions, and so a single point of control keeps
    the node registry in sync with the task spec.
    """

    def node_fn(state: AgentState) -> AgentState:
        log.debug("graph node %s entered", name)
        try:
            result = globals()[name](state)
        except Exception as exc:  # noqa: BLE001
            log.exception("node %s failed: %s", name, exc)
            raise
        circuit = state.get("circuit") or {}
        if isinstance(circuit, list):
            circuit = circuit[0] if circuit else {}
        log.warning("node %s in thread %s; circuit_keys=%s", name, state.get("thread_id", "?"), sorted(circuit.keys()))
        return result

    return node_fn


def make_graph() -> StateGraph:
    """Build and compile the StateGraph for the tutoring pipeline."""
    graph = StateGraph(AgentState)

    # Register the ten workflow nodes. They are registered here, in the order
    # the task specifies, so a developer can audit the pipeline at a glance:
    #   1 router, 2 diagnose, 3 retrieve, 4 rerank, 5 generate,
    #   6 hint_escalate, 7 what_if, 8 verify, 9 score_confidence, 10 compose.
    nodes: dict[str, Any] = {
        # Orchestrator
        "router": _run_node("router"),
        # Diagnosis
        "diagnose": _run_node("diagnose"),
        # Knowledge / RAG
        "retrieve": _run_node("retrieve"),
        "rerank": _run_node("rerank"),
        "generate": _run_node("generate"),
        # Tutor
        "hint_escalate": _run_node("hint_escalate"),
        "what_if": _run_node("what_if"),
        # Critic / Verification
        "verify": _run_node("verify"),
        # Evaluation
        "score_confidence": _run_node("score_confidence"),
        # Output contract
        "compose": _run_node("compose"),
    }

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # ---- Edges ----
    graph.add_edge("router", "diagnose")
    graph.add_edge("diagnose", "retrieve")
    graph.add_edge("retrieve", "rerank")

    # Conditional: concept questions pull knowledge straight to generate;
    # wiring/diagnosis/hint questions go through... well, they all go to
    # generate after rerank. The real branching is what-if and the verify
    # loop, so keep it simple and deterministic.
    graph.add_edge("rerank", "generate")

    # The tutor's workflow (optional branches based on routing decision).
    graph.add_conditional_edges(
        "generate",
        _tutor_branch,
        {
            "hint_escalate": "hint_escalate",
            "what_if": "what_if",
            "verify": "verify",
            "score_confidence": "score_confidence",
            "compose": "compose",
        },
    )
    graph.add_edge("hint_escalate", "verify")
    graph.add_edge("what_if", "verify")

    graph.add_conditional_edges(
        "verify",
        _verify_gate,
        {
            "generate": "generate",
            "score_confidence": "score_confidence",
        },
    )
    graph.add_edge("score_confidence", "compose")

    graph.set_entry_point("router")
    graph = graph.compile()
    return graph


def _tutor_branch(state: dict[str, Any]) -> str:
    """Decide which next node the generate node hands off to.

    The router's ``intent`` decides the branch; absent intent defaults to a
    verify-then-compose loop (the safe default).
    """
    intent = state.get("intent")
    if intent == "hint":
        return "hint_escalate"
    if intent == "what_if":
        return "what_if"
    # concept / diagnosis / general: verify the draft, then compose unless the
    # critic found a discrepancy.
    return "verify"
