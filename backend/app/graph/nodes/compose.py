"""app/graph/nodes/compose.py — node 10 of 10.

Runs when the graph finishes. Assembles the final structured response from
the canonical homes of every fact:

    * answer       - the tutor's prose draft (optionally post-processed)
    * diagnosis_summary - deterministic summary from the diagnose node
    * faults       - the engine's deterministic faults
    * citations    - chunk_ids the generator actually used
    * what_if      - optional hypothetical result
    * confidence   - from score_confidence
    * stage / next_action - lifecycle + routing guidance for the UI

The compose node is the single output contract. It never reads from a
secondary source to fill the response: every field comes from its one
canonical home, which keeps the response auditable.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.graph.state import ExperimentStage, HintLevel

log = logging.getLogger(__name__)


def compose(state: dict[str, Any]) -> dict[str, Any]:
    """Node 10: produce the final response snapshot."""
    tutor = state.get("tutor") or {}
    circuit = state.get("circuit") or {}
    knowledge = state.get("knowledge") or {}
    verification = state.get("verification") or {}

    answer = tutor.get("draft_answer")
    if not answer:
        answer = "I'm not sure how to answer that yet. Please ask about the encoding scheme, wiring, or waveform — I'll point you at the right explanation."

    diagnosis_summary = (circuit.get("diagnosis") or {}).get("summary")
    faults = (circuit.get("diagnosis") or {}).get("faults") or []
    citations = [
        c.get("chunk_id") for c in (knowledge.get("reranked") or []) if c.get("chunk_id")
    ]
    what_if = tutor.get("what_if")

    confidence = (state.get("verification") or {}).get("confidence")

    return {
        "final_response": {
            "turn_id": uuid.uuid4().hex[:12],
            "answer": answer,
            "hint_level": (state.get("tutor") or {}).get("hint_level", 0),
            "hint_text": (state.get("tutor") or {}).get("hint_text"),
            "diagnosis_summary": diagnosis_summary,
            "faults": faults,
            "citations": citations,
            "what_if": what_if,
            "confidence": confidence,
            "stage": (state.get("experiment") or {}).get("stage", "observe"),
            "next_action": state.get("next_action"),
            "enough_evidence": (state.get("knowledge") or {}).get("enough_evidence"),
        }
    }
