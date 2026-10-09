"""app/graph/nodes/router.py — node 1 of 10.

Classifies the student's free text into a routing decision so later nodes
know what the question really is. The classifier is a small deterministic
keyword heuristic over the query. It is deliberately conservative: if the
query is ambiguous the router classifies it as ``general`` and lets later
nodes decide.

Results are written into the ``intent`` channel of AgentState so every
downstream node can read them without re-tokenising the query.

The five conceptual roles (Orchestrator, Diagnosis, Knowledge/RAG,
Tutor, Critic) are folded into the ten *workflow* nodes below; this node is
purely the Orchestrator's first act.
"""

from __future__ import annotations

import re
from typing import Any

from app.graph.state import Intent

# Deterministic keyword table: each key is a concept, each list of needles
# triggers that concept. Order matters — more specific phrases win.
CONCEPT_TABLE: list[tuple[str, list[str]]] = [
    ("what_if", ["what if", "what would happen if", "what happens if", "if i", "what would my", "hypothetial", "counterfactual"]),
    ("hint", ["hint", "give me a hint", "can you hint", "how do i fix", "clue", "nudge", "help me"]),
    ("diagnosis", ["wrong", "not working", "broken", "fault", "error", "miswired", "missing", "why is it", "why does it not", "why is my", "incorrect", "doesn't work", "not showing"]),
    ("concept", ["what is", "how does", "explain", "why does", "what's", "how do", "what are", "what makes", "difference between", "line coding", "encoding", "waveform", "nrz", "manchester", "ami", "rz", "unipolar"]),
    ("general", []),
]

# Routing decision as written into AgentState.
RoutingDecision = dict[str, str]  # keys are the ten node names


def classify_query(query: str) -> str:
    """Return the highest-level intent for *query*.

    Maps free text onto the ``Intent`` vocabulary (see app/graph/state.py),
    which the orchestrator's conditional edges use to choose the next
    node. Order matters: more specific patterns (what-if, hint) are checked
    before broad ones (concept, diagnosis) so that a question like
    "what if i skip the encoder" is routed to what_if rather than
    diagnosis/concept.
    """
    normalized = query.lower().strip()
    for intent, needles in CONCEPT_TABLE:
        if intent == "general":
            continue
        for needle in needles:
            if re.search(re.escape(needle), normalized):
                return intent
    return "general"


def router(state: dict[str, Any]) -> dict[str, Any]:
    """Node 1: examine *state["query"]* and set intent + next_action.

    Returns a shallow partial update that the graph reducer folds into the
    existing AgentState. Nothing else changes.
    """
    query = state.get("query", "")
    intent = classify_query(query)

    # Build the routing map the task specifies. "intent" is the
    # orchestrator's classification (one of Intent); "next_action" is the
    # workflow node the graph will run next (a ten-node name).
    next_action: str | None = None
    if intent == "what_if":
        # A hypothesis question: run the deterministic engine for the
        # hypothetical, verify, then compose.
        next_action = "what_if"
    elif intent == "hint" or "hint" in query.lower() or "fix" in query.lower() or "help" in query.lower():
        # A hint / remediation question: pull knowledge, generate a draft,
        # then escalate hints and verify in sequence.
        next_action = "hint_escalate"
    elif intent == "diagnosis":
        # A wiring / fault question: run diagnose, then the normal path.
        next_action = "diagnose"
    elif intent == "concept":
        # A concept / explanation question: pull knowledge and generate.
        next_action = "retrieve"
    else:
        # general / ambiguous: ground in circuit facts (diagnose)
        # before deciding the rest of the pipeline.
        next_action = "diagnose"

    return {
        "intent": intent,
        "next_action": next_action,
    }
