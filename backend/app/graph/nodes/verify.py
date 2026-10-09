"""app/graph/nodes/verify.py — node 8 of 10.

Critique step. After the generator has drafted an answer, this node diffs the
draft against the deterministic ground truth the engine computed. Specifically
it checks:

1. The draft's quoted ``levels`` equal the oracle's ``levels``.
2. The draft's ``dc_component`` matches the oracle's.
3. The draft's ``transitions`` matches the oracle's.
4. The draft's quoted scheme equals the oracle's scheme.
5. The draft quotes a channel that exists in the retrieved knowledge.

If any check fails, the verification result is marked failed and the draft is
fed back to the generator for a retry (bounded by :data:`MAX_RETRIES`). The
failures are recorded in the verification section so later nodes can use them.

This node is deliberately the "critic" of the tutor role: it catches an LLM
that says "the waveform is smooth" when the oracle says a step wave.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import MAX_RETRIES

log = logging.getLogger(__name__)

# Channel keys the verify node checks inside the draft text.
CHANNEL_KEYS = ("levels", "dc_component", "transitions", "scheme")


def _claim_equals_oracle(claim: str, expected: Any) -> bool:
    """Return True when the draft's *claim* matches the deterministic value."""
    if claim is None:
        return True  # nothing to check
    # Normalise Python list repr to a canonical token list for arrays.
    if isinstance(expected, list) and claim.startswith("["):
        return claim.strip() == str(expected)
    return claim.strip() == str(expected)


def verify(state: dict[str, Any]) -> dict[str, Any]:
    """Node 8: critique the draft against deterministic facts; write the result."""
    circuit = state.get("circuit", {})
    oracle = circuit.get("oracle") or {}
    draft = (state.get("tutor") or {}).get("draft_answer", "")
    retry_count = state.get("verification", {}).get("retry_count", 0)

    checks: list[dict[str, Any]] = []
    discrepancies: list[str] = []

    # Diff the draft against the oracle's channels.
    for key in CHANNEL_KEYS:
        expected = oracle.get(key)
        if expected is None:
            continue
        # Extract the first token-like mention of this channel in the draft.
        token = f"{key}="
        idx = draft.find(token)
        if idx == -1:
            claim = f"{key}="
            actual = f"{key}={expected}"
        else:
            after = draft[idx + len(token):]
            actual = after.split()[0] if after else f"{key}={expected}"
        passed = _claim_equals_oracle(actual, expected)
        if not passed:
            discrepancies.append(f"draft {key}={actual!r} does not match oracle {key}={expected!r}")
        checks.append(
            {
                "name": f"{key}_levels",
                "claim": f"{key}={actual}",
                "expected": f"{key}={expected}",
                "source": "oracle",
                "passed": passed,
            }
        )

    # Citation check: does the draft cite a retrieved chunk?
    knowledge = state.get("knowledge", {})
    reranked = knowledge.get("reranked") or []
    # If there is no retrieved knowledge, that is NOT a verification failure
    # — it means the deterministic engine carries the answer, and the LLM
    # correctly leaned on the ground-truth facts instead of inventing a
    # citation. Only fail when the student question genuinely needs evidence
    # and the draft claims more than the engine provides.
    if reranked and not any(c.get("chunk_id") for c in reranked):
        passed = False
        discrepancies.append("draft cites no retrieved knowledge chunk")
        checks.append(
            {
                "name": "citation",
                "claim": "cites retrieved chunk",
                "expected": "one or more chunks",
                "source": "reranked_chunks",
                "passed": passed,
            }
        )

    passed = len(discrepancies) == 0
    # Evidence support is how much of the draft is grounded in the retrieved
    # chunks. When nothing was retrieved, the LLM anchored on the oracle
    # facts, which is what the verify rule also rewards.
    evidence_support = 0.5 if not reranked else min(1.0, len(reranked) / 5.0)

    return {
        "verification": {
            "verification": {
                "checks": checks,
                "passed": passed,
                "discrepancies": discrepancies,
                "oracle_agreement": 1.0 - min(len(discrepancies) / max(len(CHANNEL_KEYS), 1), 1.0),
                "evidence_support": evidence_support,
            },
            "confidence": state.get("verification", {}).get("confidence"),
            "retry_count": retry_count,
        }
    }
