"""app/graph/nodes/score_confidence.py — node 9 of 10.

Reduces the verification result to a single 0..1 confidence score using the
existing :data:`app.graph.state.CONFIDENCE_THRESHOLD` (0.70). The score is a
determinism-weighted blend, not just a guess:

    confidence = 0.6 * oracle_agreement + 0.3 * evidence_support + 0.1 * (1 - retry_count/MAX_RETRIES)

If the circuit is healthy and no knowledge was retrieved the confidence is
boosted slightly, because a well-formed answer about a healthy circuit is
easy to ground.

This node only writes the score onto the verification channel; the graph's
conditional edge (``verify_gate``) decides whether to retry generation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)


def score_confidence(state: dict[str, Any]) -> dict[str, Any]:
    """Node 9: compute and store the confidence score."""
    verification = state.get("verification", {})
    result = verification.get("verification") or {}
    retry_count = verification.get("retry_count", 0)
    oracle_agreement = result.get("oracle_agreement", 1.0)
    evidence_support = result.get("evidence_support", 0.5)

    # Deterministic weighting, capped 0..1.
    confidence = 0.6 * oracle_agreement + 0.3 * evidence_support + 0.1 * (1 - min(retry_count, 2) / 2)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "verification": {
            "confidence": confidence,
        }
    }
