"""app/graph/nodes/hint_escalate.py — node 6 of 10.

Turns the student's need into a ladder of hints. The ladder is keyed by a
``HintLevel`` (0..3, from :data:`app.graph.state.HintLevel`), so the answer
can escalate one step at a time (up to :data:`CONFIDENCE_THRESHOLD`/retry
limits) without repeating the same nudge.

The hints are built from:
1. the deterministic diagnosis (a missing connection, a misconception, a
   scheme mismatch),
2. the retrieved knowledge, and
3. the current mastery record if the student profile is available.

This node is deterministic where possible, so a wiring fault always produces
the same hint ladder regardless of the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import HintLevel

log = logging.getLogger(__name__)


def _level_text(level: int, faults: list[dict[str, Any]], query: str) -> str:
    """Return a hint at *level* (0 = nudge, 1 = guiding question, 2 = worked
    example, 3 = full answer)."""
    if not faults:
        return "No fault detected — this circuit runs. Try the oscilloscope stats panel to see the DC component and transitions."

    top = faults[0]
    if level == 0:
        return f"A good place to start: check whether {top['name']} is present."
    if level == 1:
        return f"Think about the required connections. Which wire is missing before this signal can reach the scope?"
    if level == 2:
        return f"Re-read the wiring kit: {top['evidence']}. Add the missing connection, then re-check the circuit."
    return f"Here is the fix: connect {top['from']} to {top['to']} as required."


def hint_escalate(state: dict[str, Any]) -> dict[str, Any]:
    """Node 6: set the hint level and text on the tutor section."""
    query = state.get("query", "")
    faults = (state.get("circuit") or {}).get("diagnosis") or {}
    faults = faults.get("faults") or []
    # The current hint level is written by the verify_gate when he retries.
    hint_level = state.get("tutor", {}).get("hint_level", 0)

    # Escalate at most one level per retry (bounded by MAX_RETRIES in the
    # graph). Clamp to the 0..3 ladder.
    hint_level = min(max(hint_level, 0), 3)

    # If the circuit is healthy, a "hint" question is really a concept
    # request, so return a gentle explanation nudge.
    if not faults:
        hint_level = 0
        hint_text = "Start by understanding the encoding scheme. The oscilloscope shows the waveform and the stats panel shows the DC component and transitions — compare them to the scheme you chose."
    else:
        hint_text = _level_text(hint_level, faults, query)

    # If the student has a mastery record, use it to pick a gentler tone.
    mastery = (state.get("student") or {}).get("mastery", {})
    if mastery:
        known = [k for k, v in mastery.items() if isinstance(v, dict) and v.get("correct", 0) > 0]
        if known and hint_level < 2:
            hint_level = 0  # never start a student at a worked example

    return {
        "tutor": {
            "hint_level": hint_level,
            "hint_text": hint_text,
        }
    }
