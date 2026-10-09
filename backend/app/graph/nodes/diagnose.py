"""app/graph/nodes/diagnose.py — node 2 of 10.

Performs a deterministic diagnosis over the student's circuit before any
LLM call. The diagnosis is built only from:

* the Experiment Engine's ``build_circuit_state`` result (``wiring_status``,
  ``oracle``, ``blocks``), and
* the misconception catalog in ``line_coding.json``.

The LLM is ONLY responsible for phrasing; it never invents a circuit fact
here. If the circuit is healthy or not yet run, this node records that as
well instead of returning an empty fault list.
"""

from __future__ import annotations

import logging
from typing import Any

from app.experiments.engine import build_circuit_state

log = logging.getLogger(__name__)


def diagnose(state: dict[str, Any]) -> dict[str, Any]:
    """Node 2: deterministic diagnosis of the current circuit state.

    The diagnosis is built ONLY from the Experiment Engine's output:

        * wiring_status  -> missing / unexpected connections (deterministic)
        * oracle         -> scheme, levels, dc_component, transitions

    The LLM is responsible for phrasing; it never invents a circuit fact
    here. If the circuit is healthy or not yet run, the node records that
    instead of returning an empty fault list.
    """
    circuit = state.get("circuit", {})
    # experiment_id lives on the experiment section; circuit keeps the
    # engine's own circuit object (engine_result).
    experiment_id = state.get("experiment", {}).get("experiment_id")
    if not experiment_id:
        experiment_id = circuit.get("experiment_id")

    # The graph guarantees the engine has already built this state (see
    # graph.py: nodes do not rerun the engine themselves).  If a caller
    # supplies a bare state without a cached engine result, rebuild it from
    # the raw circuit below so the diagnosis stays deterministic and in
    # sync with the student's exact wiring.
    circuit_state = circuit.get("engine_result")
    if not circuit_state:
        # Fall back to the raw circuit fields the graph is supposed to have
        # already set; this branch only triggers in tests/raw calls.
        block_inputs = circuit.get("block_inputs", {})
        connections = circuit.get("connections", [])
        circuit_state = build_circuit_state(
            experiment_id=experiment_id,
            block_inputs=block_inputs,
            connections=connections,
        )

    engine_wiring = circuit_state.get("wiring_status") or {}
    engine_oracle = circuit_state.get("oracle")
    blocks = circuit.get("blocks", [])
    connections = circuit.get("connections", [])

    # Deterministic findings: always use the engine's numbers.
    faults: list[dict[str, Any]] = []

    if engine_wiring.get("valid") is not True:
        for m in engine_wiring.get("missing", []):
            faults.append(
                {
                    "kind": "wiring",
                    "block_id": m.get("from", "").split(".")[0] if "." in m.get("from", "") else None,
                    "misconception_id": None,
                    "name": f"Missing {m['from']} -> {m['to']}",
                    "description": f"Required connection {m['from']} -> {m['to']} is absent.",
                    "evidence": f"missing {m['from']} -> {m['to']}",
                    "severity": "error",
                }
            )
        for u in engine_wiring.get("unexpected", []):
            faults.append(
                {
                    "kind": "wiring",
                    "block_id": u.get("from", "").split(".")[0] if "." in u.get("from", "") else None,
                    "misconception_id": None,
                    "name": f"Unexpected {u['from']} -> {u['to']}",
                    "description": f"Connection {u['from']} -> {u['to']} is not required.",
                    "evidence": f"unexpected {u['from']} -> {u['to']}",
                    "severity": "warning",
                }
            )

    # If the circuit is healthy, record a clean summary so the generator
    # can confirm with the oracle rather than guessing.
    if engine_wiring.get("valid") is True and engine_oracle:
        pass

    summary = _diagnosis_summary(faults, engine_oracle)

    return {
        "circuit": {
            "wiring_status": engine_wiring,
            "oracle": engine_oracle,
            "diagnosis": {
                "faults": faults,
                "summary": summary,
                "diagnosed": bool(faults) or engine_wiring.get("valid") is True,
            },
        }
    }


def _diagnosis_summary(faults: list[dict[str, Any]], oracle: dict[str, Any] | None) -> str:
    """Compose a short, deterministic summary of the diagnosis.

    The generator expands this into pedagogy afterwards. This keeps the LLM
    away from the circuit numbers entirely.
    """
    if not faults:
        if oracle:
            scheme = oracle.get("scheme", "?")
            transitions = oracle.get("transitions")
            dc = oracle.get("dc_component")
            parts = [f"scheme={scheme}", f"transitions={transitions}"]
            if dc is not None:
                parts.append(f"dc={dc:.3f}")
            return "Healthy circuit: " + ", ".join(parts)
        return "No faults detected in the current circuit."
    return f"{len(faults)} fault(s) found: " + "; ".join(f["name"] for f in faults[:3])


def _diagnosis_summary(faults: list[dict[str, Any]], oracle: dict[str, Any] | None) -> str:
    """Compose a short, machine-readable summary of the diagnosis.

    The generator expands this into pedagogy afterwards. This keeps the LLM
    away from the circuit numbers entirely.
    """
    if not faults:
        if oracle:
            return f"Healthy circuit: scheme={oracle.get('scheme')}, transitions={oracle.get('transitions')}, dc={oracle.get('dc_component'):.3f}"
        return "No faults detected in the current circuit."
    return f"{len(faults)} fault(s) found: " + "; ".join(f["name"] for f in faults[:3])
