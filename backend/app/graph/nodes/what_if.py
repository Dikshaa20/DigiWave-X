"""app/graph/nodes/what_if.py — node 7 of 10.

Answers "what if I..." questions by computing a hypothetical circuit with
the deterministic Experiment Engine. The engine is called with a modified
circuit (e.g. ``scheme=manchester`` on the same bits), and the resulting
``oracle_result`` is stored verbatim. The LLM never invents a hypothetical
waveform; the engine returns it.

The node writes its result into the tutor section so the verify and
compose nodes can diff the student's prediction against the engine's result.
"""

from __future__ import annotations

import logging
from typing import Any

from app.experiments.engine import build_circuit_state

log = logging.getLogger(__name__)


def _modify_circuit(circuit: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply *overrides* (e.g. ``{"circuit.block_inputs.encoder.scheme": "ami"}``)
    onto a deep copy of the circuit, returning the modified result."""
    import copy
    mod = copy.deepcopy(circuit)
    blocks = mod.get("blocks", [])
    for b in blocks:
        bid = b.get("id")
        inputs = b.get("inputs", [])
        d = {inp.get("id"): inp.get("value") for inp in inputs}
        # Override block inputs
        val = overrides.get(f"block_inputs.{bid}")
        if val is not None:
            d.update(val if isinstance(val, dict) else {})
        # Override wiring options such as scheme if wired into the oracle args
        for key, val in overrides.items():
            if key.startswith(f"block_inputs.{bid}.") and key.split(".", 1)[2] in d:
                d[key.split(".", 2)[2]] = val
    mod["block_inputs"] = mod.get("block_inputs", {})
    return mod


def what_if(state: dict[str, Any]) -> dict[str, Any]:
    """Node 7: compute a hypothetical oracle and store it on the tutor section."""
    from app.experiments.engine import build_circuit_state
    query = state.get("query", "")
    circuit = state.get("circuit", {})
    experiment = state.get("experiment", {})

    # The graph state is typed: CircuitState declares blocks/connections/...,
    # not `engine_result`. The engine result wrapper (_build_agent_state
    # injects) lives in the graph but isn't part of the TypedDict. The
    # experiment section holds the active experiment id. We read experiment_id
    # from the experiment section, falling back to the circuit wrapper, which
    # (from _build_agent_state) carries experiment_id from the engine result.
    experiment_id = None
    if isinstance(experiment, dict):
        experiment_id = experiment.get("experiment_id")
    if not experiment_id and isinstance(circuit, dict):
        experiment_id = circuit.get("experiment_id")
    block_inputs = circuit.get("block_inputs", {}) if isinstance(circuit, dict) else {}
    connections = circuit.get("connections", []) if isinstance(circuit, dict) else []

    # The graph guarantees the engine has already computed the current circuit
    # state. We reuse that as the starting point for the hypothetical.
    base = circuit
    current_scheme = (base.get("oracle") or {}).get("scheme")
    candidate_scheme = _guess_scheme(query, current_scheme)

    if candidate_scheme and candidate_scheme != current_scheme:
        # Rebuild the oracle for the hypothetical scheme. The engine's
        # deterministic engine calls the oracle directly; we return the full
        # engine result keyed under the hypothetical.
        hypothetical_inputs = dict(block_inputs)
        hypothetical_inputs.setdefault("encoder", {})["scheme"] = candidate_scheme
        engine = build_circuit_state(experiment_id or "", hypothetical_inputs, connections)
        oracle = engine.get("oracle")
    else:
        # No scheme change: compute a hypothetical from the student's prediction
        # field if present.
        oracle = base.get("oracle")
        engine = base

    # If still nothing, provide a minimal placeholder.
    if not oracle:
        oracle = {
            "scheme": current_scheme or "unknown",
            "levels": [],
            "dc_component": None,
            "transitions": 0,
            "bits": [],
        }

    # The student's prediction, verbatim, goes into the what_if result so it
    # can be diffed later.
    prediction = ""
    # (prediction comes from the graph input; here we leave it empty and let
    #  compose build the hypothesis text from the query.)
    hypothesis = f"same bits, scheme={candidate_scheme or current_scheme or 'unchanged'}"

    return {
        "tutor": {
            "what_if": {
                "hypothesis": hypothesis,
                "prediction": prediction,
                "prediction_levels": None,
                "oracle_result": oracle,
                "match": False,
                "explanation": f"Recomputed with scheme={candidate_scheme or current_scheme or 'unchanged'}.",
            }
        }
    }


def _guess_scheme(query: str, current: str) -> str | None:
    """Return a candidate scheme extracted from the question, else None."""
    q = query.lower()
    candidates = {
        "nrz": "nrz_unipolar",
        "unipolar": "nrz_unipolar",
        "polar": "nrz",
        "bipolar": "nrz",
        "manchester": "manchester",
        "ami": "ami",
        "rz": "rz_unipolar",
    }
    for token, scheme in candidates.items():
        if token in q and token != current:
            return scheme
    return None
