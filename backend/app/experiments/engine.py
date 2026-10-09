"""
app/experiments/engine.py — deterministic experiment engine for DigiWave X.

Reads app/experiments/<experiment_id>.json and produces the ENGINE-OWNED
sections of the graph state:

    circuit.blocks          -> instantiated kit blocks with student-set inputs
    circuit.connections     -> student-built wiring
    circuit.wiring_status   -> deterministic validation vs required_connections
    circuit.oracle          -> deterministic ground truth (app.oracle.*)

No LLM is involved anywhere here. The engine is a pure function of
(experiment JSON, parsed block inputs, parsed connections) and returns
plain dicts/JSON-serializable values so it drops straight into a state channel.

Design notes
------------
* Wiring status (missing / unexpected connections) is deterministic fact, so
  it lives here — never in an LLM call.
* The oracle result is always JSON-serializable (lists, floats, ints, strings).
  NumPy arrays are converted at the boundary by the engine, never returned.
* The misconception 'detect' tags in the JSON are NOT implemented here. They
  are interpreted by the Diagnosis Agent (app/graph/nodes/diagnosis.py), which
  maps each tag to the matching deterministic oracle function. This engine
  stays a pure data + oracle layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.oracle.encoders import ENCODERS, encode

# ---------------------------------------------------------------------------
# Small deterministic helpers that the experiment JSON's "validation" fields
# express (currently the only instance is the bit-data regex in line_coding.json)
# ---------------------------------------------------------------------------


def _validate_string_field(value: str, validation: str | None, field_name: str) -> None:
    """Run a JSON-schema-style validation rule against a string input."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if validation:
        # Normalize the rule to be fully-anchored so re.match checks the whole
        # string (the JSON may store '^...' and '...$' around the pattern; we
        # ignore any outer anchors and enforce them ourselves so invalid input
        # like '101a0' is rejected, not loosely matched).
        pattern = validation.strip()
        if not (pattern.startswith("^") and pattern.endswith("$")):
            pattern = "^" + pattern + "$"
        if not re.match(pattern, value):
            raise ValueError(
                f"{field_name} failed validation rule: {validation!r} (got {value!r})"
            )


def _coerce_number(value: Any, field_name: str, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {number}")
    return number


# ---------------------------------------------------------------------------
# Wiring model (mirrors required_connections[] / ports[] / inputs[] in the JSON)
# ---------------------------------------------------------------------------


def _split_port_ref(port_ref: str) -> tuple[str, str]:
    """'source.out' -> ('source', 'out'). Raises on malformed refs."""
    if "." not in port_ref:
        raise ValueError(f"port ref {port_ref!r} must be <block>.<port>")
    block_id, port_id = port_ref.split(".", 1)
    if not block_id or not port_id:
        raise ValueError(f"port ref {port_ref!r} is incomplete")
    return block_id, port_id


def _format_ref(ref: tuple[str, str] | str) -> str:
    """Normalize a (block, port) tuple or a 'block.port' string back to
    the canonical 'block.port' string used by the wiring kit and the
    experiment JSON."""
    if isinstance(ref, str):
        return ref
    return f"{ref[0]}.{ref[1]}"


def _key(a: tuple[str, str]) -> tuple[str, str]:
    """Normalize a connection key to a sortable tuple."""
    return a


def _resolve_block(inputs: dict[str, Any], block_id: str, port_ref: str) -> Any:
    """Look up the value of a block input from the parsed state, e.g.
    'encoder.scheme' -> inputs['encoder']['scheme'] or 'source.bits'."""
    block_id, input_id = port_ref.split(".", 1)
    if block_id not in inputs:
        raise KeyError(f"block {block_id!r} not in parsed inputs")
    return inputs[block_id].get(input_id)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExperimentNotFound(KeyError):
    """Raised when the experiment id is not in app/experiments/."""


def _load_experiment(experiment_id: str, experiments_dir: Path | None = None) -> dict[str, Any]:
    directory = experiments_dir or Path(__file__).parent
    path = directory / f"{experiment_id}.json"
    if not path.exists():
        raise ExperimentNotFound(f"experiment {experiment_id!r} not found in {directory}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _parse_block_inputs(raw_blocks: list[dict], parsed_inputs: dict[str, dict]) -> dict[str, Any]:
    """Validate the student's block input values against the schema in the JSON.

    A block input may be left *unspecified*; when it has a JSON "default" and
    the student did not supply a value we fall back to that default instead of
    raising "missing input". This keeps the engine one-step-behind the kit:
    every default that the UI already fills in for the student is honoured.

    Returns a dict keyed by block id, e.g. {'source': {'bits': '10110'},
    'encoder': {'scheme': 'ami', 'voltage': 1.0}} (voltage only lives on the
    scope, but we store all student values here for convenience).
    """
    blocks: dict[str, dict[str, Any]] = {}
    for raw in raw_blocks:
        block_id = raw["id"]
        schema = {item["id"]: item for item in raw.get("inputs", [])}
        inputs = {}
        for inp_id, item in schema.items():
            inp_type = item.get("type")
            value = parsed_inputs.get(block_id, {}).get(inp_id)
            # Default: honour the JSON "default" when the student did not set a
            # value for this input. If there is no default either, raise.
            if value is None:
                default = item.get("default")
                if default is None:
                    raise KeyError(f"missing input {block_id}.{inp_id}")
                value = default
            if inp_type == "bitstring":
                _validate_string_field(value, item.get("validation"), inp_id)
                # normalize whitespace so '10110' / '1 0 1 1 0' both work
                inputs[inp_id] = re.sub(r"\s+", "", value)
            elif inp_type == "select":
                options = set(item.get("options", []))
                if value not in options:
                    raise ValueError(
                        f"{block_id}.{inp_id} must be one of {sorted(options)}, got {value!r}"
                    )
                inputs[inp_id] = value
            elif inp_type == "number":
                inputs[inp_id] = _coerce_number(
                    value, f"{block_id}.{inp_id}", minimum=item.get("min")
                )
            else:
                raise ValueError(
                    f"unsupported input type {inp_type!r} for {block_id}.{inp_id}"
                )
        blocks[block_id] = inputs
    return blocks


def _normalize_connection(connection) -> tuple[str, str]:
    """Normalize a connection to a (block, port) tuple, whether the input is
    a 'source.out' string, a {'from': 'source.out', 'to': 'encoder.in'} dict,
    or a (block, port) tuple."""
    if isinstance(connection, dict):
        return _split_port_ref(connection["from"]), _split_port_ref(connection["to"])
    if isinstance(connection, (tuple, list)) and len(connection) == 2:
        return _split_port_ref(str(connection[0])), _split_port_ref(str(connection[1]))
    return _split_port_ref(str(connection))


def _validate_wiring(
    experiment: dict[str, Any], connections: list[dict[str, str]]
) -> dict[str, Any]:
    """Compare student wiring against required_connections and return the
    deterministic wiring_status dict the state expects."""
    # Normalize required connections (from JSON strings like 'source.out')
    # and student connections (from the wiring kit) to (block, port) tuples so
    # the comparison is apples-to-apples regardless of input format.
    required = {
        (_split_port_ref(conn["from"]), _split_port_ref(conn["to"]))
        for conn in experiment.get("required_connections", [])
    }
    present_pairs = {_normalize_connection(c) for c in connections}
    missing = sorted(required - present_pairs)
    unexpected = sorted(present_pairs - required)

    return {
        "required": [{"from": _format_ref(a), "to": _format_ref(b)} for a, b in sorted(required)],
        "present": [{"from": _format_ref(c["from"]), "to": _format_ref(c["to"])} for c in connections],
        "missing": [{"from": _format_ref(a), "to": _format_ref(b)} for a, b in missing],
        "unexpected": [{"from": _format_ref(a), "to": _format_ref(b)} for a, b in unexpected],
        "valid": not missing and not unexpected,
    }


def _build_oracle_result(experiment: dict[str, Any], blocks: dict[str, dict]) -> dict[str, Any]:
    """Run the experiment's oracle function on the student's block inputs,
    returning the JSON-friendly OracleResult."""
    oracle = experiment.get("oracle") or {}
    function_name = oracle.get("function", "encode")
    args_map = oracle.get("args", {})

    # Normalize argument maps: "encoder.scheme" -> value from blocks
    normalized_args: dict[str, Any] = {}
    for arg_name, path in args_map.items():
        normalized_args[arg_name] = _resolve_block(blocks, path.split(".")[0], path)

    # Today the JSON binds 'oracle.function' = 'encode', which is the
    # Python callable ENCODERS['encode'] (the only oracle the engine supports
    # today). A future experiment (DSSS, interleaving, ...) registers a
    # second pure-NumPy oracle in the same registry; nothing else changes.
    if function_name != "encode":
        raise ValueError(
            "oracle function " + repr(function_name) + " is not yet supported (available: encode)"
        )

    # 'bits' from the JSON is a string like '10110'; normalize to a list
    # of ints before hitting the oracle so the state channel always carries
    # list[int] regardless of the chosen encoder.
    bits = [int(b) for b in str(normalized_args["bits"])]
    # Dispatch to the encoder chosen by the student's scheme (the ENGINE's
    # registry key maps 1:1 to app.oracle.encoders.ENCODERS).
    waveform = ENCODERS[normalized_args["scheme"]](bits)

    samples_per_bit = 2  # encoder default
    return {
        "kind": "line_coding",
        "scheme": normalized_args["scheme"],
        "bits": bits,
        "samples_per_bit": samples_per_bit,
        "levels": [float(v) for v in waveform],
        "dc_component": float(sum(waveform) / len(waveform)),
        "transitions": sum(
            1 for i in range(1, len(waveform)) if waveform[i] != waveform[i - 1]
        ),
    }


def build_circuit_state(
    experiment_id: str, block_inputs: dict[str, dict], connections: list[dict[str, str]]
) -> dict[str, Any]:
    """Public API: given the experiment id, the student's block inputs, and the
    wiring, produce the circuit state section (the ENGINE's only output).

    Returns plain JSON-serializable dicts. Everything downstream (blocks,
    wiring_status, oracle) is pure Python — no numpy arrays leak out.
    """
    experiment = _load_experiment(experiment_id)
    blocks = _parse_block_inputs(experiment["blocks"], block_inputs)
    wiring_status = _validate_wiring(experiment, connections)
    oracle = _build_oracle_result(experiment, blocks)

    # BlockInstance records for the state (drop static schema like position)
    circuit_blocks = []
    for raw in experiment["blocks"]:
        block_inputs = blocks.get(raw["id"], {})
        circuit_blocks.append(
            {
                "id": raw["id"],
                "inputs": [
                    {**inp, "value": block_inputs.get(inp["id"])}
                    for inp in raw.get("inputs", [])
                ],
            }
        )

    return {
        "experiment_id": experiment_id,
        "blocks": circuit_blocks,
        "connections": [{"from": c["from"], "to": c["to"]} for c in connections],
        "wiring_status": wiring_status,
        "dirty": False,  # reset after the engine builds state; the router
        "oracle": oracle,  # re-runs when dirty (set by nodes, not this engine)
        "diagnosis": None,  # set by diagnosis.py, never by the engine
    }
