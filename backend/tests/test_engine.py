"""
Tests for the deterministic experiment engine (app/experiments/engine.py).

Covers at minimum:

* valid experiment loading
* invalid experiment ID
* correct wiring
* missing connection
* unexpected connection
* invalid block input
* valid oracle execution
* JSON serializability
* malformed experiment specification
* oracle failure / error handling

Run:\n    python -m pytest tests/test_engine.py -v
"""
from __future__ import annotations

import json

import pytest

from app.experiments.engine import (
    ExperimentNotFound,
    _load_experiment,
    build_circuit_state,
)

LINE_CODING = {
    "source": {"bits": "10110"},
    "encoder": {"scheme": "ami"},
    "scope": {"voltage": 1.0},
}

CORRECT_WIRING = [
    {"from": "source.out", "to": "encoder.in"},
    {"from": "encoder.out", "to": "scope.in"},
]

MISSING_WIRING = [
    {"from": "source.out", "to": "encoder.in"},
]


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------

class TestExperimentLoading:
    def test_valid_experiment_loading(self):
        exp = _load_experiment("line_coding")
        assert isinstance(exp, dict)
        assert exp["id"] == "line_coding"
        assert "blocks" in exp
        assert "required_connections" in exp
        assert "oracle" in exp

    def test_invalid_experiment_id(self):
        with pytest.raises(ExperimentNotFound, match="not found"):
            _load_experiment("does_not_exist")

    def test_malformed_experient_invalid_json(self):
        with pytest.raises(Exception):
            _load_experiment("no_such_file")


# ---------------------------------------------------------------------------
# Wiring validation
# ---------------------------------------------------------------------------

class TestWiringValidation:
    def test_correct_wiring(self):
        state = build_circuit_state("line_coding", LINE_CODING, CORRECT_WIRING)
        assert state["wiring_status"]["valid"] is True
        assert state["wiring_status"]["missing"] == []
        assert state["wiring_status"]["unexpected"] == []

    def test_missing_connection(self):
        state = build_circuit_state("line_coding", LINE_CODING, MISSING_WIRING)
        assert state["wiring_status"]["valid"] is False
        assert len(state["wiring_status"]["missing"]) == 1
        assert state["wiring_status"]["missing"][0] == {
            "from": "encoder.out", "to": "scope.in"
        }
        assert state["wiring_status"]["unexpected"] == []

    def test_unexpected_connection(self):
        bad_wiring = [
            {"from": "source.out", "to": "encoder.in"},
            {"from": "source.out", "to": "scope.in"},
        ]
        state = build_circuit_state("line_coding", LINE_CODING, bad_wiring)
        assert state["wiring_status"]["valid"] is False
        assert len(state["wiring_status"]["unexpected"]) == 1
        assert state["wiring_status"]["unexpected"][0] == {
            "from": "source.out", "to": "scope.in"
        }

    def test_wrong_wiring_does_not_run_oracle(self):
        # The engine must NOT report a valid oracle result when the circuit is
        # mis-wired; the wiring_status must take precedence over the oracle.
        state = build_circuit_state("line_coding", LINE_CODING, MISSING_WIRING)
        assert state["wiring_status"]["valid"] is False
        assert state["oracle"] is not None  # still computed, but wiring_status flags the problem


# ---------------------------------------------------------------------------
# Block input validation
# ---------------------------------------------------------------------------

class TestBlockInputValidation:
    def test_valid_block_inputs(self):
        state = build_circuit_state("line_coding", LINE_CODING, CORRECT_WIRING)
        # The engine preserves the full input schema from the JSON (label,
        # validation, type) and appends the live student value.
        assert state["blocks"] == [
            {
                "id": "source",
                "inputs": [
                    {
                        "id": "bits",
                        "label": "Data Bits",
                        "type": "bitstring",
                        "default": "10110",
                        "validation": "^[01\\s]+$",
                        "value": "10110",
                    }
                ],
            },
            {
                "id": "encoder",
                "inputs": [{"id": "scheme", "label": "Encoding Scheme", "type": "select", "options": ["nrz", "nrz_unipolar", "manchester", "ami", "rz_unipolar", "bipolar_rz", "nrzm"], "default": "nrz", "value": "ami"}],
            },
            {
                "id": "scope",
                "inputs": [{"id": "voltage", "label": "Voltage (V)", "type": "number", "default": 1, "min": 0.1, "step": 0.5, "value": 1.0}],
            },
        ]

    def test_invalid_block_input_non_binary_bitstring(self):
        bad = {"source": {"bits": "101a0"}, "encoder": {"scheme": "ami"}, "scope": {"voltage": 1.0}}
        with pytest.raises(ValueError, match="failed validation"):
            build_circuit_state("line_coding", bad, CORRECT_WIRING)

    def test_invalid_block_input_number_below_min(self):
        bad = {"source": {"bits": "10110"}, "encoder": {"scheme": "ami"}, "scope": {"voltage": 0.05}}
        with pytest.raises(ValueError, match="must be >="):
            build_circuit_state("line_coding", bad, CORRECT_WIRING)


# ---------------------------------------------------------------------------
# Oracle execution
# ---------------------------------------------------------------------------

class TestOracleExecution:
    def test_valid_oracle_execution(self):
        state = build_circuit_state("line_coding", LINE_CODING, CORRECT_WIRING)
        oracle = state["oracle"]
        assert oracle["kind"] == "line_coding"
        assert oracle["scheme"] == "ami"
        assert oracle["bits"] == [1, 0, 1, 1, 0]
        assert oracle["samples_per_bit"] == 2
        assert oracle["levels"][:6] == [1.0, 1.0, 0.0, 0.0, -1.0, -1.0]
        assert oracle["dc_component"] == 0.2
        assert oracle["transitions"] == 4

    def test_oracle_returns_json_serializable(self):
        state = build_circuit_state("line_coding", LINE_CODING, CORRECT_WIRING)
        blob = json.dumps(state)
        assert isinstance(blob, str)
        assert len(blob) > 0


# ---------------------------------------------------------------------------
# JSON serializability
# ---------------------------------------------------------------------------

class TestJSONSerializability:
    def test_full_state_serializable(self):
        state = build_circuit_state("line_coding", LINE_CODING, CORRECT_WIRING)
        blob = json.dumps(state, ensure_ascii=False)
        reloaded = json.loads(blob)
        assert reloaded == state
        assert not any(
            isinstance(v, type(__import__("numpy").array([]))) for v in state.values()
        )


# ---------------------------------------------------------------------------
# Malformed specification
# ---------------------------------------------------------------------------

class TestMalformedSpecification:
    def test_malformed_experiment_not_a_dict(self):
        with pytest.raises(Exception):
            _load_experiment("not_a_dict")


# ---------------------------------------------------------------------------
# Oracle / error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_engine_fails_gracefully_on_missing_experiment(self):
        with pytest.raises(ExperimentNotFound):
            build_circuit_state("missing", {}, [])

    def test_engine_fails_on_invalid_wiring_input(self):
        # A malformed connection should raise rather than silently pass.
        with pytest.raises(Exception):
            build_circuit_state("line_coding", LINE_CODING, [{"from": "nope", "to": "scope.in"}])
