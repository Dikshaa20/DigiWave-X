"""
Tests for the deterministic experiment API (app/api.py).

Covers at minimum:

* valid complete circuit
* missing connection
* unexpected connection
* invalid experiment id (HTTP error)
* invalid input (engine contract still enforced)

Run:

    python -m pytest tests/test_api.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared payload templates
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "10110"},
        "encoder": {"scheme": "ami"},
        "scope": {"voltage": 1},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        {"from": "encoder.out", "to": "scope.in"},
    ],
}

MISSING_WIRING_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "10110"},
        "encoder": {"scheme": "ami"},
        "scope": {"voltage": 1},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        # encoder.out -> scope.in is deliberately omitted
    ],
}

UNEXPECTED_WIRING_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "10110"},
        "encoder": {"scheme": "ami"},
        "scope": {"voltage": 1},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        {"from": "source.out", "to": "scope.in"},
    ],
}

BAD_BITS_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "101a0"},
        "encoder": {"scheme": "ami"},
        "scope": {"voltage": 1},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        {"from": "encoder.out", "to": "scope.in"},
    ],
}

BAD_SCHEME_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "10110"},
        "encoder": {"scheme": "bipolar"},
        "scope": {"voltage": 1},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        {"from": "encoder.out", "to": "scope.in"},
    ],
}

BAD_VOLTAGE_PAYLOAD = {
    "experiment_id": "line_coding",
    "block_inputs": {
        "source": {"bits": "10110"},
        "encoder": {"scheme": "ami"},
        "scope": {"voltage": 0.05},
    },
    "connections": [
        {"from": "source.out", "to": "encoder.in"},
        {"from": "encoder.out", "to": "scope.in"},
    ],
}


# ---------------------------------------------------------------------------
# 1. Valid complete circuit
# ---------------------------------------------------------------------------

class TestValidCircuit:
    def test_valid_complete_circuit_returns_200_and_valid_wiring(self):
        response = client.post("/circuit/state", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["wiring_status"]["valid"] is True
        assert body["oracle"]["bits"] == [1, 0, 1, 1, 0]

    def test_valid_oracle_levels_match_engine(self):
        response = client.post("/circuit/state", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["oracle"]["kind"] == "line_coding"
        assert body["oracle"]["scheme"] == "ami"
        assert body["oracle"]["levels"] == [
            1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0
        ]
        assert body["oracle"]["dc_component"] == pytest.approx(0.2)
        assert body["oracle"]["transitions"] == 4


# ---------------------------------------------------------------------------
# 2. Missing connection
# ---------------------------------------------------------------------------

class TestMissingConnection:
    def test_missing_connection_returns_state_and_valid_false(self):
        response = client.post("/circuit/state", json=MISSING_WIRING_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["wiring_status"]["valid"] is False
        assert len(body["wiring_status"]["missing"]) == 1
        assert body["wiring_status"]["missing"][0] == {
            "from": "encoder.out",
            "to": "scope.in",
        }
        assert body["wiring_status"]["unexpected"] == []

    def test_engine_state_is_returned_even_when_wiring_is_invalid(self):
        response = client.post("/circuit/state", json=MISSING_WIRING_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        # The engine still computes the oracle; only wiring_status marks it invalid
        assert body["oracle"]["bits"] == [1, 0, 1, 1, 0]


# ---------------------------------------------------------------------------
# 3. Unexpected connection
# ---------------------------------------------------------------------------

class TestUnexpectedConnection:
    def test_unexpected_connection_returns_state_and_valid_false(self):
        response = client.post("/circuit/state", json=UNEXPECTED_WIRING_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["wiring_status"]["valid"] is False
        assert len(body["wiring_status"]["unexpected"]) == 1
        assert body["wiring_status"]["unexpected"][0] == {
            "from": "source.out",
            "to": "scope.in",
        }

    def test_unexpected_connection_does_not_affect_oracle(self):
        response = client.post("/circuit/state", json=UNEXPECTED_WIRING_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["oracle"]["bits"] == [1, 0, 1, 1, 0]


# ---------------------------------------------------------------------------
# 4. Invalid experiment id
# ---------------------------------------------------------------------------

class TestInvalidExperimentId:
    def test_invalid_experiment_id_returns_404(self):
        bad = dict(VALID_PAYLOAD)
        bad["experiment_id"] = "does_not_exist"
        response = client.post("/circuit/state", json=bad)
        assert response.status_code == 404

    def test_invalid_experiment_id_returns_no_fake_state(self):
        response = client.post("/circuit/state", json={
            "experiment_id": "does_not_exist",
            "block_inputs": {},
            "connections": [],
        })
        assert response.status_code == 404
        assert "detail" in response.json()


# ---------------------------------------------------------------------------
# 5. Invalid input
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_invalid_bits_returns_400(self):
        response = client.post("/circuit/state", json=BAD_BITS_PAYLOAD)
        assert response.status_code == 400

    def test_invalid_scheme_returns_400(self):
        response = client.post("/circuit/state", json=BAD_SCHEME_PAYLOAD)
        assert response.status_code == 400

    def test_invalid_voltage_returns_400(self):
        response = client.post("/circuit/state", json=BAD_VOLTAGE_PAYLOAD)
        assert response.status_code == 400

    def test_invalid_input_returns_no_fake_oracle(self):
        for payload in (BAD_BITS_PAYLOAD, BAD_SCHEME_PAYLOAD, BAD_VOLTAGE_PAYLOAD):
            response = client.post("/circuit/state", json=payload)
            assert response.status_code == 400
            assert "oracle" not in response.json()
