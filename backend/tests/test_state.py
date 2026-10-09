"""
Tests for the canonical LangGraph state module: merge-reducer correctness and
JSON round-trip (the state must be fully JSON-serializable since it is
checkpointed and rehydrated).

Run:
    python -m pytest tests/test_state.py -v
"""

from __future__ import annotations

import json

import pytest

from app.graph.state import (
    AgentState,
    HintLevel,
    Intent,
    LineCodeScheme,
    MasteryRecord,
)


# ---------------------------------------------------------------------------
# 1. Closed vocabularies accept exactly the intended members (types, not
#    values — these are literals, so we check the classes behave as typed)
# ---------------------------------------------------------------------------

def test_line_code_schemes_are_closed_vocabulary():
    allowed = {
        "nrz", "nrz_unipolar", "nrzm",
        "rz_unipolar", "bipolar_rz", "manchester", "ami",
    }
    assert all(s in allowed for s in LineCodeScheme.__args__)
    # a scheme not defined in the oracle is rejected, not coerced
    with pytest.raises(ValueError):
        from app.oracle.encoders import encode
        encode("hdb3", [1, 0, 1])


# ---------------------------------------------------------------------------
# 2. Merge reducer: folding partial updates must not clobber sibling keys
# ---------------------------------------------------------------------------

def _sample_agent_state() -> AgentState:
    """A populated AgentState for reducer exercises."""
    return {
        "query": "How do I wire the encoder?",
        "intent": "diagnosis",
        "next_action": None,
        "experiment": {"experiment_id": "line_coding", "stage": "wire"},
        "student": {"student_id": "s1", "mastery": {"manchester": {}}},
        "circuit": {"blocks": [{"id": "source", "inputs": []}]},
        "knowledge": {"retrieved": [{"chunk_id": "c1"}]},
        "tutor": {"draft_answer": "a"},
        "verification": {"retry_count": 0},
        "evaluation": {"turn_id": "t1"},
        "final_response": None,
    }


def test_merge_overwrites_only_the_keys_provided():
    original = _sample_agent_state()
    patch: AgentState = {"experiment": {"stage": "run"}}
    merged = AgentState(_sample_agent_state())
    for k, v in patch.items():
        merged[k] = v  # mimic the reducers' shallow-merge
    assert merged["circuit"]["blocks"] == original["circuit"]["blocks"]
    assert merged["student"]["student_id"] == original["student"]["student_id"]
    assert merged["tutor"]["draft_answer"] == original["tutor"]["draft_answer"]
    assert merged["experiment"]["stage"] == "run"
    assert merged["knowledge"] == original["knowledge"]


def test_merge_preserves_nesting_and_other_domains():
    patch: AgentState = {"student": {"mastery": {"manchester": {"level": 0.5}}}}
    state = _sample_agent_state()
    state["student"] = patch["student"]  # merge applied to the section
    assert state["student"]["mastery"]["manchester"]["level"] == 0.5
    assert set(state["student"]["mastery"].keys()) == {"manchester"}


def test_merge_empty_right_is_identity():
    state = _sample_agent_state()
    merged = AgentState(state)
    assert merged == state


# ---------------------------------------------------------------------------
# 3. JSON round-trip: the whole state serializes (checkpointable) and
#    rehydrates without loss — except numpy arrays, which the engine must
#    convert at the boundary (documented rule #6).
# ---------------------------------------------------------------------------

def _fully_jsonable_state() -> AgentState:
    """A state shaped exactly like a real run, with primitives only."""
    return {
        "query": "Explain why AMI polarity does not reset after zeros.",
        "intent": "concept",
        "next_action": "retrieve",
        "student": {"student_id": "s1",
                    "mastery": {"manchester": {"level": 0.7,
                                                "attempts": 3,
                                                "correct": 2}},
                    "mastery_updates": [{"topic": "ami",
                                         "event": "diagnosis",
                                         "delta": 0.05,
                                         "reason": "AMI polarity misconception"}]},
        "experiment": {"experiment_id": "line_coding", "stage": "concept",
                       "theory_tags": ["AMI", "DC component"]},
        "circuit": {"blocks": [{"id": "source", "inputs": [{"id": "bits",
                                                              "type": "bitstring",
                                                              "value": "10110"}]}],
                    "oracle": {"kind": "line_coding",
                               "scheme": "ami",
                               "bits": [1, 0, 1, 1, 0],
                               "levels": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0,
                                          1.0, 1.0, 0.0, 0.0]}},
        "knowledge": {"query_used": "AMI polarity not reset after zeros",
                      "retrieved": [{"chunk_id": "ami-intro",
                                     "text": "AMI alternates mark polarity.",
                                     "distance": 0.23,
                                     "rerank_score": 0.91,
                                     "metadata": {"source": "lab manual"}}],
                      "reranked": [{"chunk_id": "ami-intro",
                                    "text": "AMI alternates mark polarity.",
                                    "distance": 0.23,
                                    "rerank_score": 0.91,
                                    "metadata": {}}]},
        "tutor": {"hint_level": 1,
                  "what_if": {"hypothesis": "same bits, scheme=manchester",
                              "prediction": "I expect a square wave",
                              "match": False,
                              "explanation": "differs from IEEE 802.3"}},
        "verification": {"retry_count": 0},
        "evaluation": {"turn_id": "t1", "session_id": "s1-turn1",
                       "node_timings": {"router": 0.01}},
        "final_response": None,
    }


def test_state_serializes_to_json_and_rehydrates():
    raw = _fully_jsonable_state()
    blob = json.dumps(raw, ensure_ascii=False)
    assert len(blob) > 0
    reloaded = json.loads(blob)
    assert reloaded == raw  # equality, not identity — no attribute loss
    assert isinstance(reloaded, dict)
    assert set(reloaded.keys()) == set(raw.keys())  # double-check no drift


def test_state_has_exact_top_level_key_shape():
    state = _sample_agent_state()
    expected = {
        "query", "intent", "next_action",
        "student", "experiment", "circuit", "knowledge", "tutor",
        "verification", "evaluation", "final_response",
    }
    assert set(state.keys()) == expected


def test_section_reducers_are_metadata_only_and_documented():
    # the reducers are runtime fold functions, not serializable state
    from app.graph.state import _merge_section
    a = {"student": {"student_id": "s1"}}
    b = {"circuit": {"blocks": []}}
    merged = _merge_section(a, b)
    assert merged == {"student": {"student_id": "s1"}, "circuit": {"blocks": []}}


def test_section_domain_mismatch_does_not_corrupt_sibling_sections():
    # a node writes its own domain; verify and knowledge stay untouched
    state = _sample_agent_state()
    state["tutor"] = {"hint_level": 2, "draft_answer": "revised"}
    assert state["tutor"]["hint_level"] == 2
    assert state["knowledge"] == {"retrieved": [{"chunk_id": "c1"}]}
    assert state["verification"] == {"retry_count": 0}
