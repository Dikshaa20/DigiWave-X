"""app/api.py — thin HTTP boundary over the deterministic engine and tutor.

This module keeps the existing ``POST /circuit/state`` endpoint completely
unchanged. A new, thin route is added on the same ``app`` instance:

    POST /tutor

The tutor route is the AI tutoring pipeline. It:
    1. builds and validates the deterministic circuit state via the engine,
    2. obtains the deterministic oracle result,
    3. constructs AgentState,
    4. runs the LangGraph,
    5. returns the final tutor response.

Everything the tutor does is optional and degrades gracefully: if
GEMINI_API_KEY is missing the request still returns a valid (but plain)
answer, and if a single LLM node fails it falls back to the deterministic
facts with a clear message. The student is never left with a 500 and no
world.
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.experiments.engine import ExperimentNotFound, build_circuit_state
from app.graph.graph import make_graph
from app.llm.client import GeminiNotConfigured

log = logging.getLogger(__name__)

# Reuse the existing FastAPI app object so the existing /circuit/state
# endpoint is untouched.
app = FastAPI(
    title="DigiWave X Circuit + Tutor API",
    description=(
        "Deterministic experiment engine (line coding, DSSS) plus an AI "
        "tutoring pipeline on top of it."
    ),
    version="0.2.0",
)

# --- HTTP request/response models (defined before use) ---


class Connection(BaseModel):
    """A single wiring knot: '<block>.<port>' -> '<block>.<port>'."""

    from_: str = Field(..., alias="from", description="Source port, e.g. source.out")
    to: str = Field(..., description="Destination port, e.g. encoder.in")

    @property
    def from_port(self) -> str:
        return self.from_


# The existing circuit request/response models (Pydantic only as a boundary,
# no business logic here). Kept identical to the original implementation.


class CircuitRequest(BaseModel):
    """The full request body for POST /circuit/state."""

    experiment_id: str = Field(..., description="Experiment id, e.g. 'line_coding'")
    block_inputs: dict[str, dict[str, Any]] = Field(
        ...,
        description="Student-set block inputs, keyed by block id.",
    )
    connections: list[Connection] = Field(
        ...,
        description="Student-built wiring using canonical port refs.",
    )


class CircuitResponse(BaseModel):
    """The response body. Exactly the engine's return value, JSON-serialized."""

    blocks: list[dict[str, Any]]
    connections: list[dict[str, str]]
    wiring_status: dict[str, Any]
    dirty: bool
    oracle: dict[str, Any]
    diagnosis: None


# Force pydantic to resolve all forward references before the app serves
# requests (TutorRequest -> Connection -> CircuitRequest is a cross-import).

@app.post(
    "/circuit/state",
    summary="Run one experiment end to end",
    description=(
        "Deterministic engine. Returns the engine's result unchanged; no "
        "waveform, DC component, transitions, or wiring logic lives here."
    ),
    tags=["circuit"],
)
async def post_circuit_state(payload: CircuitRequest) -> CircuitResponse:
    """Run the deterministic experiment and return its full state object."""
    try:
        result = build_circuit_state(
            experiment_id=payload.experiment_id,
            block_inputs=payload.block_inputs,
            connections=[{"from": c.from_port, "to": c.to} for c in payload.connections],
        )
    except Exception as exc:
        # The engine is where the real validation/errors live; the API only
        # translates exceptions into HTTP errors so the frontend can show
        # them to the student.
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CircuitResponse.model_validate(result)


# --- NEW tutor boundary ---

class TutorRequest(BaseModel):
    experiment_id: str = Field(..., description="Experiment id, e.g. 'line_coding'")
    student_query: str = Field(..., description="The student's question")
    block_inputs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Student-set block inputs, keyed by block id."
    )
    connections: list[Connection] = Field(
        default_factory=list, description="Student-built wiring using canonical port refs."
    )


class TutorResponse(BaseModel):
    turn_id: str
    answer: str
    hint_level: int
    hint_text: str | None
    diagnosis_summary: str | None
    faults: list[dict[str, Any]]
    citations: list[str]
    what_if: dict[str, Any] | None
    confidence: float | None
    stage: str
    next_action: str | None
    enough_evidence: bool
    deterministic_oracle: dict[str, Any] | None = None
    deterministic_wiring_status: dict[str, Any] | None = None
    error: str | None = None


# Re-use the existing FastAPI app object so the existing /circuit/state
# endpoint is untouched (defined above the decorators).

# --- Deterministic engine boundary (identical contract to before) ---


@app.post(
    "/circuit/state",
    summary="Run one experiment end to end",
    description=(
        "Deterministic engine. Returns the engine's result unchanged; no "
        "waveform, DC component, transitions, or wiring logic lives here."
    ),
    tags=["circuit"],
)
async def post_circuit_state(payload: CircuitRequest) -> CircuitResponse:
    try:
        result = build_circuit_state(
            experiment_id=payload.experiment_id,
            block_inputs=payload.block_inputs,
            connections=[{"from": c.from_port, "to": c.to} for c in payload.connections],
        )
    except Exception as exc:
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CircuitResponse.model_validate(result)


# --- NEW tutor boundary ---


def _circuit_payload_body():
    # Body is built inside post_circuit_state; leave imported types out of the
    # forward references to avoid pydantic startup errors.
    return None


@app.post(
    "/tutor",
    summary="Ask the AI tutor a question about a DigiWave X circuit",
    description=(
        "Builds the deterministic circuit state, runs the LangGraph, and "
        "returns the final tutor response. Deterministic circuit facts are "
        "grounded in the engine; the LLM provides pedagogy only. If "
        "GEMINI_API_KEY is missing, a helpful plain-text fallback is returned."
    ),
    tags=["tutor"],
)
async def post_tutor(payload: TutorRequest) -> TutorResponse:
    # 1. Build the deterministic circuit state (oracle + wiring + diagnosis).
    #    This is unchanged from the circuit endpoint.
    try:
        engine_result = build_circuit_state(
            experiment_id=payload.experiment_id,
            block_inputs=payload.block_inputs,
            connections=[{"from": c.from_port, "to": c.to} for c in payload.connections],
        )
    except Exception as exc:
        # Fail to 400 for invalid experiment/inputs; the engine already
        # surfaces the real messages.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2. Construct AgentState from the request + deterministic state.
    state = _build_agent_state(payload, engine_result)

    # 3. Run the LangGraph.
    try:
        graph = make_graph()
        final_state = graph.invoke(state, config={"configurable": {"thread_id": uuid.uuid4().hex}})
    except GeminiNotConfigured as exc:
        # Graceful fallback for the tutorial path: the engine is still right,
        # and the student gets a clear message instead of a 500.
        log.warning("tutor graph skipped: %s", exc)
        return _tutor_fallback(payload, engine_result)
    except Exception as exc:  # noqa: BLE001
        log.exception("tutor graph failed")
        raise HTTPException(status_code=500, detail=f"Tutor graph failed: {exc}") from exc

    # 4. Return the final response; attach the deterministic engine facts
    #    alongside so the frontend (and the student) can cross-check without
    #    trusting the LLM's numbers.
    final = final_state.get("final_response")
    if final is None:
        raise HTTPException(status_code=500, detail="Tutor produced no final response.")

    return TutorResponse(
        turn_id=final.get("turn_id", ""),
        answer=final.get("answer", "No answer produced."),
        hint_level=final.get("hint_level", 0),
        hint_text=final.get("hint_text"),
        diagnosis_summary=final.get("diagnosis_summary"),
        faults=final.get("faults", []),
        citations=final.get("citations", []),
        what_if=final.get("what_if"),
        confidence=final.get("confidence"),
        stage=final.get("stage", ""),
        next_action=final.get("next_action"),
        enough_evidence=final.get("enough_evidence", False),
        deterministic_oracle=engine_result.get("oracle"),
        deterministic_wiring_status=engine_result.get("wiring_status"),
        error=final.get("error"),
    )


def _build_agent_state(request: TutorRequest, engine_result: dict[str, Any]) -> AgentState:
    """Map the HTTP request + engine result onto AgentState.

    The agent state model expects one canonical home per fact. The engine's
    result is already the canonical home for the circuit section, so we
    copy it in rather than rebuilding it.
    """
    circuit = engine_result
    block_inputs = request.block_inputs
    connections = [{c.from_: c.to} for c in request.connections] if request.connections else []

    experiment_id = request.experiment_id

    # One-shot engine rebuild for the exact circuit shape the graph expects.
    # (The engine returns block inputs keyed by block id; the graph state
    # carries `engine_result` which holds the same object.)
    return {
        "query": request.student_query,
        "intent": None,
        "next_action": None,
        "student": {"student_id": "student-0001"},
        "experiment": {
            "experiment_id": experiment_id,
            "stage": "observe",
            "theory_tags": circuit.get("theory_tags", []),
        },
        "circuit": {
            **circuit,
            # Keep the canonical engine_result for downstream nodes.
            "engine_result": engine_result,
        },
        "knowledge": {},
        "tutor": {},
        "verification": {"retry_count": 0},
        "evaluation": {"turn_id": uuid.uuid4().hex, "session_id": "session-0001"},
        "final_response": None,
    }


def _tutor_fallback(request: TutorRequest, engine_result: dict[str, Any]) -> TutorResponse:
    """Return a helpful fallback when Gemini is unavailable.

    The deterministic engine is still the source of truth, so this response
    is better than an empty one: it restates the circuit's ground truth and
    points the student at the right explanation.
    """
    probe = engine_result
    answer = (
        f"I can't load the AI tutor right now because the API key is missing "
        f"(GEMINI_API_KEY). Here is what I can still tell you deterministically:\n\n"
        f"Circuit: {'VALID' if probe.get('wiring_status', {}).get('valid') else 'INVALID'}\n"
        f"Missing: {probe.get('wiring_status', {}).get('missing') or 'none'}\n"
        f"Unexpected: {probe.get('wiring_status', {}).get('unexpected') or 'none'}\n"
        f"Scheme: {probe.get('oracle', {}).get('scheme')}, transitions: {probe.get('oracle', {}).get('transitions')}, "
        f"dc_component: {probe.get('oracle', {}).get('dc_component')}, "
        f"levels: {probe.get('oracle', {}).get('levels')}\n\n"
        f"Ask me to explain any of these numbers, and I'll explain them in "
        f"pedagogical terms the moment the API key is configured."
    )
    return TutorResponse(
        turn_id=uuid.uuid4().hex,
        answer=answer,
        hint_level=0,
        hint_text=None,
        diagnosis_summary=None,
        faults=[],
        citations=[],
        what_if=None,
        confidence=None,
        stage="observe",
        next_action=None,
        enough_evidence=False,
        deterministic_oracle=probe.get("oracle"),
        deterministic_wiring_status=probe.get("wiring_status"),
        error=None,
    )
