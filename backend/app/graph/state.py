"""
app/graph/state.py — canonical shared state for the DigiWave X agent graph.

The one typed state object all 10 nodes pass through (5 agent roles:
Orchestrator, Diagnosis, Knowledge/RAG, Tutor, Critic).

DESIGN RULES
------------
1. ONE state object travels through the graph. Nodes never talk to each
   other except by reading/writing this state.
2. Seven separated domains (student / experiment / circuit / knowledge /
   tutor / verification / evaluation) + three request-scoped scalars + one
   output contract. Nothing else belongs here.
3. Every fact has exactly ONE canonical home. Other nodes read it; nobody
   copies it (deliberate exceptions are noted in the docstrings).
4. Sections use total=False TypedDicts so nodes return PARTIAL updates;
   the Annotated merge reducer folds each section across nodes.
5. The graph is strictly sequential (one node at a time, conditional edges
   never branch in parallel), so last-writer-wins inside a section is safe.
   If parallel branches are ever added, promote the contended fields
   (retry_count, confidence) to their own channels with explicit reducers.
6. NumPy never enters state: the experiment engine converts oracle arrays
   to plain lists at the boundary, keeping channels JSON-serializable for
   checkpointing and eval-log serialization.
7. Reuse over reinvention: scheme strings are pinned to the keys of
   app.oracle.encoders.ENCODERS; waveform fields follow the {-1,0,+1}
   samples_per_bit convention documented in encoders.py; block/connection
   shapes mirror app/experiments/line_coding.json; misconception ids are
   FKs to the JSON's misconceptions[].id.

Reducers
--------
Section reducers fold partial updates like {"tutor": {"draft_answer": "..."}}
into the existing section so sibling keys survive. Only one node writes a
given section per super-step, so that pattern is race-free here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# Closed vocabularies — pinned to existing DigiWave code, not invented here
# ---------------------------------------------------------------------------

LineCodeScheme = Literal[
    "nrz", "nrz_unipolar", "nrzm",
    "rz_unipolar", "bipolar_rz", "manchester", "ami",
]
"""Exactly the keys of app.oracle.encoders.ENCODERS: any scheme string in
state dispatches straight to encode()/decode() with no mapping layer."""

Intent = Literal["diagnosis", "concept", "what_if", "hint", "quiz", "chitchat"]
"""Orchestrator's classification of the query — one per turn."""

ExperimentStage = Literal["build", "wire", "run", "observe", "interpret"]
"""Lifecycle position tracked by the orchestrator; drives the conditional
edges and what the UI treats as 'next'."""

HintLevel = Literal[0, 1, 2, 3]
"""Hint Escalation ladder: 0 nudge → 1 guiding question → 2 partial worked
example → 3 full answer. Escalates at most one level per retry."""

MAX_RETRIES = 2              # architecture constant: verify→generate loop cap
CONFIDENCE_THRESHOLD = 0.70  # below this, the verify_gate loops to generate


# ---------------------------------------------------------------------------
# 1. STUDENT — profile store snapshot in, queued updates out
# ---------------------------------------------------------------------------

class MasteryRecord(TypedDict, total=False):
    """One row of the persistent mastery table, snapshotted into state at graph
    entry."""
    topic: str
    """Join key to the profile store and to theory tags, e.g. 'manchester',
    'ami_polarity'."""
    level: float
    """0.0–1.0 mastery estimate; hint_escalate reads it for the starting
    hint level."""
    attempts: int
    """Separates 'never seen' from 'seen and failed'."""
    correct: int
    """Verified-correct subset of attempts (verification passed)."""
    last_seen: str | None
    """ISO timestamp; lets hint policy decay stale mastery."""
    misconception_ids: list[str]
    """misconceptions[].id values this student has exhibited — diagnosis and
    hint escalation bias toward known weaknesses."""


class MasteryUpdate(TypedDict, total=False):
    """Mastery event queued during the turn. Nothing writes the profile store
    mid-graph; compose persists these and clears the list (one write,
    auditable)."""
    topic: str
    event: Literal["diagnosis", "hint_escalated", "what_if", "verification", "quiz"]
    delta: float
    reason: str


class StudentState(TypedDict, total=False):
    student_id: str
    mastery: dict[str, MasteryRecord]
    mastery_updates: list[MasteryUpdate]
    """Read-heavy domain: router (personalization), hint_escalate (level
    choice), generate (depth/tone). Writes happen only via mastery_updates."""


# ---------------------------------------------------------------------------
# 2. EXPERIMENT — identity + lifecycle (schema stays in the JSON)
# ---------------------------------------------------------------------------

class ExperimentState(TypedDict, total=False):
    """Experiment identity and lifecycle. Static schema stays in
    app/experiments/<id>.json; state carries instance + lifecycle data only."""
    experiment_id: str
    title: str
    objective: str
    """Grounds the generator's framing of the answer."""
    theory_tags: list[str]
    """Seed terms the retriever adds to its query ('line coding',
    'Manchester', 'DC component')."""
    stage: ExperimentStage
    stage_history: list[ExperimentStage]
    """Append-only trail, not a copy of `stage`: the router uses it to detect
    regressions (student rewires mid-'observe' → back to 'wire'); also eval
    metadata — these are history, not duplicates."""


# ---------------------------------------------------------------------------
# 3. CIRCUIT — the virtual kit: wiring + deterministic oracle + diagnosis
# ---------------------------------------------------------------------------

class BlockInput(TypedDict, total=False):
    """A block input with its student-set value. Mirrors inputs[] entries in
    the experiment JSON (id/type), plus the live value."""
    id: str
    type: Literal["bitstring", "select", "number"]
    value: str | float
    """The student-set value, engine-validated against the JSON's
    validation regex / options / min-step."""


class BlockInstance(TypedDict, total=False):
    id: str
    inputs: list[BlockInput]
    """Static schema (label, ports, position) stays in the experiment JSON —
    state carries instance data only."""


PortRef = TypedDict("PortRef", {"block": str, "port": str})
Connection = TypedDict("Connection", {"from": PortRef, "to": PortRef})
"""Functional syntax because 'from' is a keyword; keys mirror
required_connections[] in the JSON exactly."""


class WiringStatus(TypedDict, total=False):
    """Engine's comparison of student wiring vs required_connections —
    the deterministic half of the diagnosis."""
    required: list[Connection]
    present: list[Connection]
    missing: list[Connection]
    """Required connections absent from the kit — hard faults."""
    unexpected: list[Connection]
    """Wired but not required (e.g. source straight to scope, skipping the
    encoder) — the classic wiring misconception."""
    valid: bool


class OracleResult(TypedDict, total=False):
    """THE deterministic ground truth for the current turn, computed once by
    the engine from circuit state via app.oracle.* — never by an LLM.
    Generator quotes these values; Verifier diffs claims against them.
    What-if recomputes a HYPOTHETICAL variant separately
    (tutor.what_if.oracle_result) so the two never collide."""
    kind: Literal["line_coding", "dsss"]
    scheme: LineCodeScheme
    bits: list[int]
    samples_per_bit: int
    """Carried because decode()/check_ami_violation() need it; defaults to
    encoders.SAMPLES_PER_BIT."""
    levels: list[float]
    """encode() output in the {-1, 0, +1} convention. Deliberately UNSCALED:
    scope voltage is applied at the API layer so Verifier comparisons stay
    exact regardless of the voltage the student typed."""
    dc_component: float
    """mean(levels) — the deterministic statistic behind 'zero DC'
    (Manchester) vs 'nonzero DC' (NRZ/unipolar) claims."""
    transitions: int
    """Level-change count — the statistic behind self-clocking/bandwidth
    claims."""
    ami_violations: list[int]
    """Bit indices from decoders.check_ami_violation(); [] when N/A or clean.
    Direct Diagnosis evidence, inspectable — never an opaque pass/fail."""
    stages: dict[str, list[float]]
    """Named intermediate signals for multi-stage oracles, mirroring
    run_dsss_chain() per-stage outputs (spread_signal, tx_signal, ...),
    so the DSSS experiment slots in without a schema change."""


class Fault(TypedDict, total=False):
    kind: Literal["wiring", "parameter", "waveform", "misconception"]
    block_id: str | None
    misconception_id: str | None
    """FK to misconceptions[].id in the experiment JSON."""
    name: str
    description: str
    """Copied from the JSON only for MATCHED misconceptions, so faults are
    self-contained for compose/eval — the catalog itself stays in the file."""
    evidence: str
    """The deterministic proof, e.g. 'check_ami_violation → bit 2' or
    'missing required connection encoder.in'."""
    severity: Literal["error", "warning"]


class DiagnosisResult(TypedDict, total=False):
    """Diagnosis Agent's structured output. Engine/oracle facts only — no LLM
    prose; phrasing happens later in the Generator."""
    faults: list[Fault]
    summary: str
    diagnosed: bool
    """False when the circuit is healthy or not yet assessable (stage < run)
    — distinct from 'assessed, no faults found'."""


class CircuitState(TypedDict, total=False):
    blocks: list[BlockInstance]
    connections: list[Connection]
    wiring_status: WiringStatus | None
    dirty: bool
    """Set when the kit is edited after the last oracle run; the router uses
    it to force re-diagnosis instead of trusting stale results."""
    oracle: OracleResult | None
    diagnosis: DiagnosisResult | None


# ---------------------------------------------------------------------------
# 4. KNOWLEDGE — retriever + reranker outputs
# ---------------------------------------------------------------------------

class RetrievedChunk(TypedDict, total=False):
    chunk_id: str
    text: str
    distance: float | None
    """ChromaDB embedding distance from the retriever (lower = closer)."""
    rerank_score: float | None
    """Cross-encoder score from the reranker (higher = more relevant)."""
    metadata: dict[str, Any]
    """Ingestion metadata: source doc, heading, experiment_tags."""


class KnowledgeState(TypedDict, total=False):
    """RAG pipeline. Both lists are kept deliberately: compose/verify read
    reranked, the eval writer reads BOTH orderings to measure what
    reranking actually changed."""
    query_used: str
    """The query actually sent to ChromaDB — diagnosis-aware, so it may
    differ from the raw query. Recorded, not copied: the raw query stays
    top-level."""
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    enough_evidence: bool
    """Reranker's verdict that top evidence clears a relevance floor. False
    tells the generator to lean on oracle facts and say so — instead of
    inventing citations."""


# ---------------------------------------------------------------------------
# 5. TUTOR — generator + hint escalation + what-if
# ---------------------------------------------------------------------------

class WhatIfResult(TypedDict, total=False):
    hypothesis: str
    """The counterfactual, e.g. 'same bits, scheme=manchester'."""
    prediction: str
    """The student's prediction, verbatim — eval needs the raw text."""
    prediction_levels: list[float] | None
    """The prediction parsed into the {-1,0,+1} convention when possible —
    enables an exact array diff, not just prose comparison."""
    oracle_result: OracleResult
    """Deterministic recomputation for the HYPOTHETICAL circuit — oracle
    computed, never LLM-guessed (core design principle)."""
    match: bool
    explanation: str


class TutorState(TypedDict, total=False):
    """All three Tutor roles write here."""
    draft_answer: str | None
    """Generator's grounded draft. Facts are injected verbatim from
    circuit.oracle / circuit.diagnosis / knowledge.reranked; the LLM writes
    prose around them and asserts no numbers of its own."""
    citations: list[str]
    """chunk_ids actually used — compose copies them into final_response."""
    facts_used: list[str]
    """The deterministic facts injected into the draft — the Verifier's
    checklist: each must appear in the draft and match the oracle."""
    hint_level: HintLevel
    hint_text: str | None
    what_if: WhatIfResult | None


# ---------------------------------------------------------------------------
# 6. VERIFICATION — critic + the retry loop's working memory
# ---------------------------------------------------------------------------

class Check(TypedDict, total=False):
    name: str
    """e.g. 'scheme_levels', 'dc_component', 'transition_count', 'citation'."""
    claim: str
    expected: str
    """Ground truth, stringified from circuit.oracle /
    tutor.what_if.oracle_result."""
    actual: str
    source: Literal["oracle", "what_if", "reranked_chunks", "diagnosis", "hint_policy"]
    passed: bool


class VerificationResult(TypedDict, total=False):
    checks: list[Check]
    passed: bool
    discrepancies: list[str]
    """Human-readable failures — fed back verbatim into the Generator on
    retry. This is the retry mechanism's working memory."""
    oracle_agreement: float
    """Fraction of oracle-diffable claims that match — score_confidence
    input."""
    evidence_support: float
    """Are cited chunks present in reranked and topically relevant —
    score_confidence input."""


class VerificationState(TypedDict, total=False):
    verification: VerificationResult | None
    confidence: float | None
    """0..1, set by score_confidence; verify_gate compares against
    CONFIDENCE_THRESHOLD."""
    retry_count: int
    """Increments only when verify_gate loops back to generate; hard-capped
    at MAX_RETRIES = 2. Read-then-write by the gate, its only writer."""


# ---------------------------------------------------------------------------
# 7. EVALUATION — eval_log.db metadata (pointers, not copies)
# ---------------------------------------------------------------------------

class EvaluationState(TypedDict, total=False):
    """Holds ONLY what no other section owns. The interactions-table columns
    (query, retrieved chunks, diagnosis, hint level, prediction, confidence,
    label) are assembled at compose time FROM their canonical homes — copying
    them here would create two sources of truth."""
    turn_id: str
    session_id: str
    started_at: str
    node_timings: dict[str, float]
    llm_meta: dict[str, Any]
    """Provider, model, token usage — captured where LLM calls happen."""
    label: str | None
    """Human/verification label; NULL until offline labeling fills it."""
    label_status: Literal["unlabeled", "pending", "human_verified", "rejected"]
    logged: bool
    """compose sets True after writing the row — re-runs stay idempotent."""


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class FinalResponse(TypedDict, total=False):
    """Composer's structured JSON output — the API boundary object. It is
    intentionally denormalized: a snapshot for the frontend, which is
    serialization at the edge, not duplication inside the graph."""
    turn_id: str
    answer: str
    hint_level: HintLevel
    hint_text: str | None
    diagnosis_summary: str | None
    faults: list[Fault]
    citations: list[str]
    what_if: WhatIfResult | None
    confidence: float | None
    stage: ExperimentStage
    next_action: str | None
    enough_evidence: bool


def _merge_section(left: dict | None, right: dict | None) -> dict:
    """Shallow-merge so nodes return partial updates, e.g.
    {"tutor": {"draft_answer": "..."}} — sibling keys survive."""
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict):
    """The one state object that travels through all 10 nodes."""
    # -- request-scoped scalars: read by nearly every node, owned by no domain
    query: str
    """The student's message, verbatim — retrieval seed, eval column, prompts."""
    intent: Intent | None
    """Router's classification of THIS query. Turn-scoped, unlike mastery."""
    next_action: str | None
    """Router's plan (node or UI step). Advisory: the graph's conditional
    edges are hard-coded and stay authoritative over it."""

    # -- the seven domains, each foldable via partial updates
    student: Annotated[StudentState, _merge_section]
    experiment: Annotated[ExperimentState, _merge_section]
    circuit: Annotated[CircuitState, _merge_section]
    knowledge: Annotated[KnowledgeState, _merge_section]
    tutor: Annotated[TutorState, _merge_section]
    verification: Annotated[VerificationState, _merge_section]
    evaluation: Annotated[EvaluationState, _merge_section]

    # -- output contract (written once by compose; plain replace)
    final_response: FinalResponse | None
