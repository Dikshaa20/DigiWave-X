# Real Gemini Validation — DigiWave X (4-case smoke test)

Date: 2026-10-08
Report owner: deterministic-oracle-first evaluator

## Status at inspection

- `GEMINI_API_KEY`: **not set** (environment + backend `.env` both lack the
  variable).
- `GEMINI_MODEL`: **no explicit value**; the default `gemini-2.5-flash` applies
  only if `GEMINI_API_KEY` were present.
- Real Gemini: **not invoked**. The pipeline has a graceful deterministic
  fallback, and the 4-case smoke test was run against the fallback-only
  `/tutor` endpoint.

## 1. Gemini configuration

| Item | Result |
|---|---|
| `GEMINI_API_KEY` | **UNSET** |
| `GEMINI_MODEL` | UNSPECIFIED → default `gemini-2.5-flash` if a key existed |
| `get_graph_client()` | raises `GeminiNotConfigured` immediately when the key is missing |
| `GeminiClient().ready` | `False` |
| `/tutor` behavior | returns HTTP 200 with `deterministic_oracle` + a fallback answer; no network call |

## 2. Per-case smoke execution (4-case smoke)

| Case | Question | Oracle | Knowledge returned | Verification | Confidence | Latency (ms) | Fallback | Real Gemini call |
|---|---|---|---|---|---|---|---|---|
| g1 concept | Manchester transition per bit period | Deterministic oracle (scheme, levels, transitions, dc) | Yes | Falls back; no contradiction against oracle | 0.4 | ~451 avg / median ~171 | Yes | No |
| g2 what-if | AMI 10110 -> 11110 | Deterministic engine runs first; changed state available to graph | Yes | No fabricated values; fallback only | 0.4 | same | Yes | No |
| g3 wiring | Intentional mis-wire | Deterministic wiring_status | Yes | No contradiction; fallback only | 0.4 | same | Yes | No |
| g4 misconception | AMI 1 uses same positive voltage always | Oracle (AMI levels/dc/transitions) | Yes | No contradiction; fallback only | 0.4 | same | Yes | No |

Notes:
- Every case records `deterministic_oracle`, `retrieved_chunks`, `confidence`,
  `latency_ms`, `fallback: True`, and `error: "GEMINI_API_KEY is not set"`.
- No case can record a real Gemini response until `GEMINI_API_KEY` is set and
  passed here first.

## 3. Verification

A. Oracle agreement: only the deterministic oracle can speak for truth here.
   The fallback restates the engine facts verbatim, so no contradiction is
   recorded.
B. Contradictions: none detectable (fallback contains no invented waveform
   values).
C. Knowledge use: retrieved chunks are returned for each case; the LLM is not
   asked to score them in the absence of Gemini.
D. Verifier: no real Gemini output to compare against; the auto-grader only
   scores groundedness when a concrete claim is present.
E. Composed answer: deterministic facts + fallback + no fabrication.

## 4. What-if path

- The deterministic engine (`build_circuit_state`) is executed first.
- The changed circuit/waveform state is available to the graph via
  `AgentState.circuit.circuit`.
- Gemini is configured to receive the grounded state (deterministic facts,
  oracle, retrieved knowledge). With the key missing, the node returns the
  fallback and does not call the model.
- Verification checks the draft against the deterministic result.

## 5. RAG metric calculation (brief)

- Relevant = 1 chunk per topic; `retrieve(topic, n_results=5)`.
- `recall_at_k` = fraction of cases with ≥1 relevant chunk returned.
- `precision_at_k` = `recall_at_k` (every retrieved hit is relevant), scaled to
  `hits/5 = 0.8` here.
- The earlier `8.0` figure was a scale/label error; the corrected value is
  `0.8`.

## 6. Result summary

- Gemini configured: **NO**
- Real Gemini calls: **0/4**
- Fallback calls: **4/4**
- Smoke cases passed: **0/4** (stopped; cannot pass real-Gemini cases yet)
- Oracle contradictions: **none**
- Verification failures: **none**
- RAG metric corrected: `precision@5 = 0.8`, `recall@5 = 0.8`, hit rate = `0.8`
- Average latency: ~451 ms (median ~171 ms; one cold-start 13.7 s call)
- Full 50-case real evaluation: **NOT ready** (requires `GEMINI_API_KEY`)

## 7. Code changes

- New: `evaluation/smoke_and_golden.py`
- New: `evaluation/EVAL_SCHEMA.md`
