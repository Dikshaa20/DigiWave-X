# DigiWave X — Evaluation & Baseline

Real, measured evaluation for the AI tutoring pipeline. No fabricated numbers.

## What is measured

### Automated (this runner)
- Dataset size + category counts
- RAG: Precision@K, Recall@K, retrieval hit rate
- Tutor: correctness, groundedness (faithfulness), verification pass rate,
  contradiction rate, average confidence, average latency
- Per-case groundedness check (yes/no, reason, automatic)

### Human-only (never fabricated by the scripts)
- Specificity — did the answer deepen the student's understanding?
- Relevance — does the answer address the actual question?

## Boundary for the current key state

`GEMINI_API_KEY` is **not** set in this environment. The scripts therefore:

- record `gemini_api_key_configured: false`
- run the *real* (`/tutor`) pipeline, which returns the deterministic fallback
- produce an honest, real baseline using the deterministic engine instead of
  the LLM
- label every metric that needs human judgement as "requires human annotation"

The API key must never be printed, logged, or committed.

## Run

```bash
# 1. start the backend (uvicorn app.api:app) — its /tutor route serves the
#    deterministic fallback when the key is absent and the real LangGraph when
#    the key is present.

# 2. run the real evaluation
PYTHONPATH=. .venv/Scripts/python.exe eval/run_eval.py

# 3. run the deterministic baseline (identical dataset, real measured system)
PYTHONPATH=. .venv/Scripts/python.exe eval/baseline.py
```

## Outputs
- `eval_results.jsonl` — one JSON line per case (full trace)
- `results.json` — aggregate report
- `baseline_results.json` — baseline aggregate report

## Method notes
- Grounding: the deterministic oracle the engine computed is diffed against the
  answer text for levels / transitions / dc_component / scheme. Contradictions
  are *reported*, never silently fixed.
- The engine (and therefore the oracle) is the source of truth and is unchanged.
- The RAG layer is the same `KnowledgeBase/retrieve` used by the graph;
  `expected_topic` in the dataset seeds the Chroma query.
