# Evaluation report schema (`evaluation/results.json`)

This document describes the structure of `evaluation/results.json` produced by
`evaluation/eval_runner.py` and `evaluation/smoke_and_golden.py`.  It is the
single source of truth for what a consumer of the evaluation data can rely on.

---

## 1. Top-level keys

```json
{
  "meta": { ... },
  "rag": { ... },
  "tutor": { ... },
  "per_case": [ ... ],
  "human_only_metrics": { ... }
}
```

| Key | Type | Meaning |
|---|---|---|
| `meta` | object | Global run configuration + counts. Never fabricates LLM scores here. |
| `rag` | object | Retrieval-hit metrics. Deterministic â€” does not depend on Gemini. |
| `tutor` | object | Aggregate tutor metrics. `correctness` / `groundedness` are **automatic** (derived by comparing the answer text to the deterministic oracle â€” they are not LLM-judged). |
| `per_case` | array | One object per dataset case, with `http_status`, `answer`, `confidence`, `latency_ms`, `deterministic_oracle`, `retrieved_chunks`. |
| `human_only_metrics` | object | Metrics that **require human annotation**. Always present; never auto-derived. |

---

## 2. `meta`

```json
"meta": {
  "gemini_api_key_configured": false,
  "gemini_real_calls": 50,
  "deterministic_fallback_only": 50,
  "verification_mode": "automatic",
  "dataset_size": 50,
  "category_counts": { "concept": 15, "wiring": 10, "waveform": 10, "misconception": 10, "what-if": 5 }
}
```

| Field | Type | Meaning |
|---|---|---|
| `gemini_api_key_configured` | boolean | `true` only if `GEMINI_API_KEY` is set in the environment. **Never auto-set.** |
| `gemini_real_calls` | integer | Number of HTTP 200 responses whose body contains a real LLM answer. In a no-key run this is the count of fallback responses (i.e. it equals `deterministic_fallback_only`). |
| `deterministic_fallback_only` | integer | Number of HTTP 200 responses whose answer is the deterministic fallback (the LLM was never loaded). |
| `verification_mode` | string | `"automatic"` (auto-scoring against deterministic oracle) or `"human"` (requires human annotation). |
| `dataset_size` | integer | Total cases in the run. |
| `category_counts` | object | Per-category case counts. |

**Important:** `gemini_real_calls` is a misnomer for a no-Gemini run â€” it counts fallback HTTP 200 bodies, not real Gemini calls. When `GEMINI_API_KEY` is configured, the runner is extended to emit `meta.gemini_calls` separately so a key being present never invents a score.

---
```json
"rag": {
  "k": 5,
  "precision_at_k": 0.8,
  "recall_at_k": 0.8,
  "retrieval_hit_rate": 0.8
}
```

Deterministic.  `precision_at_k` and `recall_at_k` are computed by
`KnowledgeBase.retrieve(topic, n_results=5)` per case.  These metrics do **not**
depend on Gemini and are valid in a no-key run.

A **hit** is a case whose retrieve() returned at least one chunk; a case is
missed when 0 chunks return. With `n_results=5`, recall_at_k = hits /
total_cases and precision_at_k = hits / 5. retrieval_hit_rate equals
recall_at_k.

## 4. `tutor`

```json
"tutor": {
  "total_answered": 50,
  "correctness": 1.0,
  "groundedness": 1.0,
  "contradiction_rate": 0.0,
  "average_confidence": 0.4,
  "average_latency_ms": 451.64,
  "human_evaluated": {
    "specificity": "requires human annotation",
    "relevance": "requires human annotation"
  }
}
```

| Field | Meaning |
|---|---|
| `total_answered` | Number of HTTP 200 cases. |
| `correctness` | Auto-scored: fraction of answered cases whose answer has **no contradictions** against the deterministic oracle. **Not LLM-judged.** |
| `groundedness` | Auto-scored (same basis as `correctness`). |
| `contradiction_rate` | Fraction of answered cases flagged as contradictions. |
| `average_confidence` | Mean of `confidence` (0.4 for the fallback). |
| `average_latency_ms` | Mean HTTP latency of answered cases. |
| `human_evaluated` | Always `"requires human annotation"` â€” human judgment on whether the answer actually deepened understanding / addressed the question. **Never auto-computed.** |

---

## 5. `per_case`

Each element:

```json
{
  "rank": 1,
  "id": "c1",
  "category": "concept",
  "grounded": "no_contradictions",
  "grounding_discrepancies": [],
  "grounding_type": "automatic",
  "answer": "I couldn't load the tutor model right now ...",
  "confidence": 0.4,
  "latency_ms": 33.95,
  "http_status": 200,
  "deterministic_oracle": { "kind": "line_coding", "scheme": "ami", "levels": [...], "transitions": 4, "dc_component": 0.2 },
  "retrieved_chunks": 5
}
```

| Field | Meaning |
|---|---|
| `answer` | The `/tutor` response body. In a no-key run this is the deterministic fallback. |
| `grounded` | `"no_contradictions"` (auto), `"contradiction"`, or `"n/a"`. |
| `grounding_type` | `"automatic"` (auto-scoring) or `"human"` (not yet). |
| `deterministic_oracle` | The engine's concrete result. Used by the auto-groundedness check. |
| `retrieved_chunks` | Number of chunks returned by `KnowledgeBase.retrieve(topic, n_results=5)`. |

---

## 6. `human_only_metrics`

```json
{
  "specificity": "require human annotation (did the answer deepen the student's understanding?)",
  "relevance": "require human annotation (does the answer address the actual question?)"
}
```

Per the honest-fallback boundary, these are **always** marked as requiring human
annotation.  They are never projected onto the LLM.

---

## 7. What "no-Gemini" means for the numbers

In a no-Gemini run:

* `meta.gemini_api_key_configured = false`
* `meta.gemini_real_calls == meta.deterministic_fallback_only`
* `tutor.correctness` / `tutor.groundedness` are **automatic** (deterministic-oracle-based), **not** LLM-grounded.
* `human_only_metrics.specificity` / `.relevance` are `"requires human annotation"`.
* `RAG.*`, `verification_mode: "automatic"` are valid and deterministic.

**Bottom line:** the only numbers you are licensed to report as "real" are the
deterministic ones (status codes, oracle facts, retrieval hits, latency).  Any
LLM correctness / groundedness / relevance / specificity number is explicitly
**not evaluated** because no real Gemini call was made.
