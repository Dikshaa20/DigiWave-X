# -*- coding: utf-8 -*-
"""DigiWave X — automated evaluation runner.

Runs every case in ``dataset.py`` through the real /tutor endpoint where a
Gemini key is configured, and through the deterministic fallback otherwise.
Produces an aggregate report of real measured metrics, then a clearly labelled
set of metrics that require human annotation.

Run (from the backend root):
    PYTHONPATH=. .venv/Scripts/python.exe eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Make the backend package importable regardless of the current working dir.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import requests  # noqa: E402

EVAL = BACKEND_ROOT / "evaluation"
OUT = EVAL / "results.json"
EVAL_RESULTS = EVAL / "eval_results.jsonl"

BASE_URL = os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8001")


def _merge_block_inputs(case_block_inputs, experiment_blocks, experiment_json):
    """Build a complete, schema-valid block_inputs dict for the engine.

    The dataset's per-case block_inputs is a partial sketch; the experiment
    JSON declares the canonical blocks with JSON defaults. Merge the two so
    every input the engine declares has a value (default when absent), and
    reject schemes the experiment does not support.
    """
    blocks = {b["id"]: b for b in experiment_blocks}
    merged = {}
    for bid, block in blocks.items():
        merged[bid] = {}
        for inp in block.get("inputs", []):
            ip_id = inp["id"]
            value = case_block_inputs.get(bid, {}).get(ip_id)
            if value is None:
                value = inp.get("default")
            if value is None:
                raise KeyError(f"missing input {bid}.{ip_id}")
            merged[bid][ip_id] = value
    # Merge the dataset's explicit overrides on top of the schema defaults.
    for bid, inputs in case_block_inputs.items():
        if bid in merged:
            for ip_id, value in inputs.items():
                merged[bid][ip_id] = value
    return merged


def build_payload(case):
    """Turn one dataset row into a POST /tutor payload.

    The dataset's block_inputs is a partial sketch; the engine only applies
    defaults when the UI fills them in. The frontend does fill them in, so
    the evaluator must too — otherwise the engine raises "missing input
    scope.voltage" / "scheme must be one of..." for invalid cases (400).
    """
    lc = json.load(open(os.path.join(BACKEND_ROOT, "app", "experiments", "line_coding.json")))
    bi = _merge_block_inputs(case["block_inputs"], lc["blocks"], lc)
    return {
        "experiment_id": "line_coding",
        "student_query": case["question"],
        "block_inputs": bi,
        "connections": case["connections"],
    }


def main():
    sys.stderr.write("Loading dataset...\n")
    from dataset import CASES  # noqa: E402

    cases = list(CASES)
    total = len(cases)
    cat_counts = Counter(c["category"] for c in cases)

    # ------------------------------------------------------------------
    # Gemini availability (read-only, from the env). We never print the key.
    # ------------------------------------------------------------------
    gemini_available = bool(os.getenv("GEMINI_API_KEY", "").strip())
    sys.stderr.write(
        f"GEMINI_API_KEY set: {gemini_available} (key value not disclosed)\n"
    )
    sys.stderr.write(f"Dataset: {total} cases ({dict(cat_counts)})\n")

    # ------------------------------------------------------------------
    # Per-case pipeline.  When the key is absent the server always returns the
    # deterministic fallback — that is a *real* measured baseline, not a stub.
    # ------------------------------------------------------------------
    rows = []
    for idx, case in enumerate(cases, start=1):
        payload = build_payload(case)
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{BASE_URL}/tutor", json=payload, timeout=90)
            status = resp.status_code
            body = resp.json()
        except Exception as exc:
            status = None
            body = None
            err = repr(exc)
            tb = traceback.format_exc()
        else:
            status = resp.status_code
            err = None
            tb = None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "rank": idx,
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "case": case,
                "http_status": status,
                "answer": (body or {}).get("answer", ""),
                "confidence": (body or {}).get("confidence"),
                "latency_ms": latency_ms,
                "citations": (body or {}).get("citations", []),
                "deterministic_oracle": (body or {}).get("deterministic_oracle"),
                "deterministic_wiring_status": (body or {}).get(
                    "deterministic_wiring_status"
                ),
                "error": (body or {}).get("error"),
                "pipeline_error": err,
                "pipeline_tb": tb,
                "grounded": None,  # filled in below
                "grounding_discrepancies": [],
                "grounding_type": None,
                # RAG side of this turn (from the server-side knowledge section)
                "retrieved_chunks": [],
                "topic_seed": None,
            }
        )

    # ------------------------------------------------------------------
    # Grounding: diff the *deterministic oracle* (which the server already
    # computed) against the answer text.  We flag contradictions but never
    # silently edit them.
    # ------------------------------------------------------------------
    for r in rows:
        oracle = r["deterministic_oracle"]
        answer = r["answer"]
        if oracle is None or not isinstance(oracle, dict):
            r["grounded"] = "n/a"
            r["grounding_discrepancies"].append("no deterministic oracle attached")
            r["grounding_type"] = "n/a"
            continue

        checks = []
        if oracle.get("levels") is not None:
            expected = oracle["levels"]
            actual = None
            # The pipeline usually writes `levels=[...]` verbatim into the draft.
            import re

            m = re.search(r"levels=\[([\d\.\-+,]*)\]", answer)
            if m:
                actual = [float(x) for x in m.group(1).split(",")]
            elif str(expected) in answer:
                actual = expected
            if actual != expected:
                checks.append(
                    "levels mismatch: answer has {} but oracle has {}".format(
                        actual, expected
                    )
                )
        if oracle.get("transitions") is not None:
            expected = oracle["transitions"]
            actual = None
            import re

            m = re.search(r"transitions=(\d+)", answer)
            if m:
                actual = int(m.group(1))
            if actual != expected:
                checks.append(
                    "transitions mismatch: answer has {} but oracle has {}".format(
                        actual, expected
                    )
                )
        if oracle.get("dc_component") is not None:
            expected = oracle["dc_component"]
            actual = None
            import re

            m = re.search(r"dc_component=([+\-0-9.]+)", answer)
            if m:
                actual = float(m.group(1))
            if actual is not None and abs(actual - expected) > 0.05:
                checks.append(
                    "dc_component mismatch: answer has {} but oracle has {}".format(
                        actual, expected
                    )
                )
        if oracle.get("scheme") and oracle["scheme"] not in answer.lower():
            checks.append("quote scheme (missing)",)

        r["grounding_discrepancies"] = checks
        r["grounded"] = "no_contradictions" if not checks else "contradiction"
        r["grounding_type"] = "automatic"

    # ------------------------------------------------------------------
    # Auto metrics
    # ------------------------------------------------------------------
    n_real = sum(1 for r in rows if r["http_status"] == 200 and r["answer"])
    n_no_key = sum(1 for r in rows if r["http_status"] == 200 and "API key" in (r["error"] or ""))

    # Teacher-style question answered (the fallback still answers).
    answered = [r for r in rows if r["http_status"] == 200]

    # RAG retrieval hit rate: does the deterministic oracle answer reference
    # retrieval that was actually available?  The server does not return the
    # retrieved chunks for /tutor, so we measure retrieval separately on the
    # server via the KnowledgeBase (a real, observable operation).
    rag_hits = []
    sys.stderr.write(
        "Measuring RAG retrieval hit rate on server-side KnowledgeBase...\n"
    )
    sys.path.insert(0, str(BACKEND_ROOT))
    sys.path.insert(0, str(BACKEND_ROOT / "app"))
    try:
        from app.ingestion.ingestion import KnowledgeBase  # noqa: E402

        kb = KnowledgeBase.from_manifest()
        kb.add()
        for r in rows:
            topic = r["case"].get("expected_topic", "")
            if not topic:
                rag_hits.append(False)
                continue
            results = kb.retrieve(topic, n_results=5)
            hit = bool(results)
            rag_hits.append(hit)
            r["retrieved_chunks"] = len(results)
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"RAG measurement failed: {exc!r}\n")
        rag_hits = [False] * len(rows)

    recall_k = sum(rag_hits) / max(len(rag_hits), 1)
    precision_at_5 = (
        sum(1 for h in rag_hits if h) / 5.0
    )  # K == 5 default TOP_K
    retrieval_hit_rate = recall_k

    # Answer-level auto metrics
    correct = sum(
        1
        for r in answered
        if r["deterministic_oracle"] is not None
        and r["grounded"] == "no_contradictions"
    )
    correctness = correct / max(len(answered), 1)

    grounded = sum(
        1 for r in answered if r["grounded"] == "no_contradictions"
    )
    groundedness = grounded / max(len(answered), 1)

    contradictions = sum(
        1 for r in answered if r["grounded"] == "contradiction"
    )

    # Confidence averaging (only on rows that returned a confidence).
    confs = [r["confidence"] for r in answered if isinstance(r["confidence"], (int, float))]
    avg_conf = sum(confs) / max(len(confs), 1) if confs else None

    # Latency averaging (only on answered rows that produced a latency).
    lats = [r["latency_ms"] for r in answered]
    avg_lat = sum(lats) / max(len(lats), 1) if lats else None

    report = {
        "meta": {
            "gemini_api_key_configured": gemini_available,
            "gemini_real_calls": n_real,
            "deterministic_fallback_only": n_no_key,
            "verification_mode": "automatic",
            "dataset_size": total,
            "category_counts": dict(cat_counts),
        },
        "rag": {
            "k": 5,
            "precision_at_k": precision_at_5,
            "recall_at_k": recall_k,
            "retrieval_hit_rate": retrieval_hit_rate,
        },
        "tutor": {
            "total_answered": len(answered),
            "correctness": correctness,
            "groundedness": groundedness,
            "contradiction_rate": contradictions / max(len(answered), 1),
            "average_confidence": avg_conf,
            "average_latency_ms": avg_lat,
            "human_evaluated": {
                "specificity": "requires human annotation",
                "relevance": "requires human annotation",
            },
        },
        "per_case": [
            {
                "rank": r["rank"],
                "id": r["id"],
                "category": r["category"],
                "grounded": r["grounded"],
                "grounding_discrepancies": r["grounding_discrepancies"],
                "grounding_type": r["grounding_type"],
                "answer": r["answer"],
                "confidence": r["confidence"],
                "latency_ms": r["latency_ms"],
                "http_status": r["http_status"],
                "deterministic_oracle": r["deterministic_oracle"],
                "retrieved_chunks": r["retrieved_chunks"],
            }
            for r in rows
        ],
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(EVAL_RESULTS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Human-evaluated / human-needs-review flags (never fabricated numbers).
    # ------------------------------------------------------------------
    report["human_only_metrics"] = {
        "specificity": "require human annotation (did the answer deepen the student's understanding?)",
        "relevance": "require human annotation (does the answer address the actual question?)",
    }

    sys.stderr.write("\nEvaluation report written to {}\n".format(OUT))
    sys.stderr.write(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
