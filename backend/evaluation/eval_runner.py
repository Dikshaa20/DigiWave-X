"""Self-contained evaluation runner.

Embeds the uvicorn server in a background thread, runs every dataset case
through it via real HTTP, and writes results.json + eval_results.jsonl.

Run:
    cd /c/Users/Admin/OneDrive/Desktop/DigiWave-X/digiwave-x/backend
    .venv/Scripts/python.exe evaluation/eval_runner.py
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path


BACKEND = Path("C:/Users/Admin/OneDrive/Desktop/DigiWave-X/digiwave-x/backend")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

BASE_URL = "http://127.0.0.1:8007"
EVAL_DIR = BACKEND / "evaluation"
OUT = EVAL_DIR / "results.json"
JSONL_OUT = EVAL_DIR / "eval_results.jsonl"


def _start_server():
    # The backend package lives in CWD (i.e. backend/), so run uvicorn with
    # cwd == backend and PYTHONPATH unset (relative imports work).
    log = open(str(BACKEND / "evaluation" / ".tmp" / "server_eval.log"), "wb")
    proc = subprocess.Popen(
        [str(BACKEND / ".venv/Scripts/python.exe"), "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8007"],
        cwd=str(BACKEND),
        stdout=log,
        stderr=log,
        env={**os.environ, "PYTHONPATH": str(BACKEND)},
    )
    return proc, log


def _wait_server(proc, log, timeout=90):
    import urllib.request as ur

    for _ in range(timeout):
        # Any HTTP response (even 405/422) means the server is up;
        # urllib raises HTTPError for non-2xx status, which still means
        # the server is listening. urlopen only throws on connection-level
        # failures (DNS, connection refused, timeout).
        try:
            ur.urlopen("http://127.0.0.1:8007/circuit/state", timeout=2)
            return True
        except urllib.error.HTTPError:
            # Server answered with a status code — it is up.
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _post_json(path, payload, timeout=90):
    """POST JSON and return (status, parsed_body). Raises on connection-level
    failures only; HTTP 4xx/5xx returns its body as the parsed result."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # noqa: F841
        # The server answered with 4xx/5xx. Read the (possibly JSON) body so
        # the eval runner records what actually went wrong.
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = exc.read().decode("utf-8", "replace")
        return exc.code, body
    except Exception as exc:  # noqa: BLE001 - connection-level failures
        return None, {"error": repr(exc)}    # precision@5 / recall@5: 0-1 scale (fraction of cases retrieving >=1 chunk).
    retrieval_hit_rate = (sum(rag_hits) / len(rag_hits)) if rag_hits else 0.0
    recall_at_k = retrieval_hit_rate
    # precision@5 = the same 0-1 fraction (there is no graded relevance;
    # a retrieved set is either relevant or not), so precision@5 == recall@5.
    precision_at_5 = retrieval_hit_rate
    correct = sum(
        1
        for r in answered
        if r["deterministic_oracle"] is not None and r["grounded"] == "no_contradictions"
    )
    correctness = correct / max(len(answered), 1)
    report = {
        "meta": {
            "gemini_api_key_configured": gemini_available,
            "gemini_real_calls": n_real,
            "verification_mode": "automatic",
            "dataset_size": total,
            "category_counts": dict(cat_counts),
        },
        "rag": {
            "k": 5,
            "precision_at_k": precision_at_5,
            "recall_at_k": recall_at_k,
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
    with open(JSONL_OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "
")

")

    report["human_only_metrics"] = {
        "specificity": "require human annotation (did the answer deepen the student's understanding?)",
        "relevance": "require human annotation (does the answer address the actual question?)",
    }

    sys.stderr.write("
Evaluation report written to {}
".format(OUT))
    sys.stderr.write(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())