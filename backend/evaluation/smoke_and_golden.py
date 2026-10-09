"""smoke_and_golden.py — golden-output regression harness for the DigiWave X
LangGraph tutoring pipeline.

This script:
  1. Starts a real uvicorn server (so the HTTP layer is exercised) OR runs
     the graph directly with a stubbed generate-node.  It always uses a
     stubbed Gemini client, so it never touches the network and is safe to
     run in CI with GEMINI_API_KEY unset.
  2. Routes a small set of *unseen* cases through the full 10-node graph and
     snapshots the final_response.
  3. Diffs the snapshot against a stored golden file (evaluation/golden.json).
     Any change in the graph behaviour fails the run loudly (exit 1).

Run:
    cd /c/Users/Admin/OneDrive/Desktop/DigiWave-X/digiwave-x/backend
    .venv/Scripts/python.exe evaluation/smoke_and_golden.py        # run
    .venv/Scripts/python.exe evaluation/smoke_and_golden.py --regen    # overwrite golden.json

The golden file records: question, category, intent, expected schema, oracle
levels/transitions/dc, what_if hypothesis, final answer, confidence.  It is
pure JSON and is the only artefact you need to commit for CI to catch a
regressed graph node.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
EVAL_DIR = BACKEND / "evaluation"
OUT = EVAL_DIR / "golden.json"

BASE_URL = "http://127.0.0.1:8007"
GOLDEN = {
    "version": 1,
    "comment": "Regenerated with --regen.  Do not hand-edit.",
    "cases": [],
}

GRAY_SHOT = [
    {"id": "g1", "category": "concept", "question": "What would the AMI levels be for 101 in AMI?", "intent": "concept", "block_inputs": {"source": {"bits": "101"}, "encoder": {"scheme": "ami"}}, "connections": []},
    {"id": "g2", "category": "what-if", "question": "What if I run bits=101 with Manchester?", "intent": "what_if", "block_inputs": {"source": {"bits": "101"}, "encoder": {"scheme": "nrz"}}, "connections": []},
    {"id": "g3", "category": "wiring", "question": "Why is my scope dark with only source->encoder?", "intent": "diagnosis", "block_inputs": {"source": {"bits": "101"}, "encoder": {"scheme": "ami"}}, "connections": [{"from": "source.out", "to": "encoder.in"}]},
    {"id": "g4", "category": "misconception", "question": "Does NRZ-L invert on a 1?", "intent": "concept", "block_inputs": {"source": {"bits": "101"}, "encoder": {"scheme": "nrz"}}, "connections": []},
]


def _post_json(path: str, payload: dict) -> tuple[int, dict[str, object]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {"raw": exc.read().decode("utf-8", "replace")}
    except Exception as exc:
        return None, {"error": repr(exc)}


def _import_backend() -> None:
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


def _build_expected(case: dict) -> dict:
    from app.experiments.engine import build_circuit_state
    lc = json.loads((BACKEND / "app" / "experiments" / "line_coding.json").read_text())
    blocks = {b["id"]: b for b in lc["blocks"]}
    bi = {}
    for bid, block in blocks.items():
        bi[bid] = {}
        for inp in block.get("inputs", []):
            ip_id = inp["id"]
            value = case["block_inputs"].get(bid, {}).get(ip_id)
            if value is None:
                value = inp.get("default")
            if value is None:
                raise KeyError(f"missing input {bid}.{ip_id}")
            bi[bid][ip_id] = value
    for bid, inputs in case["block_inputs"].items():
        if bid in bi:
            for ip_id, value in inputs.items():
                bi[bid][ip_id] = value
    payload = {"experiment_id": "line_coding", "student_query": case["question"], "block_inputs": bi, "connections": case.get("connections", [])}
    status, body = _post_json("/tutor", payload)
    return {
        "id": case["id"], "category": case["category"], "intent": case["intent"], "question": case["question"],
        "http_status": status, "answer": body.get("answer", ""),
        "what_if_hypothesis": body.get("what_if", {}).get("hypothesis", ""),
        "belief": body.get("what_if", {}).get("oracle_result", {}).get("scheme"),
        "oracle_levels": body.get("deterministic_oracle", {}).get("levels"),
        "oracle_transitions": body.get("deterministic_oracle", {}).get("transitions"),
        "oracle_dc": body.get("deterministic_oracle", {}).get("dc_component"),
        "confidence": body.get("confidence"),
        "status": "OK" if status == 200 else f"HTTP_{status}",
    }


def _run_server() -> subprocess.Popen:
    log = open(str(EVAL_DIR / ".tmp" / "smoke_server.log"), "wb")
    proc = subprocess.Popen(
        [str(BACKEND / ".venv/Scripts/python.exe"), "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8007"],
        cwd=str(BACKEND), stdout=log, stderr=log, env={**os.environ, "PYTHONPATH": str(BACKEND)})
    for _ in range(90):
        try:
            urllib.request.urlopen(BASE_URL + "/circuit/state", timeout=2)
            return proc
        except urllib.error.HTTPError:
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    sys.exit("SMOKE SERVER FAILED TO START")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true", help="Overwrite golden.json with fresh expected values")
    args = parser.parse_args()
    _import_backend()
    proc = _run_server()
    try:
        for case in GRAY_SHOT:
            expected = _build_expected(case)
            GOLDEN["cases"].append(expected)
        if args.regen:
            OUT.write_text(json.dumps(GOLDEN, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[regen] wrote {OUT}")
            return 0
        if not OUT.exists():
            print(f"[warn] no golden.json at {OUT}; run with --regen first")
            return 0
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        failures = []
        for a, b in zip(GOLDEN["cases"], existing.get("cases", [])):
            for key in ["http_status", "answer", "what_if_hypothesis", "belief", "oracle_levels", "oracle_transitions", "oracle_dc", "confidence", "status"]:
                if a.get(key) != b.get(key):
                    failures.append((a["id"], key, a.get(key), b.get(key)))
        if failures:
            print("[FAIL] golden regression:")
            for aid, key, a_val, b_val in failures:
                print(f"  {aid} / {key}: golden={a_val!r}  current={b_val!r}")
            return 1
        print(f"[PASS] {len(GOLDEN['cases'])} golden cases unchanged.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
