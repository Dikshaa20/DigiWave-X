import os
import sys
import time
import subprocess
import urllib.request

sys.path.insert(0, "C:/Users/Admin/OneDrive/Desktop/DigiWave-X/digiwave-x/backend")

# Absolute path to the venv python (Windows NT path format).
PY = "C:/Users/Admin/OneDrive/Desktop/DigiWave-X/digiwave-x/backend/.venv/Scripts/python.exe"

log = open("backend/.tmp/server.log", "wb")
proc = subprocess.Popen(
    [PY, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd="backend",
    stdout=log,
    stderr=log,
    env={**os.environ, "PYTHONPATH": "backend"},
)

import urllib.request as ur

up = False
for _ in range(60):
    try:
        r = ur.urlopen("http://127.0.0.1:8000/circuit/state", timeout=2)
        up = True
        break
    except Exception:
        time.sleep(0.5)
print("server_up:", up, flush=True)

if not up:
    print("SERVER_FAILED", flush=True)
    proc.terminate()
    sys.exit(1)

import requests
from evaluation.dataset import CASES

def build_payload(case):
    bi = dict(case["block_inputs"])
    if "source" in bi and "bits" in bi["source"]:
        bi.setdefault("scope", {})
        bi["scope"].setdefault("voltage", 1.0)
    return {
        "experiment_id": "line_coding",
        "student_query": case["question"],
        "block_inputs": bi,
        "connections": case["connections"],
    }

rows = []
for idx, case in enumerate(CASES, 1):
    payload = build_payload(case)
    t0 = time.perf_counter()
    try:
        resp = requests.post("http://127.0.0.1:8000/tutor", json=payload, timeout=60)
        status = resp.status_code
        body = resp.json()
    except Exception as e:
        status = None
        body = None
        err = repr(e)
    latency = (time.perf_counter() - t0) * 1000.0
    rows.append(
        {
            "rank": idx,
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "case": case,
            "http_status": status,
            "answer": body.get("answer") if body else None,
            "confidence": body.get("confidence") if body else None,
            "latency_ms": latency,
            "error": (body or {}).get("detail", "") if status else None,
            "pipeline_error": err if "err" in dir() else None,
        }
    )

print("TOTAL:", len(rows), "HTTP_200:", sum(1 for r in rows if r["http_status"] == 200), flush=True)

answered = [r for r in rows if r["http_status"] == 200]
print("answered:", len(answered), flush=True)
for r in rows[:3]:
    print(r["id"], r["http_status"], str(r["answer"])[:120] if r["answer"] else None, flush=True)
for r in rows[-3:]:
    print(r["id"], r["http_status"], str(r["answer"])[:120] if r["answer"] else None, flush=True)

proc.terminate()
try:
    proc.wait(timeout=10)
except Exception:
    proc.kill()
print("DONE", flush=True)
