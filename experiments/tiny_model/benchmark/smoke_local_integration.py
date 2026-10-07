"""
Real end-to-end smoke test of the final local MarketingIQ architecture (no cloud LLM, no API key).

Needs the running Q4_K_M llama-server (PHASE7_LOCAL_QWEN.md). The FastAPI app runs in-process
(TestClient) with the real local planner, DuckDB, findings/recommendation engines and chart builder.

    venv\\Scripts\\python experiments\\tiny_model\\benchmark\\smoke_local_integration.py

Covers total, breakdown, comparison, trend, relationship, chart, follow-up, why/explain,
recommendation and an unsupported request. Correctness: each question's hand-written expected plan
is replayed through the same app code with only the model call stubbed; the real answer (text +
chart) must be identical. Also counts model calls per question and confirms no cloud-LLM module
is loaded. Questions are checked not to be in eval_frozen.jsonl.
"""
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.chdir(REPO / "backend")
sys.path.insert(0, str(REPO / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import analyst  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
planner = analyst.planner
REV_PLATFORM = {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform"}

# (question, expected plan for its planning call or None when no call is expected, session tag)
SMOKE = [
    ("What is total revenue?", {"intent": "total", "metrics": ["revenue"]}, None),
    ("how much revenue did each platform bring in", REV_PLATFORM, "conv"),
    ("switch to profit", {"intent": "breakdown", "metrics": ["profit"], "dimension": "platform"}, "conv"),
    ("why?", None, "conv"),
    ("What should we improve?", None, "conv"),
    ("compare Google Ads and TikTok on revenue", {"intent": "compare_entities", "metrics": ["revenue"],
                                                  "dimension": "platform", "entities": ["Google Ads", "TikTok"]}, None),
    ("how has revenue moved month by month", {"intent": "trend", "metrics": ["revenue"], "grain": "month"}, None),
    ("relationship between spend and revenue", {"intent": "relationship", "metrics": ["spend", "revenue"]}, None),
    ("show revenue by platform as a bar chart", {**REV_PLATFORM, "visualization": "bar"}, None),
    ("Why is TikTok ROAS lower than Google Ads?", {"intent": "compare_entities", "metrics": ["roas"],
                                                   "dimension": "platform", "entities": ["TikTok", "Google Ads"]}, None),
    ("How can we reduce CPA across devices?", {"intent": "breakdown", "metrics": ["cpa"], "dimension": "device"}, None),
    ("What should we do about loss-making campaigns?", {"intent": "campaign_list", "metrics": ["profit"],
                                                        "conditions": [{"metric": "profit", "operator": "<", "value": 0}]}, None),
    ("Compare impressions across browsers", {"intent": "breakdown", "metrics": ["impressions"], "dimension": "browser"}, None),
]


def norm(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def pct(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def post(question, sid):
    return client.post("/api/chat", json={"messages": [{"role": "user", "content": question}], "filters": {},
                                          "session_id": sid}).json()


def main_():
    frozen = {norm(json.loads(line)["question"])
              for line in (HERE.parent / "dataset" / "eval_frozen.jsonl").read_text(encoding="utf-8").splitlines() if line}
    clash = [q for q, _, _ in SMOKE if norm(q) in frozen]
    if clash:
        sys.exit(f"smoke questions found in eval_frozen.jsonl: {clash}")
    if not planner.healthy(timeout_s=2):
        sys.exit(f"llama-server is not reachable at {planner.url}; start it first (PHASE7_LOCAL_QWEN.md)")
    real_complete = planner.complete
    real_complete(planner.prompt("warm-up request, please ignore"))   # untimed: shared system-prompt prefix

    calls = {"n": 0, "ms": 0.0, "raw": None}

    def counted(prompt):
        calls["n"] += 1
        t0 = time.perf_counter()
        out = real_complete(prompt)
        calls["ms"] += (time.perf_counter() - t0) * 1000
        calls["raw"] = out.get("content")
        return out

    sessions, expected_sessions, rows = {}, {}, []
    for question, plan, tag in SMOKE:
        sid = sessions.setdefault(tag, str(uuid.uuid4())) if tag else str(uuid.uuid4())
        esid = expected_sessions.setdefault(tag, str(uuid.uuid4())) if tag else str(uuid.uuid4())
        planner.complete = counted
        calls.update(n=0, ms=0.0, raw=None)
        t0 = time.perf_counter()
        body = post(question, sid)
        backend_ms = (time.perf_counter() - t0) * 1000
        real = dict(calls)
        # The expected answer: the same app code, with the model call returning the expected plan.
        planner.complete = (lambda prompt, p=plan: {"content": json.dumps(p), "stop_type": "eos", "timings": {}})
        expected = post(question, esid)
        planner.complete = real_complete
        correct = body["answer"] == expected["answer"] and body.get("chart") == expected.get("chart") \
            and body["route"] == expected["route"]
        rows.append({"question": question, "route": body["route"], "model_calls": real["n"], "qwen_ms": real["ms"],
                     "backend_ms": backend_ms, "raw_output": real["raw"], "expected_calls": 0 if plan is None else 1,
                     "chart": (body.get("chart") or {}).get("type"), "correct": correct})
        print(f"{'OK  ' if correct else 'DIFF'} {backend_ms:6.0f} ms  calls {real['n']}  {body['route']:15s} {question!r}"
              f"\n      model: {real['raw']}")
        if not correct:
            print("      expected:", expected["answer"][:300], "\n      got:     ", body["answer"][:300])

    planned = [r for r in rows if r["model_calls"]]
    q = [r["qwen_ms"] for r in planned]
    b = [r["backend_ms"] for r in rows]
    cloud = sorted(m for m in sys.modules if m.split(".")[0] in ("groq", "openai", "anthropic", "google")
                   or "groq" in m.lower() or "gemini" in m.lower())
    summary = {
        "questions": len(rows), "correct": f"{sum(r['correct'] for r in rows)}/{len(rows)}",
        "model_calls_as_expected": f"{sum(r['model_calls'] == r['expected_calls'] for r in rows)}/{len(rows)}",
        "second_model_calls": sum(r["model_calls"] > 1 for r in rows),
        "routes": {r["question"]: r["route"] for r in rows},
        "charts": {r["question"]: r["chart"] for r in rows if r["chart"]},
        "qwen_ms": {"p50": statistics.median(q), "p95": pct(q, 95)} if q else None,
        "backend_ms": {"p50": statistics.median(b), "p95": pct(b, 95)},
        "cloud_llm_modules_loaded": cloud,
    }
    print("\n" + json.dumps(summary, indent=1))
    out = HERE / "results" / f"final_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main_()
