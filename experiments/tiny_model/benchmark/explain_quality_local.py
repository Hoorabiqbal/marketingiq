"""
HISTORICAL (failed gate, kept as evidence): local Qwen explanation/recommendation quality test.

It needs backend/local_explainer.py, which was removed after this gate FAILED (wrong budget direction,
loss-making read as revenue < 0, garbled CPA advice); see results/explain_quality_*.json. The final
architecture answers the same questions with deterministic findings: findings_quality_local.py.
It no longer runs against the current backend.

Original description: local explanation/recommendation quality test (the gate before Groq is removed).

Needs the running Q4_K_M llama-server. For each question: evidence plan (Qwen planner or the
remembered plan) -> validate_plan() -> DuckDB evidence -> Qwen explanation -> grounding check.
Prints every answer in full so quality can be judged by reading, and saves the run to
benchmark/results/. No question is from eval_frozen.jsonl (checked).

    venv\\Scripts\\python experiments\\tiny_model\\benchmark\\explain_quality_local.py
"""
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
os.chdir(REPO / "backend")
sys.path.insert(0, str(REPO / "backend"))

import data_tools as dt  # noqa: E402

dt.load_data(str(REPO / "data" / "tech_advertising_campaigns_dataset.csv"))
import local_explainer  # noqa: E402
import local_planner  # noqa: E402

# (setup questions run through the planner first (session memory), question, kind, about_previous)
CASES = [
    ([], "Why is TikTok ROAS lower than Google Ads?", "explanation", False),
    ([], "Why did revenue change from month to month?", "explanation", False),
    ([], "Explain how Google Ads and TikTok compare on revenue", "explanation", False),
    (["how has revenue moved month by month"], "explain this trend", "explanation", True),
    ([], "How can we improve revenue on weaker platforms?", "advice", False),
    ([], "Which platforms should get more budget based on spend and revenue efficiency?", "advice", False),
    (["how much revenue did each platform bring in", "switch to profit"], "why?", "explanation", True),
    (["relationship between spend and revenue"], "explain this", "explanation", True),
    ([], "How can we reduce CPA across devices?", "advice", False),
    ([], "Why is the conversion rate different between mobile and desktop?", "explanation", False),
    ([], "What should we do about loss-making campaigns?", "advice", False),
    ([], "Why is ROI highest for the Conversions objective?", "explanation", False),
    ([], "Why are browser sessions so short?", "explanation", False),   # insufficient evidence expected
]


def norm(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    frozen = {norm(json.loads(line)["question"]) for line in
              (HERE.parent / "dataset" / "eval_frozen.jsonl").read_text(encoding="utf-8").splitlines() if line}
    clash = [q for setup, q, _, _ in CASES for q in [*setup, q] if norm(q) in frozen]
    if clash:
        sys.exit(f"questions in eval_frozen.jsonl: {clash}")
    planner = local_planner.LocalPlanner()
    if not planner.healthy(2):
        sys.exit("llama-server not reachable")
    explainer = local_explainer.LocalExplainer(planner)
    planner.complete(planner.prompt("warm-up request, please ignore"))

    rows = []
    for setup, question, kind, about_previous in CASES:
        previous = None
        for s in setup:
            a = planner.answer(s, {}, previous=previous)
            previous = a.plan if a.status == "ok" else previous
        t0 = time.perf_counter()
        e = explainer.explain(question, {}, kind, previous=previous, about_previous=about_previous)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"question": question, "kind": kind, "setup": setup, "status": e.status,
                     "intent": e.plan.intent if e.plan else None, "grounding": e.grounding, "text": e.text,
                     "total_ms": ms, "model_ms": e.model_ms})
        print(f"\n### {question}  [{kind}{' after ' + ' -> '.join(setup) if setup else ''}]")
        print(f"status={e.status} plan={e.plan.intent if e.plan else None} {ms:.0f} ms grounding={e.grounding}")
        print(e.text)
    lat = [r["total_ms"] for r in rows if r["status"] in ("ok", "fallback")]
    summary = {"questions": len(rows), "status": {s: sum(r["status"] == s for r in rows) for s in
                                                  ("ok", "fallback", "insufficient", "model_unavailable")},
               "p50_ms": statistics.median(lat) if lat else None, "p95_ms": pct(lat, 95) if lat else None}
    print("\n" + json.dumps(summary, indent=1))
    out = HERE / "results" / f"explain_quality_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
