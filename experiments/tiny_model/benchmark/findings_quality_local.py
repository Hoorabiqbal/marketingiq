"""
Quality gate for the final architecture: the SAME 13 explanation/advice questions the failed local-Qwen
explainer was tested on (read from its saved results, so the set can't drift), now answered by

    Qwen plan (or the remembered plan) -> validate_plan() -> DuckDB -> findings_engine -> recommendation_engine

through the real /api/chat path with session memory. Needs the running Q4_K_M llama-server.
Prints every answer in full, runs directional checks against independent DuckDB queries, counts model
calls, and saves the run to benchmark/results/. No question is from eval_frozen.jsonl (checked).

    venv\\Scripts\\python experiments\\tiny_model\\benchmark\\findings_quality_local.py
"""
import glob
import html
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
import data_tools as dt  # noqa: E402
import main  # noqa: E402

CAUSAL_RE = re.compile(r"\b(?:because|due to|caused by|causes|results? in|led to|driven by|the reason)\b", re.I)


def previous_gate():
    files = sorted(glob.glob(str(HERE / "results" / "explain_quality_*.json")))
    if not files:
        sys.exit("the failed explainer's results (results/explain_quality_*.json) are needed to reuse its questions")
    rows = json.loads(Path(files[-1]).read_text(encoding="utf-8"))["rows"]
    return [(r["setup"], r["question"], r["kind"], r["text"]) for r in rows], Path(files[-1]).name


def norm(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def text_of(body):
    return html.unescape(body["answer"].replace("<br>", "\n"))


def checks():
    """Directional facts from independent DuckDB queries (not from the findings engine)."""
    roas = dt.rank_dimension("platform", metric="roas", limit=20)["results"]
    cpa = dt.rank_dimension("device", metric="cpa", limit=20)["results"]
    loss = dt.filter_campaigns([{"field": "profit", "operator": "<", "value": 0}], limit=1)["matched_count"]
    return {"roas_best": roas[0]["name"], "roas_worst": roas[-1]["name"], "cpa_highest": cpa[0]["name"],
            "cpa_lowest": cpa[-1]["name"], "loss_count": loss}


def main_():
    cases, source = previous_gate()
    frozen = {norm(json.loads(line)["question"]) for line in
              (HERE.parent / "dataset" / "eval_frozen.jsonl").read_text(encoding="utf-8").splitlines() if line}
    if any(norm(q) in frozen for s, q, _, _ in cases for q in [*s, q]):
        sys.exit("a gate question is in eval_frozen.jsonl")
    if not analyst.planner.healthy(2):
        sys.exit("llama-server not reachable")
    facts = checks()
    client = TestClient(main.app)
    calls = {"n": 0}
    real = analyst.planner.complete

    def counted(prompt):
        calls["n"] += 1
        return real(prompt)
    analyst.planner.complete = counted

    rows = []
    for setup, question, kind, old_text in cases:
        sid = str(uuid.uuid4())
        for s in setup:
            client.post("/api/chat", json={"messages": [{"role": "user", "content": s}], "filters": {}, "session_id": sid})
        calls["n"] = 0
        t0 = time.perf_counter()
        body = client.post("/api/chat", json={"messages": [{"role": "user", "content": question}], "filters": {},
                                              "session_id": sid}).json()
        ms = (time.perf_counter() - t0) * 1000
        t = text_of(body)
        rows.append({"question": question, "setup": setup, "kind": kind, "route": body["route"], "text": t,
                     "ms": ms, "model_calls": calls["n"], "old_text": old_text,
                     "causal_language": bool(CAUSAL_RE.search(t.replace("the data doesn't show why", "")
                                                               .replace("not that one causes the other", "")))})
        print(f"\n### {question}  [{kind}{' after ' + ' -> '.join(setup) if setup else ''}]")
        print(f"route={body['route']} model_calls={calls['n']} {ms:.0f} ms")
        print(t)
    analyst.planner.complete = real

    by_q = {r["question"]: r["text"] for r in rows}
    budget = by_q["Which platforms should get more budget based on spend and revenue efficiency?"]
    cpa = by_q["How can we reduce CPA across devices?"]
    loss = by_q["What should we do about loss-making campaigns?"]
    directional = {
        "budget: incremental budget suggested for the highest-ROAS platform":
            f"budget on {facts['roas_best']} (the highest ROAS)" in budget,
        "budget: lowest-ROAS platform is to be reviewed, not funded":
            f"review {facts['roas_worst']} (the lowest ROAS)" in budget
            and f"budget on {facts['roas_worst']}" not in budget,
        "loss-making uses profit < 0 and the DuckDB count":
            f"{facts['loss_count']:,} campaigns" in loss and "profit below $0" in loss,
        "CPA: reduction prioritized on the highest-CPA device":
            f"reduction on {facts['cpa_highest']} (the highest CPA)" in cpa,
        "CPA: lowest-CPA device used as the benchmark": f"Use {facts['cpa_lowest']} (the lowest CPA)" in cpa,
    }
    lat = [r["ms"] for r in rows]
    summary = {"source_questions": source, "questions": len(rows), "facts": facts, "directional_checks": directional,
               "causal_language": sum(r["causal_language"] for r in rows),
               "model_calls": {r["question"]: r["model_calls"] for r in rows},
               "p50_ms": statistics.median(lat), "max_ms": max(lat)}
    print("\n" + json.dumps(summary, indent=1))
    out = HERE / "results" / f"findings_quality_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main_()
