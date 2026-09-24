"""
In-process performance profile of the MarketingIQ backend (no network, no real LLM).

    python perf_profile.py                    # startup, memory, DuckDB ops, endpoint stages
    python perf_profile.py --threads 1,2,4,8  # also sweep DuckDB thread settings
    python perf_profile.py --json out.json

Sections:
  startup   time and process RSS for each startup phase (imports, CSV parse, DuckDB table)
  memory    every full copy of the dataset and its size
  duckdb    per analytics operation: whole tool time vs time inside DuckDB, queries per call
  threads   the same operations and a concurrent mix under each DuckDB `threads` setting
  endpoints /api/health, /api/query and /api/chat (direct and LLM_REQUIRED with a zero-delay
            mock provider) split into routing / DuckDB / context / provider / validation time

The external providers are patched to fail if called: nothing here uses Gemini or Groq quota.
Needs psutil (dev only: pip install psutil).
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

HERE = Path(__file__).parent
PROC = psutil.Process()


def rss_mb():
    return round(PROC.memory_info().rss / 1e6, 1)


def stats(times_ms):
    t = sorted(times_ms)
    return {"n": len(t), "min": round(t[0], 3), "median": round(statistics.median(t), 3),
            "p95": round(t[max(0, int(len(t) * 0.95) - 1)], 3), "max": round(t[-1], 3)}


# ---------------------------------------------------------------------------
# Startup (this must run first, in a fresh process)
# ---------------------------------------------------------------------------

def profile_startup():
    out = {"rss_bare_python_mb": rss_mb()}
    t = time.perf_counter()
    import duckdb  # noqa: F401
    import pandas  # noqa: F401
    out["import_pandas_duckdb_ms"], out["rss_after_pandas_duckdb_mb"] = round((time.perf_counter() - t) * 1000), rss_mb()
    t = time.perf_counter()
    import fastapi  # noqa: F401
    import google.genai  # noqa: F401
    import httpx  # noqa: F401
    import uvicorn  # noqa: F401
    out["import_web_and_sdk_ms"], out["rss_after_web_and_sdk_mb"] = round((time.perf_counter() - t) * 1000), rss_mb()

    import data_tools as dt
    phases = {}
    real_read_csv, real_create = dt.pd.read_csv, dt.create_repository

    def timed(name, fn):
        def wrapper(*a, **k):
            s = time.perf_counter()
            r = fn(*a, **k)
            phases[name] = (round((time.perf_counter() - s) * 1000, 1), rss_mb())
            return r
        return wrapper
    dt.pd.read_csv, dt.create_repository = timed("csv_parse", real_read_csv), timed("duckdb_table_build", real_create)
    os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "profile-placeholder"))
    t = time.perf_counter()
    import main  # noqa: F401  (loads the data, builds the app)
    out["import_main_ms"] = round((time.perf_counter() - t) * 1000)
    dt.pd.read_csv, dt.create_repository = real_read_csv, real_create
    out["csv_parse_ms"], out["rss_after_csv_parse_mb"] = phases["csv_parse"]
    out["duckdb_table_build_ms"], out["rss_after_duckdb_table_mb"] = phases["duckdb_table_build"]
    out["rss_after_startup_mb"] = rss_mb()
    return out


# ---------------------------------------------------------------------------
# Memory inventory
# ---------------------------------------------------------------------------

def profile_memory():
    import data_tools as dt
    repo = dt.get_repository()
    out = {"backend": dt.data_backend_name(), "csv_file_mb": round(os.path.getsize(os.environ["_PROFILE_CSV"]) / 1e6, 2)}
    df = getattr(dt, "_df", None)
    out["pandas_global_dataframe_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 2) if df is not None else None
    if hasattr(repo, "_con"):
        rows = repo._con.execute("SELECT tag, memory_usage_bytes FROM duckdb_memory() WHERE memory_usage_bytes > 0").fetchall()
        out["duckdb_memory_mb"] = {tag: round(b / 1e6, 2) for tag, b in rows}
        out["duckdb_settings"] = dict(zip(("threads", "memory_limit"), repo._con.execute(
            "SELECT current_setting('threads'), current_setting('memory_limit')").fetchone()))
    import query_router as qr
    out["router_entity_index_kb"] = round(len(json.dumps(qr._get_entity_index(), default=str)) / 1e3, 1)
    out["rss_mb"] = rss_mb()
    return out


# ---------------------------------------------------------------------------
# DuckDB operations: tool time vs DuckDB time
# ---------------------------------------------------------------------------

class _Acc:
    """Stage timings for the request being measured. A plain shared object, not thread-local:
    FastAPI runs the endpoint in a worker thread, and measurements here are sequential."""


_local = _Acc()


def _instrument_repo(repo):
    """Accumulate time spent inside DuckDB (execute + fetch) per thread."""
    real = repo._run

    def run(sql, params):
        s = time.perf_counter()
        try:
            return real(sql, params)
        finally:
            _local.sql_ms = getattr(_local, "sql_ms", 0.0) + (time.perf_counter() - s) * 1000
            _local.sql_n = getattr(_local, "sql_n", 0) + 1
    repo._run = run
    return lambda: setattr(repo, "_run", real)


def _reset():
    _local.sql_ms, _local.sql_n = 0.0, 0


def profile_duckdb(runs):
    import data_tools as dt
    from benchmark_data_layer import OPERATIONS
    restore = _instrument_repo(dt.get_repository())
    out = []
    try:
        for label, tool, kwargs in OPERATIONS:
            fn = dt.TOOL_REGISTRY[tool]
            for _ in range(5):
                fn(**kwargs)
            total, sql = [], []
            for _ in range(runs):
                _reset()
                s = time.perf_counter()
                fn(**kwargs)
                total.append((time.perf_counter() - s) * 1000)
                sql.append(_local.sql_ms)
            out.append({"operation": label, "queries_per_call": _local.sql_n, "tool": stats(total),
                        "inside_duckdb_median_ms": round(statistics.median(sql), 3),
                        "python_overhead_median_ms": round(statistics.median(total) - statistics.median(sql), 3)})
    finally:
        restore()
    return out


def _cpu_s():
    t = PROC.cpu_times()
    return t.user + t.system


def profile_threads(settings, rounds, concurrency, per_round):
    """Interleaved: every round runs each setting once, in a rotated order, so laptop turbo and
    thermal drift spread evenly over the settings instead of favouring whichever ran first.
    `threads` is a database-wide setting: every per-thread cursor shares it."""
    import data_tools as dt
    from benchmark_data_layer import OPERATIONS
    repo = dt.get_repository()
    ops = [(label, dt.TOOL_REGISTRY[tool], kwargs) for label, tool, kwargs in OPERATIONS
           if tool != "list_available_fields"]  # computed once at startup, not per request
    original = repo._con.execute("SELECT current_setting('threads')").fetchone()[0]
    acc = {n: {"seq": {label: [] for label, _, _ in ops}, "seq_cpu": 0.0, "seq_n": 0,
               "conc_wall": 0.0, "conc_cpu": 0.0, "conc_n": 0} for n in settings}
    try:
        for r in range(rounds):
            order = settings[r % len(settings):] + settings[:r % len(settings)]
            for n in order:
                repo._con.execute(f"SET threads = {int(n)}")
                a = acc[n]
                for _, fn, kwargs in ops:
                    fn(**kwargs)  # warm after the switch
                c0 = _cpu_s()
                for _ in range(per_round):
                    for label, fn, kwargs in ops:
                        s = time.perf_counter()
                        fn(**kwargs)
                        a["seq"][label].append((time.perf_counter() - s) * 1000)
                a["seq_cpu"] += _cpu_s() - c0
                a["seq_n"] += per_round * len(ops)
                work = [(fn, kwargs) for _, fn, kwargs in ops] * per_round * 2
                c0, s = _cpu_s(), time.perf_counter()
                with ThreadPoolExecutor(concurrency) as ex:
                    list(ex.map(lambda w: w[0](**w[1]), work))
                a["conc_wall"] += time.perf_counter() - s
                a["conc_cpu"] += _cpu_s() - c0
                a["conc_n"] += len(work)
    finally:
        repo._con.execute(f"SET threads = {int(original)}")
    results = []
    for n in settings:
        a = acc[n]
        per_op = {label: round(statistics.median(v), 3) for label, v in a["seq"].items()}
        results.append({"threads": n, "single_request_median_ms": per_op,
                        "sum_of_medians_ms": round(sum(per_op.values()), 2),
                        "sequential_cpu_ms_per_op": round(a["seq_cpu"] * 1000 / a["seq_n"], 2),
                        f"concurrent_{concurrency}_workers": {
                            "ops": a["conc_n"], "ops_per_s": round(a["conc_n"] / a["conc_wall"], 1),
                            "avg_cores_busy": round(a["conc_cpu"] / a["conc_wall"], 2),
                            "cpu_ms_per_op": round(a["conc_cpu"] * 1000 / a["conc_n"], 2)}})
    return results


# ---------------------------------------------------------------------------
# Endpoints, split into stages
# ---------------------------------------------------------------------------

DIRECT_QUERIES = {
    "total revenue": "What is total revenue?",
    "total spend": "What is total spend?",
    "campaign count": "How many campaigns are there?",
    "platform ranking": "Which platform has the highest ROAS?",
    "monthly trend": "Show monthly revenue.",
    "filtered campaigns": "Show campaigns with spend over 10000.",
    "ROAS threshold": "Show campaigns with ROAS over 8.",
}
LLM_QUERIES = [
    "Why is TikTok performing better than other platforms?",
    "Why do campaigns with spend over 10000 have lower ROAS?",
    "Explain what the monthly revenue trend means.",
    "Explain how creative age affects CTR.",
]


def profile_endpoints(runs):
    from unittest.mock import patch

    from fastapi.testclient import TestClient
    from google.genai import models as genai_models

    import data_tools as dt
    import grounding
    import main
    import query_router as qr
    from llm_adapter import LLMProvider

    class InstantProvider(LLMProvider):
        """Zero-delay mock: answers with the deterministic data summary (a realistic, grounded
        text for the validator to check)."""
        name, model = "mock", "instant"

        def generate_explanation(self, question, analysis, timeout_s):
            s = time.perf_counter()
            text = grounding.safe_summary(analysis).split("\n\n")[0]
            _local.provider_ms = getattr(_local, "provider_ms", 0.0) + (time.perf_counter() - s) * 1000
            return text

    def stage(module, attr, key):
        real = getattr(module, attr)

        def wrapper(*a, **k):
            s = time.perf_counter()
            try:
                return real(*a, **k)
            finally:
                setattr(_local, key, getattr(_local, key, 0.0) + (time.perf_counter() - s) * 1000)
        return patch.object(module, attr, wrapper)

    keys = ("classify_ms", "execute_ms", "sql_ms", "context_ms", "provider_ms", "validate_ms")
    restore = _instrument_repo(dt.get_repository())
    client = TestClient(main.app)
    out = {}
    guard = patch.object(genai_models.Models, "generate_content", side_effect=AssertionError("real Gemini call"))
    with guard, patch.object(main.llm_adapter, "provider", InstantProvider()), \
            stage(qr, "classify_query", "classify_ms"), stage(qr, "_execute", "execute_ms"), \
            stage(grounding, "build_llm_context", "context_ms"), \
            stage(grounding, "validate_explanation", "validate_ms"):
        def measure(label, call, check):
            for _ in range(5):
                check(call())
            samples = {k: [] for k in ("http_total_ms", "server_elapsed_ms", *keys)}
            for _ in range(runs):
                for k in keys:
                    setattr(_local, k, 0.0)
                _local.sql_n = 0
                s = time.perf_counter()
                body = call()
                samples["http_total_ms"].append((time.perf_counter() - s) * 1000)
                samples["server_elapsed_ms"].append(body.get("elapsed_ms", 0.0) if isinstance(body, dict) else 0.0)
                for k in keys:
                    samples[k].append(getattr(_local, k, 0.0))
                check(body)
            row = {"http_total": stats(samples["http_total_ms"])}
            row["median_ms"] = {k.replace("_ms", ""): round(statistics.median(v), 3) for k, v in samples.items()
                                if k != "http_total_ms"}
            row["median_ms"]["duckdb"] = row["median_ms"].pop("sql")
            row["median_ms"]["tool_python"] = round(row["median_ms"].pop("execute") - row["median_ms"]["duckdb"], 3)
            out[label] = row

        def ok(route):
            def check(body):
                assert body.get("route") == route, body
            return check

        measure("/api/health", lambda: client.get("/api/health").json(), lambda b: b["status"] == "ok" or 1 / 0)
        for label, q in DIRECT_QUERIES.items():
            measure(f"/api/query DIRECT {label}", lambda q=q: client.post("/api/query", json={"query": q}).json(),
                    ok("DIRECT_DATABASE"))
        measure("/api/chat DIRECT total revenue", lambda: client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": DIRECT_QUERIES["total revenue"]}],
                               "filters": {}}).json(), ok("DIRECT_DATABASE"))
        for q in LLM_QUERIES:
            measure(f"/api/query LLM {q}", lambda q=q: client.post("/api/query", json={"query": q}).json(),
                    ok("LLM_REQUIRED"))
        measure(f"/api/chat LLM {LLM_QUERIES[0]}", lambda: client.post(
            "/api/chat", json={"messages": [{"role": "user", "content": LLM_QUERIES[0]}], "filters": {}}).json(),
                ok("LLM_REQUIRED"))
    restore()
    return out


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", default="", help="comma-separated DuckDB thread settings to sweep, e.g. 1,2,4,8")
    ap.add_argument("--concurrency", type=int, default=8, help="Python worker threads for the concurrent mix")
    ap.add_argument("--thread-rounds", type=int, default=12, help="interleaved rounds per thread setting")
    ap.add_argument("--sections", default="startup,memory,duckdb,endpoints")
    ap.add_argument("--json")
    args = ap.parse_args()
    import logging
    sections = args.sections.split(",")
    result = {"python": sys.version.split()[0], "cpu_logical": psutil.cpu_count(),
              "cpu_physical": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    result["startup"] = profile_startup()  # always first: it measures a fresh process
    import main
    os.environ["_PROFILE_CSV"] = main.CSV_PATH
    for name in ("marketingiq.query_router", "marketingiq.llm_adapter", "marketingiq.data", "marketingiq.chat"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if "memory" in sections:
        result["memory"] = profile_memory()
    if "duckdb" in sections:
        result["duckdb"] = profile_duckdb(args.runs)
    if args.threads:
        result["threads"] = profile_threads([int(x) for x in args.threads.split(",")], args.thread_rounds,
                                            args.concurrency, per_round=5)
    if "endpoints" in sections:
        result["endpoints"] = profile_endpoints(max(50, args.runs // 2))
    result["rss_end_mb"] = rss_mb()
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main_()
