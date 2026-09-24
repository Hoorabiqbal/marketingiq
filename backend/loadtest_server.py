"""
Instrumented MarketingIQ server for load tests. loadtest.py starts it; production never uses it.

Runs the real FastAPI app (main.app) under uvicorn with two test-only changes:
  - Gemini (google-genai generate_content) and Groq calls are counted and refused, so a load
    test can never spend provider quota. The counters must stay 0.
  - --mock-llm-delay-ms N replaces the active LLM provider with a deterministic mock that sleeps
    N ms and answers with a grounded summary of the context it was given. The LLM_REQUIRED path
    (router, DuckDB, typed context, adapter, grounding validator) then runs for real without
    Gemini, Groq or the internet.

GET /__loadtest/counters returns the counters (and DuckDB's thread setting).
"""
import argparse
import os
import threading
import time

import uvicorn
from google.genai import models as genai_models

import groq_provider

COUNTERS = {"gemini_calls": 0, "groq_calls": 0, "mock_llm_calls": 0, "entity_index_builds": 0}
_lock = threading.Lock()


def _count(name):
    with _lock:
        COUNTERS[name] += 1


def _refuse_gemini(*args, **kwargs):
    _count("gemini_calls")
    raise RuntimeError("load test: real Gemini call refused")


def _refuse_groq(*args, **kwargs):
    _count("groq_calls")
    raise RuntimeError("load test: real Groq call refused")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mock-llm-delay-ms", type=float)
    ap.add_argument("--duckdb-threads", type=int, help="override DuckDB's thread count (benchmarking only)")
    args = ap.parse_args()

    genai_models.Models.generate_content = _refuse_gemini
    groq_provider.GroqProvider.generate_explanation = _refuse_groq
    os.environ.setdefault("GEMINI_API_KEY", "loadtest-placeholder")

    import query_router
    build = query_router._build_entity_index

    def counted_build():  # must run exactly once per process (main.py warms it at startup)
        _count("entity_index_builds")
        return build()
    query_router._build_entity_index = counted_build

    import grounding
    import main as app_main
    from llm_adapter import LLMProvider

    if args.duckdb_threads:
        app_main.dt.get_repository()._con.execute(f"SET threads = {int(args.duckdb_threads)}")

    if args.mock_llm_delay_ms is not None:
        delay_s = args.mock_llm_delay_ms / 1000

        class DelayProvider(LLMProvider):
            name, model = "mock", f"fixed-delay-{args.mock_llm_delay_ms:g}ms"

            def generate_explanation(self, question, analysis, timeout_s):
                _count("mock_llm_calls")
                time.sleep(delay_s)  # stands in for provider/network time; blocks a worker like a real call
                return grounding.safe_summary(analysis).split("\n\n")[0]

        app_main.llm_adapter.provider = DelayProvider()

    @app_main.app.get("/__loadtest/counters")
    def counters():
        with _lock:
            threads = app_main.dt.get_repository()._con.execute("SELECT current_setting('threads')").fetchone()[0]
            return dict(COUNTERS, provider=app_main.llm_adapter.provider.name, duckdb_threads=threads,
                        pid=os.getpid())

    uvicorn.run(app_main.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
