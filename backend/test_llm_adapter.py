"""
Tests for the LLM Adapter (llm_adapter.py) and its integration with the Query Router's
LLM_REQUIRED route, through the real GroqProvider on a fake transport (fake_groq.py).

No real Groq call is ever made: the app's provider is blocked for the whole module, and each
test that needs an answer supplies its own fake.

Run:  python test_llm_adapter.py      (or: pytest test_llm_adapter.py)
"""
import json
import logging
import os
from unittest.mock import MagicMock, patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)

import httpx
from fastapi.testclient import TestClient

import grounding
import main
import query_router as qr
from llm_adapter import BUSY_MESSAGE, GROUNDING_INSTRUCTIONS, LLMAdapter, LLMProvider, ProviderSlots

GUARD = fake_groq.block_real_calls(main.llm_adapter)  # nothing below reaches api.groq.com
client = TestClient(main.app)
WHY_Q = "Why is TikTok performing better?"
SECRET_KEY = "gsk_SECRET123_fake"


def groq_adapter(respond=fake_groq.reply("ok"), api_key=fake_groq.FAKE_KEY, timeout_s=10.0):
    provider, fake = fake_groq.provider(respond, api_key=api_key)
    return LLMAdapter(provider, timeout_s=timeout_s), fake


def sent_analysis(fake) -> dict:
    return json.loads(fake.user_prompt().split("ANALYSIS (JSON):\n", 1)[1])


class RecordingProvider(LLMProvider):
    name, model = "recording", "none"

    def __init__(self, reply="explained"):
        self.calls, self.reply = [], reply

    def generate_explanation(self, question, analysis, timeout_s):
        self.calls.append((question, analysis, timeout_s))
        return self.reply


SAMPLE = {"filters_applied": {}, "focus_metrics": ["roas"],
          "results": [{"tool": "get_totals", "tool_input": {}, "result": {"campaign_count": 5, "roas": 2.5}}]}


# 1. Provider interface ------------------------------------------------------

def test_provider_interface_accepts_question_and_analysis():
    provider = RecordingProvider()
    out = LLMAdapter(provider, timeout_s=7).explain("Why?", SAMPLE)
    assert out["status"] == "ok" and out["text"] == "explained" and out["provider"] == "recording"
    # The provider gets the typed, unit-aware context built from the analysis (grounding.py).
    assert provider.calls == [("Why?", grounding.build_llm_context("Why?", SAMPLE), 7)]
    assert provider.calls[0][1]["metric_units"]["roas"].startswith("x:")
    try:
        LLMProvider()
        raise AssertionError("LLMProvider must be abstract")
    except TypeError:
        pass


# 2 & 3. Groq gets the compact analysis, never the dataset ------------------

def test_groq_receives_compact_router_analysis():
    adapter, fake = groq_adapter(fake_groq.reply("TikTok leads on ROAS."))
    r = qr.route_query(WHY_Q, {"budget": "High"}, explainer=adapter)
    assert len(fake.requests) == 1  # one LLM call per request
    sent = sent_analysis(fake)
    expected = grounding.build_llm_context(WHY_Q, {"filters_applied": {"budget": "High"},
                                                   "focus_metrics": r["focus_metrics"], "results": r["analysis"]})
    assert sent == json.loads(json.dumps(expected))  # the router's analysis, typed — nothing else
    assert sent["active_filters"] == {"budget_tier": "High"} and sent["population"] == "dashboard-filtered view"
    assert fake.body()["messages"][0] == {"role": "system", "content": GROUNDING_INSTRUCTIONS}
    assert fake.user_prompt().startswith(f"QUESTION:\n{WHY_Q}")


def test_full_dataset_is_never_sent():
    adapter, fake = groq_adapter()
    r = qr.route_query("Why do campaigns with spend over 100 perform differently?", explainer=adapter)
    prompt = fake.user_prompt()
    fc = next(a for a in r["analysis"] if a["tool"] == "filter_campaigns")
    assert fc["result"]["matched_count"] > 1000
    assert prompt.count('"id":') <= qr.MAX_LLM_CAMPAIGN_ROWS
    assert len(prompt.encode()) < 20_000
    assert len(prompt) < os.path.getsize(main.CSV_PATH) / 100

    provider = RecordingProvider()
    df_block = {"results": [{"tool": "x", "result": {"rows": main.dt.get_dataframe()}}]}
    out = LLMAdapter(provider).explain("Why?", df_block)
    assert out["status"] == "error" and out["error"] == "invalid_analysis" and provider.calls == []
    big = {"results": [{"tool": "x", "result": {"rows": ["x" * 100] * 1000}}]}
    out = LLMAdapter(provider).explain("Why?", big)
    assert out["error"] == "analysis_too_large" and provider.calls == []


# 4. Success -------------------------------------------------------------------

def test_success_through_endpoint():
    fake = fake_groq.install(main.llm_adapter, fake_groq.reply("What the data shows: TikTok has the highest ROAS."))
    try:
        r = client.post("/api/query", json={"query": WHY_Q})
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    body = r.json()
    assert r.status_code == 200 and body["route"] == "LLM_REQUIRED" and len(fake.requests) == 1
    assert body["explanation"].startswith("What the data shows")
    assert body["llm"]["status"] == "ok" and body["llm"]["provider"] == "groq"
    assert body["llm"]["analysis_bytes"] > 0 and body["analysis"]


# 5–8. Failures become structured results, never crashes ---------------------

def _llm_failure(respond, **kw):
    adapter, fake = groq_adapter(respond, **kw)
    r = qr.route_query(WHY_Q, explainer=adapter)
    assert r["route"] == "LLM_REQUIRED" and r["analysis"] and r["explanation"] is None
    assert r["llm"]["status"] == "error" and r["llm"]["message"]
    return r["llm"], fake


def test_provider_api_error():
    llm, fake = _llm_failure(lambda req: httpx.Response(400, json={"error": {"message": "Invalid argument"}}))
    assert llm["error"] == "provider_error" and len(fake.requests) == 1


def test_server_error_retries_then_fails_cleanly():
    llm, fake = _llm_failure(lambda req: httpx.Response(503, text="overloaded"))
    assert llm["error"] == "unavailable" and len(fake.requests) == 2  # bounded: one retry
    responses = iter([httpx.Response(503, text="overloaded"), httpx.Response(200, json=fake_groq.completion("recovered"))])
    adapter, fake = groq_adapter(lambda req: next(responses))
    r = qr.route_query(WHY_Q, explainer=adapter)
    assert r["explanation"] == "recovered" and len(fake.requests) == 2


def test_rate_limit():
    llm, fake = _llm_failure(lambda req: httpx.Response(429, json={"error": {"message": "Rate limit reached"}}))
    assert llm["error"] == "rate_limited" and len(fake.requests) == 1  # never retried


def test_timeout():
    def slow(req):
        raise httpx.ReadTimeout("timed out")
    llm, _ = _llm_failure(slow)
    assert llm["error"] == "timeout"
    # The per-request HTTP timeout is set from the adapter's deadline.
    adapter, fake = groq_adapter(timeout_s=12)
    qr.route_query(WHY_Q, explainer=adapter)
    assert 0 < fake.requests[0].extensions["timeout"]["read"] <= 12
    # A deadline too short to start an attempt never calls Groq at all.
    llm, fake = _llm_failure(fake_groq.reply("must not be used"), timeout_s=0.1)
    assert llm["error"] == "timeout" and len(fake.requests) == 0


def test_missing_api_key():
    llm, fake = _llm_failure(fake_groq.reply("must not be used"), api_key=None)
    assert llm["error"] == "not_configured" and len(fake.requests) == 0
    with patch.object(main.llm_adapter.provider, "_api_key", ""):
        body = client.post("/api/query", json={"query": WHY_Q}).json()
    assert body["llm"]["error"] == "not_configured" and body["analysis"]


def test_malformed_and_unexpected_provider_responses():
    llm, _ = _llm_failure(lambda req: httpx.Response(200, json={"choices": []}))
    assert llm["error"] == "bad_response"
    llm, _ = _llm_failure(fake_groq.reply("   "))
    assert llm["error"] == "bad_response"

    def bug(req):
        raise RuntimeError("boom")
    llm, _ = _llm_failure(bug)
    assert llm["error"] == "provider_error"
    broken_adapter = MagicMock()
    broken_adapter.explain.side_effect = RuntimeError("adapter bug")
    r = qr.route_query(WHY_Q, explainer=broken_adapter)
    assert r["llm"]["error"] == "adapter_error" and r["analysis"]


def test_empty_analysis_skips_llm():
    provider = RecordingProvider()
    r = qr.route_query("Why is revenue declining?", {"objective": "No Such Objective"},
                       explainer=LLMAdapter(provider))
    assert r["llm"]["status"] == "skipped" and r["llm"]["error"] == "empty_analysis"
    assert provider.calls == [] and r["explanation"] is None


def test_secrets_never_logged_or_returned():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    loggers = (qr.logger, logging.getLogger("marketingiq.llm_adapter"))
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
    try:
        llm, fake = _llm_failure(lambda req: httpx.Response(401, json={"error": {"message": f"Invalid API Key {SECRET_KEY}"}}),
                                 api_key=SECRET_KEY)
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(handler)
            lg.setLevel(level)
    assert llm["error"] == "auth_error" and fake.requests[0].headers["authorization"] == f"Bearer {SECRET_KEY}"
    assert records and not any("SECRET123" in r.getMessage() for r in records)
    assert "SECRET123" not in json.dumps(llm)


# 9 & 10. Routing: direct bypasses the LLM, LLM_REQUIRED uses it --------------

def test_direct_queries_do_not_invoke_llm():
    spy = MagicMock()
    for q in ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue."):
        r = qr.route_query(q, explainer=spy)
        assert r["route"] == "DIRECT_DATABASE" and "llm" not in r
    assert spy.explain.call_count == 0
    assert client.post("/api/query", json={"query": "What is total spend?"}).status_code == 200
    assert GUARD.requests == []


def test_llm_required_queries_invoke_adapter():
    spy = MagicMock()
    spy.explain.return_value = {"status": "ok", "provider": "spy", "text": "because"}
    for q in (WHY_Q, "Explain the decline in CTR.", "What might be causing the ROAS difference?"):
        r = qr.route_query(q, explainer=spy)
        assert r["route"] == "LLM_REQUIRED" and r["explanation"] == "because"
        question, llm_input = spy.explain.call_args.args
        assert question == q and llm_input["results"] == r["analysis"]
    assert spy.explain.call_count == 3


# 11. Concurrency cap -------------------------------------------------------------

def test_provider_slots_cap_waiting_requests_without_queueing():
    """Only `limit` requests wait on the provider at once; the rest get 'busy' at once (no queue, no
    provider call) and still receive the analysis. Slots are released after success and failure."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    release, inside = threading.Event(), threading.Semaphore(0)

    class Blocking(LLMProvider):
        name, model = "blocking", "none"

        def __init__(self):
            self.calls = 0

        def generate_explanation(self, question, analysis, timeout_s):
            self.calls += 1
            inside.release()
            release.wait(10)
            return "explained"

    provider = Blocking()
    adapter = LLMAdapter(provider, slots=ProviderSlots(2))
    with ThreadPoolExecutor(2) as pool:
        held = [pool.submit(adapter.explain, "Why?", SAMPLE) for _ in range(2)]
        assert inside.acquire(timeout=10) and inside.acquire(timeout=10)  # both slots in use
        busy = qr.route_query(WHY_Q, {}, explainer=adapter)
        assert busy["llm"]["status"] == "error" and busy["llm"]["error"] == "busy"
        assert busy["llm"]["message"] == BUSY_MESSAGE and busy["analysis"] and provider.calls == 2
        release.set()
        assert all(f.result()["status"] == "ok" for f in held)
    assert adapter.explain("Why?", SAMPLE)["status"] == "ok" and provider.calls == 3  # released

    class Failing(LLMProvider):
        name, model = "failing", "none"

        def generate_explanation(self, question, analysis, timeout_s):
            raise RuntimeError("boom")

    failing = LLMAdapter(Failing(), slots=ProviderSlots(1))
    assert failing.explain("Why?", SAMPLE)["status"] == "error"
    assert failing.explain("Why?", SAMPLE)["error"] == "provider_error"  # the slot came back


def test_chat_explanations_use_the_app_slots():
    """When every slot is taken, a chat explanation answers at once with the data and the busy
    notice, without calling Groq; database questions are unaffected."""
    held = [main.provider_slots._sem.acquire(blocking=False) for _ in range(main.provider_slots.limit)]
    try:
        assert all(held)
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
        direct = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is total revenue?"}],
                                                "filters": {}}).json()
    finally:
        for _ in held:
            main.provider_slots._sem.release()
    assert r["route"] == "LLM_REQUIRED" and "Too many AI explanations" in r["answer"]
    assert r["answer"].startswith("What the data shows:") and GUARD.requests == []
    assert direct["route"] == "DIRECT_DATABASE"
    assert main.provider_slots.limit == int(os.getenv("LLM_MAX_CONCURRENT", "20"))


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)  # keep per-request log lines out of the test output
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} LLM ADAPTER TESTS PASSED ===")
