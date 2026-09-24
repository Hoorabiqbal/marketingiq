"""
Tests for the LLM Adapter (llm_adapter.py), the Gemini provider (gemini_provider.py)
and their integration with the Query Router's LLM_REQUIRED route.

No real Gemini call is ever made: google-genai's Models.generate_content is patched
for the whole module to fail loudly, and each test overrides it with a mock built
from the real SDK's own response types (same approach as test_app.py).

Run:  python test_llm_adapter.py      (or: pytest test_llm_adapter.py)
"""
import json
import logging
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-not-real")

import httpx
from fastapi.testclient import TestClient
from google.genai import errors as ge
from google.genai import models as genai_models
from google.genai import types

import grounding
import main
import query_router as qr
from gemini_provider import GeminiProvider
from gemini_rotator import GeminiKeyRotator
from llm_adapter import GROUNDING_INSTRUCTIONS, LLMAdapter, LLMProvider

# Safety net: any Gemini call a test forgot to mock fails instead of spending quota.
_guard = patch.object(genai_models.Models, "generate_content",
                      side_effect=AssertionError("real Gemini call attempted in a test"))
_guard.start()

client = TestClient(main.app)
WHY_Q = "Why is TikTok performing better?"
SECRET_KEY = "fake-key-SECRET123"


def text_response(text):
    candidate = types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
    return types.GenerateContentResponse(candidates=[candidate])


def api_error(cls, code, msg):
    return cls(code=code, response_json={"error": {"message": msg}})


def mock_gemini(**kwargs):
    return patch.object(genai_models.Models, "generate_content", **kwargs)


def gemini_adapter(keys=(SECRET_KEY,), timeout_s=10.0):
    provider = GeminiProvider(GeminiKeyRotator(list(keys)), "test-model", sleep=lambda s: None)
    return LLMAdapter(provider, timeout_s=timeout_s)


def sent_prompt(mock_gen) -> str:
    return mock_gen.call_args.kwargs["contents"][0].parts[0].text


def sent_analysis(mock_gen) -> dict:
    return json.loads(sent_prompt(mock_gen).split("ANALYSIS (JSON):\n", 1)[1])


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


# 2 & 3. Gemini gets the compact analysis, never the dataset ---------------

def test_gemini_receives_compact_router_analysis():
    with mock_gemini(return_value=text_response("TikTok leads on ROAS.")) as gen:
        r = qr.route_query(WHY_Q, {"budget": "High"}, explainer=gemini_adapter())
    assert gen.call_count == 1  # one LLM call per request
    sent = sent_analysis(gen)
    expected = grounding.build_llm_context(WHY_Q, {"filters_applied": {"budget": "High"},
                                                   "focus_metrics": r["focus_metrics"], "results": r["analysis"]})
    assert sent == json.loads(json.dumps(expected))  # the router's analysis, typed — nothing else
    assert sent["active_filters"] == {"budget_tier": "High"} and sent["population"] == "dashboard-filtered view"
    config = gen.call_args.kwargs["config"]
    assert config.system_instruction == GROUNDING_INSTRUCTIONS
    assert sent_prompt(gen).startswith(f"QUESTION:\n{WHY_Q}")


def test_full_dataset_is_never_sent():
    with mock_gemini(return_value=text_response("ok")) as gen:
        r = qr.route_query("Why do campaigns with spend over 100 perform differently?", explainer=gemini_adapter())
    prompt = sent_prompt(gen)
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
    with mock_gemini(return_value=text_response("What the data shows: TikTok has the highest ROAS.")) as gen:
        r = client.post("/api/query", json={"query": WHY_Q})
    body = r.json()
    assert r.status_code == 200 and body["route"] == "LLM_REQUIRED" and gen.call_count == 1
    assert body["explanation"].startswith("What the data shows")
    assert body["llm"]["status"] == "ok" and body["llm"]["provider"] == "gemini"
    assert body["llm"]["analysis_bytes"] > 0 and body["analysis"]


# 5–8. Failures become structured results, never crashes ---------------------

def _llm_failure(side_effect, adapter=None):
    with mock_gemini(side_effect=side_effect) as gen:
        r = qr.route_query(WHY_Q, explainer=adapter or gemini_adapter())
    assert r["route"] == "LLM_REQUIRED" and r["analysis"] and r["explanation"] is None
    assert r["llm"]["status"] == "error" and r["llm"]["message"]
    return r["llm"], gen


def test_gemini_api_error():
    llm, _ = _llm_failure(api_error(ge.ClientError, 400, "Invalid argument"))
    assert llm["error"] == "provider_error"


def test_server_error_retries_then_fails_cleanly():
    err = api_error(ge.ServerError, 503, "overloaded")
    llm, gen = _llm_failure([err, err, err])
    assert llm["error"] == "unavailable" and gen.call_count == 3
    with mock_gemini(side_effect=[err, text_response("recovered")]) as gen:
        r = qr.route_query(WHY_Q, explainer=gemini_adapter())
    assert r["explanation"] == "recovered" and gen.call_count == 2


def test_rate_limit():
    quota = api_error(ge.ClientError, 429, "RESOURCE_EXHAUSTED: quota")
    llm, gen = _llm_failure(quota, gemini_adapter(keys=("k1", "k2")))
    assert llm["error"] == "rate_limited" and gen.call_count == 2  # existing rotator tried both keys


def test_timeout():
    llm, _ = _llm_failure(httpx.ReadTimeout("timed out"))
    assert llm["error"] == "timeout"
    # The per-request HTTP timeout is set from the adapter's deadline.
    with mock_gemini(return_value=text_response("ok")) as gen:
        qr.route_query(WHY_Q, explainer=gemini_adapter(timeout_s=12))
    assert 0 < gen.call_args.kwargs["config"].http_options.timeout <= 12_000
    # A deadline too short to start an attempt never calls Gemini at all.
    llm, gen = _llm_failure(AssertionError("must not be called"), gemini_adapter(timeout_s=0.1))
    assert llm["error"] == "timeout" and gen.call_count == 0


def test_missing_api_key():
    llm, gen = _llm_failure(AssertionError("must not be called"), gemini_adapter(keys=()))
    assert llm["error"] == "not_configured" and gen.call_count == 0
    with patch.object(main.rotator, "clients", []):
        body = client.post("/api/query", json={"query": WHY_Q}).json()
    assert body["llm"]["error"] == "not_configured" and body["analysis"]


def test_malformed_and_unexpected_provider_responses():
    empty = types.GenerateContentResponse(candidates=[])
    with mock_gemini(return_value=empty):
        r = qr.route_query(WHY_Q, explainer=gemini_adapter())
    assert r["llm"]["error"] == "bad_response"
    with mock_gemini(return_value=text_response("   ")):
        assert qr.route_query(WHY_Q, explainer=gemini_adapter())["llm"]["error"] == "bad_response"
    llm, _ = _llm_failure(RuntimeError("boom"))
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
        llm, _ = _llm_failure(api_error(ge.ClientError, 401, f"API key not valid: {SECRET_KEY}"))
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(handler)
            lg.setLevel(level)
    assert llm["error"] == "auth_error"
    assert records and not any("SECRET123" in r.getMessage() for r in records)
    assert "SECRET123" not in json.dumps(llm)


# 9 & 10. Routing: direct bypasses the LLM, LLM_REQUIRED uses it --------------

def test_direct_queries_do_not_invoke_llm():
    spy = MagicMock()
    for q in ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue."):
        r = qr.route_query(q, explainer=spy)
        assert r["route"] == "DIRECT_DATABASE" and "llm" not in r
    assert spy.explain.call_count == 0
    with mock_gemini(side_effect=AssertionError("LLM must not be called")) as gen:
        assert client.post("/api/query", json={"query": "What is total spend?"}).status_code == 200
    assert gen.call_count == 0


def test_llm_required_queries_invoke_adapter():
    spy = MagicMock()
    spy.explain.return_value = {"status": "ok", "provider": "spy", "text": "because"}
    for q in (WHY_Q, "Explain the decline in CTR.", "What might be causing the ROAS difference?"):
        r = qr.route_query(q, explainer=spy)
        assert r["route"] == "LLM_REQUIRED" and r["explanation"] == "because"
        question, llm_input = spy.explain.call_args.args
        assert question == q and llm_input["results"] == r["analysis"]
    assert spy.explain.call_count == 3


def test_provider_slots_cap_waiting_requests_without_queueing():
    """Only `limit` requests wait on the provider at once; the rest get 'busy' at once (no queue, no
    provider call) and still receive the analysis. Slots are released after success and failure."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from llm_adapter import BUSY_MESSAGE, ProviderSlots

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


def test_chat_fallback_shares_the_provider_slots():
    from llm_adapter import BUSY_MESSAGE
    history = [{"role": "user", "content": "What is total revenue?"}, {"role": "assistant", "content": "It is $1."}]
    body = {"messages": [*history, {"role": "user", "content": "Why is that?"}], "filters": {}}
    held = [main.provider_slots._sem.acquire(blocking=False) for _ in range(main.provider_slots.limit)]
    try:
        assert all(held)
        with patch.object(genai_models.Models, "generate_content",
                          side_effect=AssertionError("Gemini must not be called while busy")):
            r = client.post("/api/chat", json=body).json()
    finally:
        for _ in held:
            main.provider_slots._sem.release()
    assert r["route"] == "FALLBACK" and "Too many AI explanations" in r["answer"]
    assert main.provider_slots.limit == int(os.getenv("LLM_MAX_CONCURRENT", "20"))


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)  # keep per-request log lines out of the test output
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} LLM ADAPTER TESTS PASSED ===")
