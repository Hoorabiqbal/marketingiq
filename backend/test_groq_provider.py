"""
Tests for the Groq provider (groq_provider.py), MarketingIQ's only LLM provider, and its
configuration (llm_providers.py). The Groq API is replaced by httpx.MockTransport (fake_groq.py),
so these tests never need a real GROQ_API_KEY and use no quota.

Run:  python test_groq_provider.py      (or: pytest test_groq_provider.py)
"""
import importlib
import json
import logging
import os
from unittest.mock import patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)
from fake_groq import FAKE_KEY, completion

import httpx
from fastapi.testclient import TestClient

import grounding
import llm_providers
import main
import query_router as qr
from groq_provider import DEFAULT_GROQ_MODEL, MAX_ATTEMPTS, GroqProvider
from llm_adapter import DATA_STILL_AVAILABLE, GROUNDING_INSTRUCTIONS, LLMAdapter, build_user_prompt

WHY_Q = "Why is TikTok performing better?"
OK_TEXT = "What the data shows: TikTok ROAS 11.18x.\nInterpretation: may be X."
OK = fake_groq.reply(OK_TEXT)


def explain(respond, question=WHY_Q, filters=None, timeout_s=30, **kw):
    provider, fake = fake_groq.provider(respond, **kw)
    r = qr.route_query(question, filters, explainer=LLMAdapter(provider, timeout_s=timeout_s))
    return r, fake


def _failure(respond, **kw):
    r, fake = explain(respond, **kw)
    assert r["route"] == "LLM_REQUIRED" and r["analysis"] and r["explanation"] is None
    assert r["llm"]["status"] == "error" and r["llm"]["provider"] == "groq"
    assert DATA_STILL_AVAILABLE in r["llm"]["message"]
    for word in ("groq", "gemini", "api key", "api_key", "gsk_"):  # users never see vendor/config details
        assert word not in r["llm"]["message"].lower(), r["llm"]["message"]
    return r["llm"], fake


# 1. Successful response ------------------------------------------------------------

def test_successful_response():
    r, fake = explain(OK)
    assert r["llm"]["status"] == "ok" and r["llm"]["provider"] == "groq"
    assert r["explanation"] == OK_TEXT and len(fake.requests) == 1


# 2. Correct model and request shape ------------------------------------------------

def test_correct_model_and_request_shape():
    provider, fake = fake_groq.provider()
    LLMAdapter(provider).explain("Why?", {"results": [{"tool": "t", "result": {"roas": 6.54}}]})
    req = fake.requests[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = fake.body()
    assert body["model"] == "openai/gpt-oss-120b" == DEFAULT_GROQ_MODEL
    assert body["temperature"] == 0.2 and body["stream"] is False
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert body["messages"][0] == {"role": "system", "content": GROUNDING_INSTRUCTIONS}
    # A non-reasoning model override gets no reasoning parameters (Groq would reject them).
    provider, fake = fake_groq.provider(model="llama-3.3-70b-versatile")
    LLMAdapter(provider).explain("Why?", {"results": [{"tool": "t", "result": {"roas": 6.54}}]})
    assert fake.body()["model"] == "llama-3.3-70b-versatile"
    assert "reasoning_effort" not in fake.body() and "include_reasoning" not in fake.body()


# 3. The user prompt is exactly the typed compact analysis --------------------------------

def test_user_prompt_is_the_typed_compact_analysis():
    r, fake = explain(OK, filters={"budget": "High"})
    expected = build_user_prompt(WHY_Q, grounding.build_llm_context(
        WHY_Q, {"filters_applied": {"budget": "High"}, "focus_metrics": r["focus_metrics"], "results": r["analysis"]}))
    assert fake.user_prompt() == expected
    assert '"metric_units"' in expected and '"analysis_scope"' in expected


# 4. No full dataset ------------------------------------------------------------------

def test_full_dataset_not_passed():
    r, fake = explain(OK, question="Why do campaigns with spend over 100 perform differently?")
    fc = next(a for a in r["analysis"] if a["tool"] == "filter_campaigns")
    assert fc["result"]["matched_count"] > 1000
    assert len(fake.requests[0].content) < 20_000
    assert 1 <= fake.user_prompt().count('"id":') <= qr.MAX_LLM_CAMPAIGN_ROWS


# 5. Missing API key ------------------------------------------------------------------

def test_missing_api_key():
    for key in (None, "", "   "):
        llm, fake = _failure(OK, api_key=key)
        assert llm["error"] == "not_configured" and len(fake.requests) == 0
    p = llm_providers.build_provider(env={})
    assert isinstance(p, GroqProvider) and not p.configured
    assert FAKE_KEY not in repr(fake_groq.provider()[0])


# 6. Timeout --------------------------------------------------------------------------

def test_timeout():
    def slow(req):
        raise httpx.ReadTimeout("timed out")
    llm, fake = _failure(slow, timeout_s=17)
    assert llm["error"] == "timeout" and len(fake.requests) == 1  # a timeout is not retried
    timeouts = fake.requests[0].extensions["timeout"]
    assert 16 < timeouts["read"] <= 17 and timeouts["connect"] == 5.0


# 7. Rate limit -----------------------------------------------------------------------

def test_rate_limit_response_not_retried():
    headers = {"retry-after": "7", "x-ratelimit-remaining-tokens": "0"}
    with patch("groq_provider.logger") as log:
        llm, fake = _failure(lambda req: httpx.Response(
            429, headers=headers, json={"error": {"message": "Rate limit reached", "type": "tokens"}}))
    assert llm["error"] == "rate_limited" and len(fake.requests) == 1
    assert llm["message"].startswith("The AI explanation service is temporarily rate-limited.")
    logged = json.loads(log.warning.call_args.args[0])
    assert logged["event"] == "groq_rate_limited" and logged["retry-after"] == "7"


# 8. Server errors --------------------------------------------------------------------

def test_server_error_bounded_retry():
    llm, fake = _failure(lambda req: httpx.Response(503, json={"error": {"message": "over capacity"}}))
    assert llm["error"] == "unavailable" and len(fake.requests) == MAX_ATTEMPTS == 2
    responses = iter([httpx.Response(500, text="boom"), httpx.Response(200, json=completion(OK_TEXT))])
    r, fake = explain(lambda req: next(responses))
    assert r["llm"]["status"] == "ok" and len(fake.requests) == 2

    def refused(req):
        raise httpx.ConnectError("connection refused")
    llm, fake = _failure(refused)
    assert llm["error"] == "unavailable" and len(fake.requests) == 2


def test_auth_error_not_logged_or_retried():
    with patch("groq_provider.logger") as log:
        llm, fake = _failure(lambda req: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    assert llm["error"] == "auth_error" and len(fake.requests) == 1 and not log.warning.called


# 9. Malformed response / unknown model ------------------------------------------------

def test_malformed_response():
    bad = [httpx.Response(200, text="<html>not json</html>"),
           httpx.Response(200, json={"choices": []}),
           httpx.Response(200, json={"choices": [{"message": {}}]}),
           httpx.Response(200, json=["not", "a", "dict"]),
           httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]}),
           httpx.Response(200, json={"choices": [{"message": {"content": None}}]})]
    for resp in bad:
        assert _failure(lambda req, resp=resp: resp)[0]["error"] == "bad_response"
    with patch("groq_provider.logger") as log:
        llm, _ = _failure(lambda req: httpx.Response(404, json={"error": {"message": "model not found"}}))
    assert llm["error"] == "not_configured" and "GROQ_MODEL" not in llm["message"]  # config stays server-side
    assert json.loads(log.warning.call_args.args[0]) == {"event": "groq_model_not_found", "model": DEFAULT_GROQ_MODEL}


# 10. Configuration ---------------------------------------------------------------------

def test_configuration():
    p = llm_providers.build_provider(env={"GROQ_API_KEY": FAKE_KEY, "GROQ_MODEL": "llama-3.3-70b-versatile"})
    assert isinstance(p, GroqProvider) and p.configured and p.model == "llama-3.3-70b-versatile"
    assert llm_providers.build_provider(env={"GROQ_API_KEY": FAKE_KEY}).model == DEFAULT_GROQ_MODEL
    assert llm_providers.timeout_seconds({}) == 25.0
    assert llm_providers.timeout_seconds({"LLM_TIMEOUT_SECONDS": "10"}) == 10.0
    # A leftover LLM_PROVIDER never switches providers; a non-groq value is logged as ignored.
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    lg = logging.getLogger("marketingiq.llm_adapter")
    level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.WARNING)
    try:
        for value in ("gemini", "qwen", "groq", ""):
            assert llm_providers.build_provider(env={"LLM_PROVIDER": value, "GROQ_API_KEY": FAKE_KEY}).name == "groq"
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    ignored = [json.loads(r.getMessage())["value"] for r in records if "llm_provider_ignored" in r.getMessage()]
    assert ignored == ["gemini", "qwen"]
    assert main.llm_adapter.provider.name == "groq" and main.llm_adapter.provider.model == DEFAULT_GROQ_MODEL


def _reload_main(**env):
    """Reload main.py with these environment variables (None removes one). GROQ_API_KEY=""
    means "no key" even when a real one is in .env (load_dotenv never overrides)."""
    for var, value in env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    m = importlib.reload(main)
    fake_groq.block_real_calls(m.llm_adapter)  # the reloaded provider must not reach the network either
    return m


def test_app_uses_groq_only():
    """DIRECT_DATABASE questions make 0 Groq calls; an explanation makes exactly one; a Groq
    failure is reported, never retried through another path."""
    client = TestClient(main.app)
    health = client.get("/api/health").json()
    assert health["llm_provider"] == "groq" and health["llm_model"] == DEFAULT_GROQ_MODEL
    assert health["llm_configured"] is True and FAKE_KEY not in json.dumps(health)
    fake = fake_groq.install(main.llm_adapter, OK)
    for q in ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue.",
              "What is the average CTR?", "Compare TikTok and LinkedIn ROAS."):
        body = client.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "filters": {}}).json()
        assert body["route"] == "DIRECT_DATABASE", (q, body)
        r = client.post("/api/query", json={"query": q}).json()
        assert r["route"] == "DIRECT_DATABASE" and "llm" not in r
    assert len(fake.requests) == 0

    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
    assert body["route"] == "LLM_REQUIRED" and body["answer"].startswith("What the data shows")
    assert len(fake.requests) == 1

    fake = fake_groq.install(main.llm_adapter, lambda req: httpx.Response(429, json={"error": {"message": "limit"}}))
    body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
    assert body["route"] == "LLM_REQUIRED" and "temporarily rate-limited" in body["answer"]
    assert "What the data shows" in body["answer"]  # the figures are still shown
    assert "key" not in body["answer"].lower() and len(fake.requests) == 1


def test_leftover_llm_provider_is_ignored_at_startup():
    m = _reload_main(LLM_PROVIDER="gemini")
    try:
        assert m.llm_adapter.provider.name == "groq"
        assert TestClient(m.app).get("/api/health").json()["llm_provider"] == "groq"
    finally:
        _reload_main(LLM_PROVIDER=None)


def test_app_starts_without_a_key():
    m = _reload_main(GROQ_API_KEY="")
    try:
        client = TestClient(m.app)
        health = client.get("/api/health").json()
        assert health["llm_provider"] == "groq" and health["llm_configured"] is False
        body = client.post("/api/query", json={"query": WHY_Q}).json()
        assert body["route"] == "LLM_REQUIRED" and body["analysis"]
        assert body["llm"]["error"] == "not_configured"
        chat = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
        assert "unavailable right now" in chat["answer"] and "What the data shows" in chat["answer"]
        assert "key" not in chat["answer"].lower()
        assert client.post("/api/query", json={"query": "What is total revenue?"}).json()["route"] == "DIRECT_DATABASE"
    finally:
        _reload_main(GROQ_API_KEY=FAKE_KEY)


if __name__ == "__main__":
    for name in ("marketingiq.query_router", "marketingiq.llm_adapter", "marketingiq.data"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} GROQ PROVIDER TESTS PASSED ===")
