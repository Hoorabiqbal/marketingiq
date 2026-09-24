"""
Tests for the Groq provider (groq_provider.py) and its selection through llm_providers.py.
The Groq API is replaced by httpx.MockTransport, so these tests never need a GROQ_API_KEY and
use no quota; Gemini and Ollama are patched/guarded to fail loudly if anything calls them.

Run:  python test_groq_provider.py      (or: pytest test_groq_provider.py)
"""
import importlib
import json
import os
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-not-real")
os.environ.pop("LLM_PROVIDER", None)

import httpx
from fastapi.testclient import TestClient
from google.genai import models as genai_models
from google.genai import types

import llm_providers
import main
import query_router as qr
from gemini_provider import GeminiProvider
from gemini_rotator import GeminiKeyRotator
from groq_provider import DEFAULT_GROQ_MODEL, MAX_ATTEMPTS, GroqProvider
from llm_adapter import GROUNDING_INSTRUCTIONS, LLMAdapter, build_user_prompt

_guard = patch.object(genai_models.Models, "generate_content",
                      side_effect=AssertionError("Gemini must not be called in Groq tests"))
_guard.start()

FAKE_KEY = "gsk_test_placeholder_not_real"
WHY_Q = "Why is TikTok performing better?"
OK_BODY = {"id": "chatcmpl-1", "object": "chat.completion", "model": DEFAULT_GROQ_MODEL,
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant",
                                    "content": "What the data shows: TikTok ROAS 11.18x.\nInterpretation: may be X."}}],
           "usage": {"prompt_tokens": 900, "completion_tokens": 120, "queue_time": 0.01, "total_time": 0.3}}


class FakeGroq:
    """httpx transport standing in for api.groq.com; records every request."""

    def __init__(self, respond):
        self.respond, self.requests = respond, []

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


def groq(respond=lambda r: httpx.Response(200, json=OK_BODY), api_key=FAKE_KEY, model=DEFAULT_GROQ_MODEL):
    fake = FakeGroq(respond)
    provider = GroqProvider(api_key=api_key, model=model, sleep=lambda s: None,
                            http_client=httpx.Client(transport=httpx.MockTransport(fake)))
    return provider, fake


def explain(respond, question=WHY_Q, filters=None, timeout_s=30, **kw):
    provider, fake = groq(respond, **kw)
    r = qr.route_query(question, filters, explainer=LLMAdapter(provider, timeout_s=timeout_s))
    return r, fake


def _failure(respond, **kw):
    r, fake = explain(respond, **kw)
    assert r["route"] == "LLM_REQUIRED" and r["analysis"] and r["explanation"] is None
    assert r["llm"]["status"] == "error" and r["llm"]["provider"] == "groq"
    return r["llm"], fake


# 1. Successful response ------------------------------------------------------------

def test_successful_response():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY))
    assert r["llm"]["status"] == "ok" and r["llm"]["provider"] == "groq"
    assert r["explanation"] == OK_BODY["choices"][0]["message"]["content"].strip()
    assert len(fake.requests) == 1


# 2. Correct model and request shape ------------------------------------------------

def test_correct_model_and_request_shape():
    provider, fake = groq()
    LLMAdapter(provider).explain("Why?", {"results": [{"tool": "t", "result": {"roas": 6.54}}]})
    req = fake.requests[0]
    assert str(req.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = fake.body()
    assert body["model"] == "openai/gpt-oss-20b" == DEFAULT_GROQ_MODEL
    assert body["temperature"] == 0.2 and body["stream"] is False
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert body["messages"][0] == {"role": "system", "content": GROUNDING_INSTRUCTIONS}  # same as Gemini
    # A non-reasoning model override gets no reasoning parameters (Groq would reject them).
    provider, fake = groq(model="llama-3.3-70b-versatile")
    LLMAdapter(provider).explain("Why?", {"results": [{"tool": "t", "result": {"roas": 6.54}}]})
    assert fake.body()["model"] == "llama-3.3-70b-versatile"
    assert "reasoning_effort" not in fake.body() and "include_reasoning" not in fake.body()


# 3. Compact analysis input, identical to Gemini's ---------------------------------------

def test_compact_analysis_identical_to_gemini_input():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY), filters={"budget": "High"})
    groq_user_prompt = fake.body()["messages"][1]["content"]
    expected = build_user_prompt(WHY_Q, {"filters_applied": {"budget": "High"},
                                         "focus_metrics": r["focus_metrics"], "results": r["analysis"]})
    assert groq_user_prompt == expected
    candidate = types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]))
    with patch.object(genai_models.Models, "generate_content",
                      return_value=types.GenerateContentResponse(candidates=[candidate])) as gen:
        qr.route_query(WHY_Q, {"budget": "High"},
                       explainer=LLMAdapter(GeminiProvider(GeminiKeyRotator(["k"]), "m")))
    assert gen.call_args.kwargs["contents"][0].parts[0].text == groq_user_prompt
    assert gen.call_args.kwargs["config"].system_instruction == fake.body()["messages"][0]["content"]


# 4. No full dataset ------------------------------------------------------------------

def test_full_dataset_not_passed():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY),
                      question="Why do campaigns with spend over 100 perform differently?")
    fc = next(a for a in r["analysis"] if a["tool"] == "filter_campaigns")
    assert fc["result"]["matched_count"] > 1000
    assert len(fake.requests[0].content) < 20_000
    campaign_rows = fake.body()["messages"][1]["content"].count('"id":')
    assert 1 <= campaign_rows <= qr.MAX_LLM_CAMPAIGN_ROWS


# 5. Missing API key ------------------------------------------------------------------

def test_missing_api_key():
    for key in (None, "", "   "):
        llm, fake = _failure(lambda req: httpx.Response(200, json=OK_BODY), api_key=key)
        assert llm["error"] == "not_configured" and len(fake.requests) == 0
    p = llm_providers.build_provider("groq", env={})
    assert isinstance(p, GroqProvider) and not p.configured
    assert FAKE_KEY not in repr(groq()[0])


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
    logged = json.loads(log.warning.call_args.args[0])
    assert logged["event"] == "groq_rate_limited" and logged["retry-after"] == "7"


# 8. Server errors --------------------------------------------------------------------

def test_server_error_bounded_retry():
    llm, fake = _failure(lambda req: httpx.Response(503, json={"error": {"message": "over capacity"}}))
    assert llm["error"] == "unavailable" and len(fake.requests) == MAX_ATTEMPTS == 2
    responses = iter([httpx.Response(500, text="boom"), httpx.Response(200, json=OK_BODY)])
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


# 9. Malformed response ---------------------------------------------------------------

def test_malformed_response():
    bad = [httpx.Response(200, text="<html>not json</html>"),
           httpx.Response(200, json={"choices": []}),
           httpx.Response(200, json={"choices": [{"message": {}}]}),
           httpx.Response(200, json=["not", "a", "dict"]),
           httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]}),
           httpx.Response(200, json={"choices": [{"message": {"content": None}}]})]
    for resp in bad:
        assert _failure(lambda req, resp=resp: resp)[0]["error"] == "bad_response"
    llm, _ = _failure(lambda req: httpx.Response(404, json={"error": {"message": "model not found"}}))
    assert llm["error"] == "not_configured" and "GROQ_MODEL" in llm["message"]


# 10. Provider selection --------------------------------------------------------------

def test_provider_selection_and_configuration():
    assert llm_providers.selected_provider({}) == "gemini"  # Gemini stays the default
    assert llm_providers.selected_provider({"LLM_PROVIDER": " GROQ "}) == "groq"
    assert llm_providers.selected_provider({"LLM_PROVIDER": "qwen"}) == "qwen"
    p = llm_providers.build_provider("groq", env={"GROQ_API_KEY": FAKE_KEY, "GROQ_MODEL": "llama-3.3-70b-versatile"})
    assert isinstance(p, GroqProvider) and p.configured and p.model == "llama-3.3-70b-versatile"
    assert llm_providers.build_provider("groq", env={"GROQ_API_KEY": FAKE_KEY}).model == DEFAULT_GROQ_MODEL
    assert llm_providers.timeout_seconds("groq", {}) == 25.0
    assert llm_providers.timeout_seconds("groq", {"LLM_TIMEOUT_SECONDS": "10"}) == 10.0
    assert main.llm_adapter.provider.name == "gemini"


def _reload_main(provider, key=None):
    """key="" means "no key" even if a real one is in .env (load_dotenv never overrides)."""
    for var, value in (("LLM_PROVIDER", provider), ("GROQ_API_KEY", key)):
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    return importlib.reload(main)


def test_app_with_groq_selected():
    """With LLM_PROVIDER=groq: DIRECT_DATABASE questions make 0 Groq, 0 Gemini and 0 Ollama calls;
    an explanation makes exactly one Groq call; a Groq failure never fails over to Gemini."""
    m = _reload_main("groq", FAKE_KEY)
    try:
        client = TestClient(m.app)
        health = client.get("/api/health").json()
        assert health["llm_provider"] == "groq" and health["llm_model"] == DEFAULT_GROQ_MODEL
        assert FAKE_KEY not in json.dumps(health)
        fake = FakeGroq(lambda req: httpx.Response(200, json=OK_BODY))
        m.llm_adapter.provider._client = httpx.Client(transport=httpx.MockTransport(fake))
        ollama_calls = []
        with patch.object(genai_models.Models, "generate_content",
                          side_effect=AssertionError("Gemini must not be called")) as gen, \
             patch("qwen_provider.QwenProvider.generate_explanation", side_effect=ollama_calls.append):
            direct = ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue.",
                      "What is the average CTR?", "Compare TikTok and LinkedIn ROAS.")
            for q in direct:
                body = client.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "filters": {}}).json()
                assert body["route"] == "DIRECT_DATABASE", (q, body)
                r = client.post("/api/query", json={"query": q}).json()
                assert r["route"] == "DIRECT_DATABASE" and "llm" not in r
            assert len(fake.requests) == 0 and ollama_calls == []

            body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
            assert body["route"] == "LLM_REQUIRED" and body["answer"].startswith("What the data shows")
            assert len(fake.requests) == 1

            m.llm_adapter.provider._client = httpx.Client(transport=httpx.MockTransport(
                lambda req: httpx.Response(429, json={"error": {"message": "rate limited"}})))
            body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
            assert body["route"] == "LLM_REQUIRED" and "usage limit" in body["answer"]
            assert "GEMINI_API_KEY" not in body["answer"]
        assert gen.call_count == 0 and ollama_calls == []  # no silent failover
    finally:
        _reload_main(None)
    assert main.llm_adapter.provider.name == "gemini"


def test_app_starts_with_groq_but_no_key():
    m = _reload_main("groq", "")
    try:
        client = TestClient(m.app)
        body = client.post("/api/query", json={"query": WHY_Q}).json()
        assert body["route"] == "LLM_REQUIRED" and body["analysis"]
        assert body["llm"]["error"] == "not_configured"
        assert client.post("/api/query", json={"query": "What is total revenue?"}).json()["route"] == "DIRECT_DATABASE"
    finally:
        _reload_main(None)


if __name__ == "__main__":
    import logging
    for name in ("marketingiq.query_router", "marketingiq.llm_adapter", "marketingiq.data"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} GROQ PROVIDER TESTS PASSED ===")
