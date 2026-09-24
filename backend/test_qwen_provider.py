"""
Tests for the Qwen/Ollama provider (qwen_provider.py) and explicit provider selection
(llm_providers.py). Ollama is replaced by httpx.MockTransport, so these tests never need
Ollama running; Gemini is patched to fail loudly if anything calls it.

Run:  python test_qwen_provider.py      (or: pytest test_qwen_provider.py)
"""
import importlib
import json
import os
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-placeholder-not-real")
os.environ.pop("LLM_PROVIDER", None)

import httpx
from fastapi.testclient import TestClient
from google.genai import errors as ge
from google.genai import models as genai_models
from google.genai import types

import llm_providers
import main
import query_router as qr
from gemini_provider import GeminiProvider
from gemini_rotator import GeminiKeyRotator
from llm_adapter import LLMAdapter, build_user_prompt
from qwen_provider import DEFAULT_OLLAMA_MODEL, QWEN_SYSTEM_PROMPT, QwenProvider, normalize_host

_guard = patch.object(genai_models.Models, "generate_content",
                      side_effect=AssertionError("Gemini must not be called in Qwen tests"))
_guard.start()

WHY_Q = "Why is TikTok performing better?"
OK_BODY = {"model": DEFAULT_OLLAMA_MODEL, "done": True, "done_reason": "stop",
           "message": {"role": "assistant", "content": "What the data shows: TikTok ROAS 11.18x.\nInterpretation: may be X."},
           "total_duration": 2_500_000_000, "load_duration": 400_000_000, "prompt_eval_count": 700,
           "prompt_eval_duration": 900_000_000, "eval_count": 60, "eval_duration": 1_200_000_000}


class FakeOllama:
    """httpx transport standing in for Ollama; records every request."""

    def __init__(self, respond):
        self.respond, self.requests = respond, []

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


def qwen(respond=lambda r: httpx.Response(200, json=OK_BODY), host="http://127.0.0.1:11434", model=DEFAULT_OLLAMA_MODEL):
    fake = FakeOllama(respond)
    provider = QwenProvider(host=host, model=model, http_client=httpx.Client(transport=httpx.MockTransport(fake)))
    return provider, fake


def explain(respond, question=WHY_Q, filters=None, timeout_s=30):
    provider, fake = qwen(respond)
    r = qr.route_query(question, filters, explainer=LLMAdapter(provider, timeout_s=timeout_s))
    return r, fake


# 1-3. Success, model, compact analysis ----------------------------------------------

def test_successful_response():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY))
    assert r["llm"]["status"] == "ok" and r["llm"]["provider"] == "qwen"
    assert r["explanation"] == OK_BODY["message"]["content"].strip()
    assert len(fake.requests) == 1


def test_correct_model_host_and_request_shape():
    provider, fake = qwen(host="localhost", model="qwen2.5:1.5b-instruct")
    LLMAdapter(provider).explain("Why?", {"results": [{"tool": "t", "result": {"roas": 6.54}}]})
    req = fake.requests[0]
    assert str(req.url) == "http://localhost:11434/api/chat"
    body = fake.body()
    assert body["model"] == "qwen2.5:1.5b-instruct" and body["stream"] is False
    assert body["options"]["temperature"] == 0.2 and body["options"]["num_predict"] == 300
    assert body["messages"][0] == {"role": "system", "content": QWEN_SYSTEM_PROMPT}


def test_compact_analysis_identical_to_gemini_input():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY), filters={"budget": "High"})
    qwen_user_prompt = fake.body()["messages"][1]["content"]
    expected = build_user_prompt(WHY_Q, {"filters_applied": {"budget": "High"},
                                         "focus_metrics": r["focus_metrics"], "results": r["analysis"]})
    assert qwen_user_prompt == expected
    # Gemini gets exactly the same user content for the same question.
    candidate = types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]))
    with patch.object(genai_models.Models, "generate_content",
                      return_value=types.GenerateContentResponse(candidates=[candidate])) as gen:
        qr.route_query(WHY_Q, {"budget": "High"},
                       explainer=LLMAdapter(GeminiProvider(GeminiKeyRotator(["k"]), "m")))
    assert gen.call_args.kwargs["contents"][0].parts[0].text == qwen_user_prompt


def test_full_dataset_not_passed():
    r, fake = explain(lambda req: httpx.Response(200, json=OK_BODY),
                      question="Why do campaigns with spend over 100 perform differently?")
    fc = next(a for a in r["analysis"] if a["tool"] == "filter_campaigns")
    assert fc["result"]["matched_count"] > 1000
    request_bytes = fake.requests[0].content
    assert len(request_bytes) < 20_000
    campaign_rows = fake.body()["messages"][1]["content"].count('"id":')
    assert 1 <= campaign_rows <= qr.MAX_LLM_CAMPAIGN_ROWS  # some rows are sent, never more than the cap


# 5-8. Failures become structured results --------------------------------------------

def _failure(respond, **kw):
    r, fake = explain(respond, **kw)
    assert r["route"] == "LLM_REQUIRED" and r["analysis"] and r["explanation"] is None
    assert r["llm"]["status"] == "error"
    return r["llm"], fake


def test_connection_failure():
    def refused(req):
        raise httpx.ConnectError("connection refused")
    llm, fake = _failure(refused)
    assert llm["error"] == "unavailable" and "Ollama" in llm["message"] and len(fake.requests) == 1


def test_timeout_and_its_configuration():
    def slow(req):
        raise httpx.ReadTimeout("timed out")
    llm, fake = _failure(slow, timeout_s=42)
    assert llm["error"] == "timeout" and len(fake.requests) == 1  # no retry
    timeouts = fake.requests[0].extensions["timeout"]
    assert timeouts["read"] == 42 and timeouts["connect"] == 3.0


def test_model_unavailable():
    llm, _ = _failure(lambda req: httpx.Response(
        404, json={"error": f'model "{DEFAULT_OLLAMA_MODEL}" not found, try pulling it first'}))
    assert llm["error"] == "not_configured" and f"ollama pull {DEFAULT_OLLAMA_MODEL}" in llm["message"]


def test_malformed_and_server_errors():
    assert _failure(lambda req: httpx.Response(200, text="<html>not json</html>"))[0]["error"] == "bad_response"
    assert _failure(lambda req: httpx.Response(200, json={"message": {}}))[0]["error"] == "bad_response"
    assert _failure(lambda req: httpx.Response(200, json={"message": {"content": "  "}}))[0]["error"] == "bad_response"
    assert _failure(lambda req: httpx.Response(200, json=["not", "a", "dict"]))[0]["error"] == "bad_response"
    llm, _ = _failure(lambda req: httpx.Response(500, json={"error": "model requires more system memory"}))
    assert llm["error"] == "unavailable" and "memory" in llm["message"]


# 9-10. Configuration and explicit selection -----------------------------------------

def test_provider_configuration():
    assert normalize_host("") == "http://127.0.0.1:11434"
    assert normalize_host("localhost") == "http://localhost:11434"
    assert normalize_host("127.0.0.1:12345") == "http://127.0.0.1:12345"
    assert normalize_host("https://ollama.example.com/") == "https://ollama.example.com"
    p = llm_providers.build_provider("qwen", env={"OLLAMA_HOST": "10.0.0.5:11434", "OLLAMA_MODEL": "qwen2.5:1.5b-instruct"})
    assert isinstance(p, QwenProvider) and p.host == "http://10.0.0.5:11434" and p.model == "qwen2.5:1.5b-instruct"
    assert llm_providers.build_provider("qwen", env={}).model == DEFAULT_OLLAMA_MODEL
    assert llm_providers.timeout_seconds("qwen", {}) == 90.0 and llm_providers.timeout_seconds("gemini", {}) == 25.0
    assert llm_providers.timeout_seconds("qwen", {"LLM_TIMEOUT_SECONDS": "12"}) == 12.0


def test_provider_selection():
    assert llm_providers.selected_provider({}) == "gemini"  # backward-compatible default
    assert llm_providers.selected_provider({"LLM_PROVIDER": " Qwen "}) == "qwen"
    assert llm_providers.selected_provider({"LLM_PROVIDER": "gemini"}) == "gemini"
    try:
        llm_providers.selected_provider({"LLM_PROVIDER": "gpt"})
        raise AssertionError("unknown provider accepted")
    except llm_providers.ProviderConfigError:
        pass
    assert main.llm_adapter.provider.name == "gemini"


def _reload_main(provider):
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    else:
        os.environ.pop("LLM_PROVIDER", None)
    return importlib.reload(main)


def test_app_with_qwen_selected():
    """With LLM_PROVIDER=qwen: direct chat never touches an LLM, explanations make exactly one
    Ollama call and zero Gemini calls, and a Qwen failure never fails over to Gemini."""
    m = _reload_main("qwen")
    try:
        client = TestClient(m.app)
        assert client.get("/api/health").json()["llm_provider"] == "qwen"
        responses = iter([httpx.Response(200, json=OK_BODY)])
        fake = FakeOllama(lambda req: next(responses))
        m.llm_adapter.provider._client = httpx.Client(transport=httpx.MockTransport(fake))
        with patch.object(genai_models.Models, "generate_content",
                          side_effect=AssertionError("Gemini must not be called")) as gen:
            for q in ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue."):
                body = client.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "filters": {}}).json()
                assert body["route"] == "DIRECT_DATABASE"
            assert len(fake.requests) == 0
            body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
            assert body["route"] == "LLM_REQUIRED" and body["answer"].startswith("What the data shows")
            assert len(fake.requests) == 1

            def down(req):
                raise httpx.ConnectError("refused")
            m.llm_adapter.provider._client = httpx.Client(transport=httpx.MockTransport(down))
            body = client.post("/api/chat", json={"messages": [{"role": "user", "content": WHY_Q}], "filters": {}}).json()
            assert body["route"] == "LLM_REQUIRED" and "Ollama" in body["answer"]
            assert "GEMINI_API_KEY" not in body["answer"]  # no Gemini-specific message for a Qwen error
        assert gen.call_count == 0  # no silent failover to Gemini
    finally:
        _reload_main(None)
    assert main.llm_adapter.provider.name == "gemini"


def test_invalid_provider_stops_startup():
    try:
        _reload_main("not-a-provider")
        raise AssertionError("startup accepted an unknown provider")
    except llm_providers.ProviderConfigError:
        pass
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
    print(f"\n=== ALL {len(tests)} QWEN PROVIDER TESTS PASSED ===")
