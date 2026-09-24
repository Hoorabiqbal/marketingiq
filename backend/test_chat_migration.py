"""
Tests for /api/chat on the Query Router + DuckDB data layer + LLM Adapter, with the
original Gemini tool-use loop as fallback.

Every Gemini call is mocked (google-genai's Models.generate_content is patched for the
whole module to fail loudly), so no API quota is used. Spies count calls to the router,
the LLM Adapter, the fallback loop and DuckDB so each path's call budget is verified:
DIRECT_DATABASE = 0 LLM calls, LLM_REQUIRED = 1 explanation call, fallback = the loop.

Run:  python test_chat_migration.py      (or: pytest test_chat_migration.py)
"""
import html
import json
import os
from contextlib import ExitStack
from unittest.mock import patch

# Two known keys so quota rotation is observable (set before main loads backend/.env,
# which never overrides variables that already exist).
os.environ["GEMINI_API_KEY_1"] = "test-key-one"
os.environ["GEMINI_API_KEY_2"] = "test-key-two"
os.environ.pop("CHAT_ROUTER_ENABLED", None)

import httpx
from fastapi.testclient import TestClient
from google.genai import errors as ge
from google.genai import models as genai_models
from google.genai import types

import campaign_repository as cr
import chat_routing
import data_tools as dt
import main
import query_router as qr

_guard = patch.object(genai_models.Models, "generate_content",
                      side_effect=AssertionError("real Gemini call attempted in a test"))
_guard.start()

client = TestClient(main.app)
N_KEYS = len(main.rotator.clients)

FRONTEND_FILTERS = {"platform": "All Platforms", "objective": "All Objectives", "vertical": "All Industry Verticals",
                    "budget": "All Budget Tiers", "retargeting": "Retargeting & Cold Combined",
                    "gender": None, "device": None, "age": None, "creative": None, "emotion": None,
                    "placement": None, "income": None}


def text_response(text):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))])


def tool_call_response(name, args):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))])


def api_error(cls, code, msg):
    return cls(code=code, response_json={"error": {"message": msg}})


def ask(question, history=(), filters=None):
    messages = [*history, {"role": "user", "content": question}]
    return client.post("/api/chat", json={"messages": messages, "filters": FRONTEND_FILTERS if filters is None else filters})


class Spies:
    """Counts every layer a chat request touches."""

    def __init__(self, gemini_side_effect=AssertionError("Gemini must not be called")):
        self.stack = ExitStack()
        s = self.stack
        self.gemini = s.enter_context(patch.object(genai_models.Models, "generate_content", side_effect=gemini_side_effect))
        self.router = s.enter_context(patch.object(qr, "route_query", wraps=qr.route_query))
        self.adapter = s.enter_context(patch.object(main.llm_adapter, "explain", wraps=main.llm_adapter.explain))
        self.fallback = s.enter_context(patch.object(main, "_gemini_tool_loop", wraps=main._gemini_tool_loop))
        self.duckdb = s.enter_context(patch.object(cr.DuckDBCampaignRepository, "_run", autospec=True,
                                                   side_effect=cr.DuckDBCampaignRepository._run))
        s.enter_context(patch.object(main.time, "sleep"))  # no real backoff waits

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()


def _assert_chat_schema(r):
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str) and body["answer"].strip()
    return body


# --- DIRECT_DATABASE: router -> DuckDB -> formatted answer, zero LLM calls -----------

def _direct(question):
    with Spies() as s:
        body = _assert_chat_schema(ask(question))
    assert body["route"] == "DIRECT_DATABASE", body
    assert s.router.call_count == 1 and s.duckdb.call_count >= 1
    assert s.gemini.call_count == 0 and s.adapter.call_count == 0 and s.fallback.call_count == 0
    return body["answer"]


def test_direct_total_revenue():
    answer = _direct("What is total revenue?")
    assert chat_routing.fmt("revenue", dt.get_totals()["revenue"]) in answer  # $284,157,706.47


def test_direct_total_spend():
    assert chat_routing.fmt("spend", dt.get_totals()["spend"]) in _direct("What is total spend?")


def test_direct_campaign_count():
    assert _direct("How many campaigns are there?") == "There are 10,000 campaigns in the dataset."


def test_direct_highest_roas_platform():
    top = dt.rank_dimension("platform", "roas", limit=1)["results"][0]
    assert _direct("Which platform has the highest ROAS?") == (
        f"{top['name']} has the highest ROAS: {chat_routing.fmt('roas', top['roas'])} "
        f"({top['campaign_count']:,} campaigns).")


def test_direct_monthly_revenue():
    answer = _direct("Show monthly revenue.")
    series = dt.trend_over_time("revenue")["series"]
    lines = answer.split("<br>")
    assert lines[0] == "Monthly revenue:" and len(lines) == len(series) + 1
    jan = next(p for p in series if p["month"] == "2025-01")
    assert f"January 2025: {chat_routing.fmt('revenue', jan['value'])}" in lines


def test_direct_respects_dashboard_filters_and_works_without_a_key():
    filters = {**FRONTEND_FILTERS, "platform": "TikTok", "budget": "High"}
    with patch.object(main.rotator, "clients", []):
        with Spies():
            body = _assert_chat_schema(ask("What is total revenue?", filters=filters))
    expected = dt.get_totals({"platform": "TikTok", "budget": "High"})["revenue"]
    assert chat_routing.fmt("revenue", expected) in body["answer"]
    assert "Filtered view: platform = TikTok, budget tier = High." in body["answer"]


def test_unavailable_data_answered_without_llm():
    with Spies() as s:
        body = _assert_chat_schema(ask("Show revenue by country"))
    assert body["route"] == "UNSUPPORTED" and "not available" in body["answer"]
    assert s.gemini.call_count == 0 and s.fallback.call_count == 0


# --- LLM_REQUIRED: router -> DuckDB -> compact analysis -> adapter -> ONE Gemini call --

def _llm(question):
    with Spies(gemini_side_effect=[text_response("What the data shows: **TikTok** leads.\nInterpretation: may <be> X.")]) as s:
        body = _assert_chat_schema(ask(question))
    assert body["route"] == "LLM_REQUIRED", body
    assert s.router.call_count == 1 and s.duckdb.call_count >= 1
    assert s.adapter.call_count == 1 and s.gemini.call_count == 1 and s.fallback.call_count == 0
    question_arg, llm_input = s.adapter.call_args.args
    assert question_arg == question
    assert llm_input["results"] == qr.route_query(question, FRONTEND_FILTERS)["analysis"]  # same compact analysis
    prompt = s.gemini.call_args.kwargs["contents"][0].parts[0].text
    assert len(prompt.encode()) < 20_000 and prompt.count('"id":') <= qr.MAX_LLM_CAMPAIGN_ROWS
    # Explanation is escaped for the HTML chat bubble; bold and line breaks kept.
    assert body["answer"] == "What the data shows: <b>TikTok</b> leads.<br>Interpretation: may &lt;be&gt; X."
    return llm_input


def test_llm_why_tiktok_better():
    llm_input = _llm("Why is TikTok performing better?")
    assert llm_input["results"][0]["tool"] == "rank_dimension"


def test_llm_explain_ctr_decline():
    assert any(a["tool"] == "get_creative_fatigue" for a in _llm("Explain the decline in CTR.")["results"])


def test_llm_roas_difference():
    llm_input = _llm("What might explain the ROAS difference?")
    assert llm_input["results"][0]["tool_input"]["metric"] == "roas"


# --- Fallback: only when the router can't plan the question --------------------------

def _fallback(question, history=(), expect_router=True):
    with Spies(gemini_side_effect=[tool_call_response("get_totals", {}), text_response("Fallback answer.")]) as s:
        body = _assert_chat_schema(ask(question, history))
    assert body["route"] == "FALLBACK" and body["answer"] == "Fallback answer.", body
    assert s.fallback.call_count == 1 and s.adapter.call_count == 0
    assert s.gemini.call_count == 2  # tool call, then final answer: the original loop works
    tool_result = s.gemini.call_args_list[1].kwargs["contents"][-1].parts[0].function_response.response
    assert tool_result["result"]["campaign_count"] == 10000  # tools still run on the data layer
    assert s.router.call_count == (1 if expect_router else 0)


def test_fallback_for_judgement_questions():
    _fallback("Which campaigns have high spend but low revenue?")
    _fallback("Which campaigns are underperforming?")
    _fallback("Is revenue growing?")


def test_fallback_for_relative_dates_and_small_talk():
    _fallback("What was revenue last month?")
    _fallback("Hi there!")


def test_fallback_for_context_dependent_follow_ups():
    history = [{"role": "user", "content": "What is TikTok's revenue?"},
               {"role": "assistant", "content": "TikTok: revenue $34,349,996.88"}]
    _fallback("What about LinkedIn?", history, expect_router=False)
    _fallback("And last month?", history, expect_router=False)
    _fallback("Why is that?", history, expect_router=False)


def test_standalone_question_with_history_still_routed():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    with Spies() as s:
        body = _assert_chat_schema(ask("What is total spend?", history))
    assert body["route"] == "DIRECT_DATABASE" and s.gemini.call_count == 0


def test_router_rollback_switch():
    with patch.object(main, "CHAT_ROUTER_ENABLED", False):
        with Spies(gemini_side_effect=[text_response("Legacy answer.")]) as s:
            body = _assert_chat_schema(ask("What is total revenue?"))
    assert body["route"] == "FALLBACK" and s.router.call_count == 0 and s.gemini.call_count == 1


# --- Failures, retries, timeouts: bounded, one path only, always a 200 answer ----------

def _llm_failure(side_effect):
    with Spies(gemini_side_effect=side_effect) as s:
        body = _assert_chat_schema(ask("Why is TikTok performing better?"))
    assert body["route"] == "LLM_REQUIRED" and s.fallback.call_count == 0  # never a second LLM path
    return body["answer"], s.gemini.call_count


def _fallback_failure(side_effect):
    with Spies(gemini_side_effect=side_effect) as s:
        body = _assert_chat_schema(ask("Which campaigns are underperforming?"))
    assert body["route"] == "FALLBACK" and s.adapter.call_count == 0
    return html.unescape(body["answer"]), s.gemini.call_count  # answers are HTML-escaped text


QUOTA = api_error(ge.ClientError, 429, "RESOURCE_EXHAUSTED: quota")
SERVER = api_error(ge.ServerError, 503, "overloaded")


def test_quota_rotates_keys_then_stops():
    assert N_KEYS >= 2
    answer, calls = _llm_failure(QUOTA)
    assert "quota" in answer and calls == N_KEYS  # each key tried once, no further retry
    answer, calls = _fallback_failure(QUOTA)
    assert "quota" in answer and calls == N_KEYS
    answer, calls = _llm_failure([QUOTA, text_response("Answer via key 2.")])
    assert answer == "Answer via key 2." and calls == 2


def test_server_errors_retried_three_times_at_most():
    answer, calls = _llm_failure(SERVER)
    assert "overloaded" in answer and calls == 3
    answer, calls = _fallback_failure(SERVER)
    assert "overloaded" in answer and calls == 3
    answer, calls = _llm_failure([SERVER, text_response("Recovered.")])
    assert answer == "Recovered." and calls == 2


def test_timeouts_terminate_without_retry():
    answer, calls = _llm_failure(httpx.ReadTimeout("timed out"))
    assert "took too long" in answer and calls == 1
    answer, calls = _fallback_failure(httpx.ReadTimeout("timed out"))
    assert "took too long" in answer and calls == 1
    with patch.object(main, "CHAT_FALLBACK_TIMEOUT_S", 0.2):  # deadline already too short
        answer, calls = _fallback_failure(AssertionError("must not be called"))
    assert "took too long" in answer and calls == 0
    with Spies(gemini_side_effect=[tool_call_response("get_totals", {}), text_response("ok")]) as s:
        _assert_chat_schema(ask("Which campaigns are underperforming?"))
        config = s.gemini.call_args_list[0].kwargs["config"]
    assert 0 < config.http_options.timeout <= main.CHAT_FALLBACK_TIMEOUT_S * 1000


def test_malformed_provider_responses():
    empty = types.GenerateContentResponse(candidates=[])
    answer, calls = _llm_failure(empty)
    assert "unusable response" in answer and calls == 1
    answer, calls = _fallback_failure(empty)
    assert "didn't return a response" in answer and calls == 1


def test_no_key_configured():
    with patch.object(main.rotator, "clients", []):
        answer, calls = _llm_failure(AssertionError("must not be called"))
        assert "GEMINI_API_KEY" in answer and calls == 0
        answer, calls = _fallback_failure(AssertionError("must not be called"))
        assert "GEMINI_API_KEY" in answer and calls == 0


def test_data_error_is_safe_and_skips_llm():
    def broken(**_):
        raise cr.DataAccessError()
    with patch.dict(dt.TOOL_REGISTRY, {"get_totals": broken}):
        with Spies() as s:
            body = _assert_chat_schema(ask("What is total revenue?"))
    assert body["route"] == "DATA_ERROR" and "analytics query failed" in body["answer"]
    assert s.gemini.call_count == 0 and s.fallback.call_count == 0
    assert "SELECT" not in body["answer"] and "Traceback" not in body["answer"]


def test_empty_analysis_never_reaches_gemini():
    with Spies() as s:
        body = _assert_chat_schema(ask("Why is revenue declining?", filters={**FRONTEND_FILTERS, "objective": "Nope"}))
    assert body["route"] == "LLM_REQUIRED" and "nothing to explain" in body["answer"]
    assert s.gemini.call_count == 0 and s.fallback.call_count == 0


def test_chat_log_line_has_route_and_no_secrets():
    records = []
    handler = __import__("logging").Handler()
    handler.emit = records.append
    level = qr.logger.level
    qr.logger.addHandler(handler)
    qr.logger.setLevel(20)
    try:
        with Spies():
            ask("What is total revenue?")
    finally:
        qr.logger.removeHandler(handler)
        qr.logger.setLevel(level)
    chat_lines = [json.loads(r.getMessage()) for r in records if "chat_answered" in r.getMessage()]
    assert chat_lines and chat_lines[-1]["route"] == "DIRECT_DATABASE" and chat_lines[-1]["elapsed_ms"] >= 0
    assert not any("test-key" in r.getMessage() for r in records)


def test_fallback_answer_is_escaped_html():
    """The dashboard inserts answers as HTML, so Gemini's fallback text is escaped like routed answers."""
    history = [{"role": "user", "content": "What is total revenue?"}, {"role": "assistant", "content": "It is $1."}]
    with Spies(gemini_side_effect=[text_response('<img src=x onerror="alert(1)"> **Revenue** rose.\nNext line')]):
        body = _assert_chat_schema(ask("Why is that?", history))
    assert body["route"] == "FALLBACK"
    assert body["answer"] == ('&lt;img src=x onerror=&quot;alert(1)&quot;&gt; <b>Revenue</b> rose.<br>Next line')


def test_cors_allows_only_the_frontend_and_local_pages():
    def preflight(origin):
        return client.options("/api/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "content-type"})
    for origin in ("https://marketingiqp.netlify.app", "http://127.0.0.1:5500", "http://localhost:8080"):
        r = preflight(origin)
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == origin, origin
    for origin in ("https://evil.example.com", "null", "https://marketingiqp.netlify.app.evil.com",
                   "http://localhost.evil.com"):
        r = preflight(origin)
        assert r.status_code == 400 and "access-control-allow-origin" not in r.headers, origin
    simple = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert simple.status_code == 200 and "access-control-allow-origin" not in simple.headers
    assert main.CORS_ALLOW_ORIGINS == ["https://marketingiqp.netlify.app"] or os.getenv("CORS_ALLOW_ORIGINS")


if __name__ == "__main__":
    import logging
    for name in ("marketingiq.query_router", "marketingiq.llm_adapter", "marketingiq.data"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} CHAT MIGRATION TESTS PASSED ===")
