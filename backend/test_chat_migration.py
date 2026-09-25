"""
Tests for /api/chat on the Query Router + DuckDB data layer + LLM Adapter (Groq only).

Groq is replaced by a fake transport (fake_groq.py), so no quota is used. Spies count calls
to the router, the LLM Adapter, Groq and DuckDB so each path's call budget is verified:
DIRECT_DATABASE = 0 LLM calls, LLM_REQUIRED = 1 Groq call, everything else (unsupported,
clarification, follow-ups) = the router's message with 0 LLM calls. There is no fallback path.

Run:  python test_chat_migration.py      (or: pytest test_chat_migration.py)
"""
import html
import json
import os
from contextlib import ExitStack
from unittest.mock import patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)

import httpx
from fastapi.testclient import TestClient

import campaign_repository as cr
import chat_routing
import data_tools as dt
import grounding
import main
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)

FRONTEND_FILTERS = {"platform": "All Platforms", "objective": "All Objectives", "vertical": "All Industry Verticals",
                    "budget": "All Budget Tiers", "retargeting": "Retargeting & Cold Combined",
                    "gender": None, "device": None, "age": None, "creative": None, "emotion": None,
                    "placement": None, "income": None}
WHY_Q = "Why is TikTok performing better?"


def refuse(request):
    raise AssertionError("Groq must not be called")


def ask(question, history=(), filters=None):
    messages = [*history, {"role": "user", "content": question}]
    return client.post("/api/chat", json={"messages": messages, "filters": FRONTEND_FILTERS if filters is None else filters})


class Spies:
    """Counts every layer a chat request touches. `respond` answers Groq requests."""

    def __init__(self, respond=refuse):
        self.stack = ExitStack()
        s = self.stack
        self.groq = fake_groq.install(main.llm_adapter, respond)
        self.router = s.enter_context(patch.object(qr, "route_query", wraps=qr.route_query))
        self.adapter = s.enter_context(patch.object(main.llm_adapter, "explain", wraps=main.llm_adapter.explain))
        self.duckdb = s.enter_context(patch.object(cr.DuckDBCampaignRepository, "_run", autospec=True,
                                                   side_effect=cr.DuckDBCampaignRepository._run))

    @property
    def groq_calls(self):
        return len(self.groq.requests)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()
        fake_groq.block_real_calls(main.llm_adapter)


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
    assert s.groq_calls == 0 and s.adapter.call_count == 0
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
    with patch.object(main.llm_adapter.provider, "_api_key", ""), Spies() as s:
        body = _assert_chat_schema(ask("What is total revenue?", filters=filters))
    expected = dt.get_totals({"platform": "TikTok", "budget": "High"})["revenue"]
    assert chat_routing.fmt("revenue", expected) in body["answer"]
    assert "Filtered view: platform = TikTok, budget tier = High." in body["answer"]
    assert s.groq_calls == 0


def test_unavailable_data_answered_without_llm():
    with Spies() as s:
        body = _assert_chat_schema(ask("Show revenue by country"))
    assert body["route"] == "UNSUPPORTED" and "not available" in body["answer"]
    assert s.groq_calls == 0 and s.adapter.call_count == 0


# --- LLM_REQUIRED: router -> DuckDB -> compact analysis -> adapter -> ONE Groq call -----

def _llm(question):
    with Spies(fake_groq.reply("What the data shows: **TikTok** leads.\nInterpretation: may <be> X.")) as s:
        body = _assert_chat_schema(ask(question))
    assert body["route"] == "LLM_REQUIRED", body
    assert s.router.call_count == 1 and s.duckdb.call_count >= 1
    assert s.adapter.call_count == 1 and s.groq_calls == 1
    question_arg, llm_input = s.adapter.call_args.args
    assert question_arg == question
    assert llm_input["results"] == qr.route_query(question, FRONTEND_FILTERS)["analysis"]  # same compact analysis
    prompt = s.groq.user_prompt()
    assert len(prompt.encode()) < 20_000 and prompt.count('"id":') <= qr.MAX_LLM_CAMPAIGN_ROWS
    # Explanation is escaped for the HTML chat bubble; bold and line breaks kept.
    assert body["answer"] == "What the data shows: <b>TikTok</b> leads.<br>Interpretation: may &lt;be&gt; X."
    return llm_input


def test_llm_why_tiktok_better():
    llm_input = _llm(WHY_Q)
    assert llm_input["results"][0]["tool"] == "rank_dimension"


def test_llm_explain_ctr_decline():
    assert any(a["tool"] == "get_creative_fatigue" for a in _llm("Explain the decline in CTR.")["results"])


def test_llm_roas_difference():
    llm_input = _llm("What might explain the ROAS difference?")
    assert llm_input["results"][0]["tool_input"]["metric"] == "roas"


def test_groq_answer_with_an_invented_number_is_never_shown():
    with Spies(fake_groq.reply("What the data shows: revenue was $211,005,852.77.")) as s:
        body = _assert_chat_schema(ask(WHY_Q))
    assert s.groq_calls == 1 and "211,005,852.77" not in body["answer"]
    assert body["answer"].startswith("What the data shows:")
    assert html.escape(grounding.WITHHELD_NOTE) in body["answer"]


# --- No fallback: questions the router can't answer get its message, zero LLM calls ------

def _clarified(question, history=(), route=None, expect_router=True):
    with Spies() as s:
        body = _assert_chat_schema(ask(question, history))
    assert body["route"] in ((route,) if route else ("NEEDS_CLARIFICATION", "UNSUPPORTED")), body
    assert s.groq_calls == 0 and s.adapter.call_count == 0
    assert s.router.call_count == (1 if expect_router else 0)
    return html.unescape(body["answer"])


def test_judgement_questions_get_the_router_message():
    for q in ("Which campaigns have high spend but low revenue?", "Which campaigns are underperforming?",
              "Is revenue growing?"):
        plan = qr.classify_query(q)
        assert _clarified(q, route=plan.route.value) == plan.message


def test_relative_dates_and_small_talk_get_the_router_message():
    assert "time reference" in _clarified("What was revenue last month?", route="NEEDS_CLARIFICATION")
    assert "isn't about the MarketingIQ campaign data" in _clarified("Hi there!", route="UNSUPPORTED")


def test_context_dependent_follow_ups_ask_for_the_full_question():
    history = [{"role": "user", "content": "What is TikTok's revenue?"},
               {"role": "assistant", "content": "TikTok: revenue $34,349,996.88"}]
    for q in ("What about LinkedIn?", "And last month?", "Why is that?"):
        assert _clarified(q, history, route="NEEDS_CLARIFICATION", expect_router=False) == chat_routing.ASK_IN_FULL


def test_empty_or_invalid_questions_are_answered_safely():
    assert _clarified("   ", route="NEEDS_CLARIFICATION", expect_router=False) == chat_routing.ASK_A_QUESTION
    assert "too long" in _clarified("revenue " * 200, route="NEEDS_CLARIFICATION")
    with Spies() as s:
        body = _assert_chat_schema(client.post("/api/chat", json={"messages": [], "filters": {}}))
    assert body["route"] == "NEEDS_CLARIFICATION" and s.groq_calls == 0


def test_standalone_question_with_history_still_routed():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    with Spies() as s:
        body = _assert_chat_schema(ask("What is total spend?", history))
    assert body["route"] == "DIRECT_DATABASE" and s.groq_calls == 0


# --- Failures, retries, timeouts: bounded, always a 200 answer with the data ---------------

def _llm_failure(respond):
    with Spies(respond) as s:
        body = _assert_chat_schema(ask(WHY_Q))
    assert body["route"] == "LLM_REQUIRED"
    answer = html.unescape(body["answer"])
    # The figures from DuckDB are still shown, then a neutral notice (no vendor, no key talk).
    assert answer.startswith("What the data shows:") and "TikTok" in answer
    assert "The analytics data is still available" in answer
    for word in ("groq", "gemini", "api key", "api_key", "gsk_"):
        assert word not in answer.lower(), answer
    return answer, s.groq_calls


def test_rate_limit_is_reported_not_retried():
    answer, calls = _llm_failure(lambda req: httpx.Response(429, json={"error": {"message": "Rate limit reached"}}))
    assert "temporarily rate-limited" in answer and calls == 1


def test_server_errors_retried_once_at_most():
    answer, calls = _llm_failure(lambda req: httpx.Response(503, text="overloaded"))
    assert "temporarily overloaded" in answer and calls == 2
    responses = iter([httpx.Response(503, text="overloaded"), httpx.Response(200, json=fake_groq.completion("Recovered."))])
    with Spies(lambda req: next(responses)) as s:
        body = _assert_chat_schema(ask(WHY_Q))
    assert body["answer"] == "Recovered." and s.groq_calls == 2


def test_timeouts_terminate_without_retry():
    def slow(req):
        raise httpx.ReadTimeout("timed out")
    answer, calls = _llm_failure(slow)
    assert "took too long" in answer and calls == 1


def test_malformed_provider_responses():
    answer, calls = _llm_failure(lambda req: httpx.Response(200, json={"choices": []}))
    assert "unusable response" in answer and calls == 1


def test_no_key_configured():
    with patch.object(main.llm_adapter.provider, "_api_key", ""):
        answer, calls = _llm_failure(refuse)
    assert "unavailable right now" in answer and calls == 0


def test_data_error_is_safe_and_skips_llm():
    def broken(**_):
        raise cr.DataAccessError()
    with patch.dict(dt.TOOL_REGISTRY, {"get_totals": broken}):
        with Spies() as s:
            body = _assert_chat_schema(ask("What is total revenue?"))
    assert body["route"] == "DATA_ERROR" and "analytics query failed" in body["answer"]
    assert s.groq_calls == 0
    assert "SELECT" not in body["answer"] and "Traceback" not in body["answer"]


def test_empty_analysis_never_reaches_groq():
    with Spies() as s:
        body = _assert_chat_schema(ask("Why is revenue declining?", filters={**FRONTEND_FILTERS, "objective": "Nope"}))
    assert body["route"] == "LLM_REQUIRED" and "nothing to explain" in body["answer"]
    assert s.groq_calls == 0


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
    assert not any(fake_groq.FAKE_KEY in r.getMessage() for r in records)


def test_llm_answer_is_escaped_html():
    """The dashboard inserts answers as HTML, so Groq's text is escaped (bold and line breaks kept)."""
    with Spies(fake_groq.reply('<img src=x onerror="alert(document.cookie)"> **Revenue** may rise.\nNext line')):
        body = _assert_chat_schema(ask(WHY_Q))
    assert body["route"] == "LLM_REQUIRED"
    assert body["answer"] == '&lt;img src=x onerror=&quot;alert(document.cookie)&quot;&gt; <b>Revenue</b> may rise.<br>Next line'


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
