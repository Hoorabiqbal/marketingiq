"""
Tests for the deterministic Query Router (query_router.py) and the /api/query endpoint.

No external LLM is used: Gemini's generate_content is patched to fail loudly if
anything calls it, so these tests never consume API quota and need no API key.
Expected numbers are computed independently from the raw DataFrame with pandas,
never hardcoded, so they verify the router returns REAL dataset values.

Run:  python test_query_router.py      (or: pytest test_query_router.py)
"""
import json
import logging
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai import models as genai_models

import main
import data_tools as dt
import query_router as qr

client = TestClient(main.app)
df = dt.get_dataframe()


def _route(query, filters=None):
    return qr.route_query(query, filters)


def _raises(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__}")


# --- DIRECT_DATABASE ---------------------------------------------------------

def test_total_revenue():
    r = _route("What is total revenue?")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "get_totals" and r["llm_required"] is False
    assert r["result"]["revenue"] == round(float(df["revenue"].sum()), 2)
    assert r["focus_metrics"] == ["revenue"]


def test_total_spend():
    r = _route("What is total spend?")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "get_totals"
    assert r["result"]["spend"] == round(float(df["ad_spend"].sum()), 2)
    assert r["focus_metrics"] == ["spend"]


def test_highest_roas_platform():
    r = _route("Which platform has the highest ROAS?")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "rank_dimension"
    assert r["tool_input"] == {"dimension": "platform", "metric": "roas", "order": "desc", "limit": 1}
    g = df.groupby("platform")[["revenue", "ad_spend"]].sum()
    assert r["result"]["results"][0]["name"] == (g["revenue"] / g["ad_spend"]).idxmax()


def test_campaign_count():
    r = _route("How many campaigns are there?")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "get_totals"
    assert r["result"]["campaign_count"] == len(df) == 10000


def test_monthly_revenue():
    r = _route("Show monthly revenue.")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "trend_over_time"
    assert r["tool_input"] == {"metric": "revenue"}
    series = r["result"]["series"]
    assert len(series) == df["month"].nunique()
    assert abs(sum(p["value"] for p in series) - df["revenue"].sum()) < 1


def test_revenue_in_specific_month():
    r = _route("What was revenue in January 2025?")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "trend_over_time" and r["period"] == "2025-01"
    assert r["result"]["series"] == [{"month": "2025-01",
                                      "value": round(float(df.loc[df["month"] == "2025-01", "revenue"].sum()), 2)}]


def test_roas_threshold_filter():
    r = _route("Show campaigns with ROAS above 8.")
    assert r["route"] == "DIRECT_DATABASE" and r["tool"] == "filter_campaigns"
    assert r["tool_input"]["conditions"] == [{"field": "roas", "operator": ">", "value": 8.0}]
    assert r["result"]["matched_count"] == int((df["ROAS"] > 8).sum())
    assert all(c["roas"] > 8 for c in r["result"]["campaigns"])


def test_compare_revenue_by_platform():
    r = _route("Compare revenue by platform.")
    assert r["tool"] == "rank_dimension" and r["tool_input"]["metric"] == "revenue"
    assert len(r["result"]["results"]) == df["platform"].nunique()


def test_our_roas_and_named_entities():
    assert _route("What is our ROAS?")["tool"] == "get_totals"
    assert _route("Compare TikTok and LinkedIn")["tool"] == "compare_entities"
    best_cpa = _route("Which platform has the best CPA?")
    assert best_cpa["tool_input"]["order"] == "asc"  # lower CPA is better


def test_dashboard_filters_are_applied():
    r = _route("What is total revenue?", {"platform": "TikTok"})
    assert r["result"]["revenue"] == round(float(df.loc[df["platform"] == "TikTok", "revenue"].sum()), 2)
    assert r["filters_applied"] == {"platform": "TikTok"}


# --- LLM_REQUIRED ------------------------------------------------------------

def _assert_llm(r, expected_tools):
    assert r["route"] == "LLM_REQUIRED" and r["llm_required"] is True and r["explanation"] is None
    tools = [a["tool"] for a in r["analysis"]]
    assert set(expected_tools) <= set(tools), tools
    assert len(tools) <= qr.MAX_LLM_TOOL_CALLS
    assert r["llm_context_bytes"] == len(json.dumps(r["analysis"], default=str)) <= qr.MAX_LLM_CONTEXT_BYTES


def test_why_tiktok_better():
    r = _route("Why is TikTok performing better?")
    _assert_llm(r, ["rank_dimension", "get_totals"])
    assert r["analysis"][0]["tool_input"]["dimension"] == "platform"


def test_explain_ctr_decline():
    _assert_llm(_route("Explain the decline in CTR."), ["get_creative_fatigue"])


def test_roas_difference_cause():
    r = _route("What might be causing the ROAS difference?")
    _assert_llm(r, ["rank_dimension"])
    assert r["analysis"][0]["tool_input"]["metric"] == "roas"


def test_llm_context_never_contains_dataset():
    # Thousands of campaigns match, but at most MAX_LLM_CAMPAIGN_ROWS may reach the LLM context.
    r = _route("Why do campaigns with spend over 100 perform differently?")
    fc = next(a for a in r["analysis"] if a["tool"] == "filter_campaigns")
    assert fc["result"]["matched_count"] > 1000
    assert len(fc["result"]["campaigns"]) <= qr.MAX_LLM_CAMPAIGN_ROWS
    assert r["llm_context_bytes"] < qr.MAX_LLM_CONTEXT_BYTES
    compacted = qr._compact({"campaigns": list(range(10000)), "series": list(range(10000))})
    assert len(compacted["campaigns"]) == qr.MAX_LLM_CAMPAIGN_ROWS
    assert len(compacted["series"]) == qr.MAX_LLM_LIST_ITEMS
    assert compacted["series_truncated_from"] == 10000


# --- Edge cases --------------------------------------------------------------

def test_empty_query():
    for q in ("", "   "):
        assert _raises(qr.InvalidQueryError, _route, q).status_code == 400
    r = client.post("/api/query", json={"query": "  "})
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_unsupported_query():
    assert _route("What's the weather in Paris?")["route"] == "UNSUPPORTED"
    r = _route("Show revenue by country")
    assert r["route"] == "UNSUPPORTED" and "not available" in r["message"]
    assert _route("Show campaigns")["route"] == "NEEDS_CLARIFICATION"
    assert _route("Show ROAS by month")["route"] == "NEEDS_CLARIFICATION"


def test_malformed_input():
    for bad in (None, 123, ["total revenue"]):
        _raises(qr.InvalidQueryError, _route, bad)
    _raises(qr.InvalidQueryError, _route, "total revenue", "not-a-dict")
    _raises(qr.InvalidQueryError, _route, "total revenue", {"platform": {"nested": 1}})
    _raises(qr.InvalidQueryError, _route, "x" * (qr.MAX_QUERY_LENGTH + 1))
    assert client.post("/api/query", json={}).status_code == 422
    assert client.post("/api/query", json={"query": 123}).status_code == 422
    assert client.post("/api/query", json={"query": "revenue", "filters": ["x"]}).status_code == 422
    assert client.post("/api/query", content=b"not json",
                       headers={"content-type": "application/json"}).status_code == 422


# --- Error handling ----------------------------------------------------------

def test_tool_unavailable():
    registry = {k: v for k, v in dt.TOOL_REGISTRY.items() if k != "get_totals"}
    with patch.dict(dt.TOOL_REGISTRY, registry, clear=True):
        r = client.post("/api/query", json={"query": "What is total revenue?"})
    assert r.status_code == 503


def test_database_error_hides_internals():
    def broken(**_):
        raise ValueError("secret internal detail")
    with patch.dict(dt.TOOL_REGISTRY, {"get_totals": broken}):
        r = client.post("/api/query", json={"query": "What is total revenue?"})
    assert r.status_code == 500
    assert "secret" not in r.text and "Traceback" not in r.text


def test_data_not_loaded():
    with patch.object(dt, "_repo", None):  # the data layer, since Phase 4 (was the raw DataFrame)
        r = client.post("/api/query", json={"query": "What is total revenue?"})
    assert r.status_code == 503


# --- Endpoint, no-LLM guarantee, logging --------------------------------------

def test_endpoint_direct_queries_never_call_an_llm():
    # Since Phase 3, LLM_REQUIRED questions call the LLM Adapter (see test_llm_adapter.py);
    # direct and clarification routes must still never touch an LLM.
    queries = ["What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue.",
               "How many campaigns are there?", "Show campaigns", "What's the weather in Paris?"]
    with patch.object(genai_models.Models, "generate_content",
                      side_effect=AssertionError("LLM must not be called")) as gen, \
         patch.object(main.rotator, "call", side_effect=AssertionError("LLM must not be called")) as rot:
        for q in queries:
            r = client.post("/api/query", json={"query": q})
            assert r.status_code == 200, (q, r.text)
    assert gen.call_count == 0 and rot.call_count == 0


def test_endpoint_returns_real_data():
    r = client.post("/api/query", json={"query": "What is total spend?", "filters": {"budget": "High"}})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "DIRECT_DATABASE" and body["tool"] == "get_totals"
    assert body["result"]["spend"] == round(float(df.loc[df["budget_tier"] == "High", "ad_spend"].sum()), 2)


def test_structured_log_line():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    level = qr.logger.level
    qr.logger.addHandler(handler)
    qr.logger.setLevel(logging.INFO)
    try:
        _route("Which platform has the highest ROAS?")
        _raises(qr.InvalidQueryError, _route, "")
    finally:
        qr.logger.removeHandler(handler)
        qr.logger.setLevel(level)
    ok, err = (json.loads(r.getMessage()) for r in records)
    assert ok["status"] == "ok" and ok["route"] == "DIRECT_DATABASE" and ok["tools"] == ["rank_dimension"]
    assert ok["query"] == "Which platform has the highest ROAS?" and ok["elapsed_ms"] >= 0
    assert err["status"] == "error" and err["error"] == "InvalidQueryError"
    assert "key" not in json.dumps(ok).lower()


if __name__ == "__main__":
    qr.logger.setLevel(logging.WARNING)  # keep the per-request INFO lines out of the test output
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} QUERY ROUTER TESTS PASSED ===")
