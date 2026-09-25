"""
Tests for the general analytics planner (query_planner.py) behind /api/chat.

Groq runs on a fake transport (fake_groq.py): each planner test supplies the JSON plan Groq would
return, so no quota is used. What is verified is everything the backend owns: which questions reach
the planner (router-understood ones never do), that one planning call is the most a question costs,
that every plan is validated before any data access, and that every number in the answer is the
data tools' own DuckDB result, with directions and rankings computed deterministically.

Run:  python test_planner.py      (or: pytest test_planner.py)
"""
import json
import logging
from unittest.mock import patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)

from fastapi.testclient import TestClient

import chat_routing
import data_tools as dt
import grounding
import main
import query_planner as qp
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)
FAILED = qr.PLANNER_HINT


def plan(**fields):
    base = {"intent": "total", "metrics": ["revenue"], "dimension": "none", "entities": [], "filters": [],
            "months": [], "years": [], "grain": "none", "conditions": [], "order": "desc", "limit": 0,
            "visualization": "none", "reason": "none"}
    base.update(fields)
    return json.dumps(base)


def refuse(request):
    raise AssertionError("Groq must not be called")


def ask(question, reply=None, filters=None):
    """/api/chat with Groq answering `reply` (raw planner text) or refusing. Returns body, Groq calls."""
    respond = fake_groq.reply(reply) if reply is not None else refuse
    fake = fake_groq.install(main.llm_adapter, respond)
    try:
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": question}], "filters": filters or {}})
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    assert r.status_code == 200, r.text
    return r.json(), len(fake.requests)


def lines(body):
    return body["answer"].split("<br>")


def fmt(metric, value):
    return grounding.fmt(metric, value)


def jan(year):
    return dt.get_totals({"month_from": f"{year}-01", "month_to": f"{year}-01"})


def year_totals(year):
    return dt.get_totals({"month_from": f"{year}-01", "month_to": f"{year}-12"})


JANUARY_REVENUE = plan(intent="period_comparison", months=[1], grain="year", order="none")


def assert_january_revenue_answer(body, calls):
    assert body["route"] == "PLANNED_DATABASE" and calls == 1, body
    j24, j25, j26 = jan(2024)["revenue"], jan(2025)["revenue"], jan(2026)["revenue"]
    text = body["answer"]
    assert f"January 2024: {fmt('revenue', j24)}" in text and f"January 2025: {fmt('revenue', j25)}" in text
    assert qp.direction([j24, j25]) == ("increasing" if j25 > j24 else "decreasing")
    assert ("Yes, it is increasing" if j25 > j24 else "No, it is decreasing") in text
    # January 2026 is partial (data ends 2026-01-30): reported, never compared.
    assert "Not compared: January 2026 is incomplete" in text and fmt("revenue", j26) in text
    assert "January 2026:" not in text


# --- natural-language regression set -----------------------------------------------

def test_nl01_january_sales_increasing_overtime():
    assert_january_revenue_answer(*ask("Is January sales increasing overtime?", JANUARY_REVENUE))


def test_nl02_january_sales_two_years():
    assert_january_revenue_answer(*ask("Is January sales of 2 years of our data increasing overtime?", JANUARY_REVENUE))


def test_nl03_january_revenue_increasing():
    assert_january_revenue_answer(*ask("Is January revenue increasing over time?", JANUARY_REVENUE))


def test_nl04_compare_january_revenue_across_years():
    # Used to be answered with all-time totals (the month was ignored); now it reaches the planner.
    assert qr.classify_query("Compare January revenue across years.").reason == "needs_planner"
    assert_january_revenue_answer(*ask("Compare January revenue across years.", JANUARY_REVENUE))


def test_nl05_january_revenue_year_over_year():
    assert_january_revenue_answer(*ask("How did January revenue change year over year?", JANUARY_REVENUE))


def test_nl06_platform_improved_roas_most():
    body, calls = ask("Which platform improved ROAS the most over time?",
                      plan(intent="change_ranking", metrics=["roas"], dimension="platform"))
    assert body["route"] == "PLANNED_DATABASE" and calls == 1
    by24 = {r["name"]: r["roas"] for r in dt.rank_dimension("platform", "roas", limit=100,
                                                            filters={"month_from": "2024-01", "month_to": "2024-12"})["results"]}
    by25 = {r["name"]: r["roas"] for r in dt.rank_dimension("platform", "roas", limit=100,
                                                            filters={"month_from": "2025-01", "month_to": "2025-12"})["results"]}
    winner = max(by24, key=lambda n: by25[n] - by24[n])  # computed here, not by the LLM
    assert lines(body)[0].startswith(f"{winner} improved the most in ROAS: {fmt('roas', by24[winner])} in 2024 "
                                     f"to {fmt('roas', by25[winner])} in 2025")
    assert "2026 is not included" in body["answer"]


def test_nl07_revenue_by_platform_for_mobile():
    body, calls = ask("Compare revenue by platform for mobile devices.")  # router: 0 Groq calls
    assert body["route"] == "DIRECT_DATABASE" and calls == 0
    for r in dt.rank_dimension("platform", "revenue", filters={"device": "Mobile"})["results"]:
        assert f"{r['name']}: {chat_routing.fmt('revenue', r['revenue'])}" in body["answer"]
    assert "device = Mobile" in body["answer"]


def test_nl08_highest_roas_platform_in_2025():
    body, calls = ask("Which platform had the highest ROAS in 2025?",
                      plan(intent="breakdown", metrics=["roas"], dimension="platform", years=[2025]))
    top = dt.rank_dimension("platform", "roas", limit=1, filters={"month_from": "2025-01", "month_to": "2025-12"})["results"][0]
    assert body["route"] == "PLANNED_DATABASE" and calls == 1
    assert lines(body)[0] == f"Highest ROAS by platform: {top['name']} ({fmt('roas', top['roas'])})."
    assert "period 2025-01 to 2025-12" in body["answer"]


def test_nl09_monthly_spend_for_tiktok():
    body, calls = ask("Show monthly spend for TikTok.")
    assert body["route"] == "DIRECT_DATABASE" and calls == 0
    first = dt.trend_over_time("spend", {"platform": "TikTok"})["series"][0]
    assert chat_routing.fmt("spend", first["value"]) in body["answer"]


def test_nl10_compare_google_ads_and_tiktok_revenue():
    body, calls = ask("Compare Google Ads and TikTok revenue.")
    assert body["route"] == "DIRECT_DATABASE" and calls == 0
    for r in dt.compare_entities("platform", ["Google Ads", "TikTok"])["results"]:
        assert chat_routing.fmt("revenue", r["revenue"]) in body["answer"]


def test_nl11_campaigns_spent_more_than_10000():
    body, calls = ask("Which campaigns spent more than $10,000?")
    assert body["route"] == "DIRECT_DATABASE" and calls == 0
    n = dt.filter_campaigns([{"field": "spend", "operator": ">", "value": 10000}])["matched_count"]
    assert f"{n:,} campaigns have spend above $10,000.00" in body["answer"]


def test_nl12_marketing_performance_over_time():
    body, calls = ask("How is our marketing performance changing over time?",
                      plan(intent="period_comparison", metrics=["revenue", "spend", "roas"], grain="year"))
    assert body["route"] == "PLANNED_DATABASE" and calls == 1
    y24, y25 = year_totals(2024), year_totals(2025)
    for m in ("revenue", "spend", "roas"):
        assert f"2024: {fmt(m, y24[m])}" in body["answer"] and f"2025: {fmt(m, y25[m])}" in body["answer"]
    assert "Not compared: 2026 is incomplete" in body["answer"]


def test_nl13_unsupported_metric():
    body, calls = ask("Is January engagement score increasing over time?",
                      plan(intent="not_answerable", metrics=[], reason="unknown_metric"))
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 1
    assert "That metric isn&#x27;t in the MarketingIQ data" in body["answer"]
    # A metric the router already knows is unavailable never costs a Groq call.
    body, calls = ask("Is January bounce rate increasing over time?")
    assert body["route"] == "UNSUPPORTED" and calls == 0


def test_nl14_unsupported_dimension():
    body, calls = ask("Which weekday improved revenue the most over time?",
                      plan(intent="not_answerable", metrics=[], reason="unknown_dimension"))
    assert body["route"] == "NEEDS_CLARIFICATION" and calls == 1
    assert "That breakdown isn&#x27;t in the MarketingIQ data" in body["answer"]
    body, calls = ask("Which country improved revenue the most over time?")
    assert body["route"] == "UNSUPPORTED" and calls == 0


def test_nl15_malformed_planner_response():
    for reply in ("Sure! January revenue went up 40%.", "{\"intent\": \"total\"", "[]", ""):
        with patch.object(qr, "_execute", side_effect=AssertionError("data accessed")):
            body, calls = ask("Is January revenue increasing over time?", reply)
        assert body["route"] == "NEEDS_CLARIFICATION" and calls == 1, reply
        assert body["answer"] == chat_routing.html.escape(FAILED)


def test_nl16_ambiguous_question():
    body, calls = ask("Are our campaigns getting better?", plan(intent="not_answerable", metrics=[], reason="ambiguous"))
    assert body["route"] == "NEEDS_CLARIFICATION" and calls == 1
    assert body["answer"] == chat_routing.html.escape(FAILED)
    assert "judgement or time reference" not in body["answer"]


# --- planner-assisted charts, call budget, fast paths ----------------------------------

def test_planner_chart_uses_database_values():
    body, calls = ask("Plot monthly ROAS", plan(intent="trend", metrics=["roas"], grain="month", visualization="line"))
    assert body["route"] == "PLANNED_DATABASE" and calls == 1
    chart = body["chart"]
    assert chart["type"] == "line" and chart["series"][0]["key"] == "roas"
    for row in chart["data"][:3]:
        assert row["roas"] == dt.get_totals({"month_from": row["month"], "month_to": row["month"]})["roas"]
    assert "2026-01" not in [r["month"] for r in chart["data"]]  # partial month excluded
    # A chart is drawn only when the user asked for one, whatever the plan says.
    body, _ = ask("Is January revenue increasing over time?", plan(intent="period_comparison", months=[1],
                                                                    visualization="bar"))
    assert body["chart"] is None


def test_existing_fast_paths_make_no_groq_calls():
    for q, route in (("What is total revenue?", "DIRECT_DATABASE"), ("Which platform has the highest ROAS?", "DIRECT_DATABASE"),
                     ("Show revenue by platform.", "DIRECT_DATABASE"), ("Show monthly revenue as a line chart.", "CHART"),
                     ("Compare revenue and spend over time.", "CHART"), ("What's the weather in Paris?", "UNSUPPORTED")):
        body, calls = ask(q)
        assert body["route"] == route and calls == 0, (q, body["route"])


def test_one_planning_call_at_most_and_errors_are_controlled():
    body, calls = ask("Is January revenue increasing over time?", JANUARY_REVENUE)
    assert calls == 1
    fake = fake_groq.install(main.llm_adapter, lambda r: fake_groq.httpx.Response(429, json={}))
    try:
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Is January revenue increasing?"}],
                                           "filters": {}}).json()
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    assert len(fake.requests) == 1 and r["route"] == "NEEDS_CLARIFICATION" and "429" not in r["answer"]
    with patch.object(main.llm_adapter.provider, "_api_key", ""):
        body, calls = ask("Is January revenue increasing over time?")
    assert calls == 0 and body["answer"] == chat_routing.html.escape(FAILED)


def test_plan_request_is_strict_schema():
    fake = fake_groq.install(main.llm_adapter, fake_groq.reply(JANUARY_REVENUE))
    try:
        client.post("/api/chat", json={"messages": [{"role": "user", "content": "Is January sales increasing overtime?"}]})
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    body = json.loads(fake.requests[0].content)
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == qp.SCHEMA
    assert "SQL" not in body["messages"][0]["content"] and FAKE_KEY_NOT_IN(body)


def FAKE_KEY_NOT_IN(body):
    return fake_groq.FAKE_KEY not in json.dumps(body)


# --- planner security: every malicious plan is rejected before any data access ------------

MALICIOUS = {
    "sql_in_entity": plan(intent="compare_entities", dimension="platform", entities=["TikTok'; DROP TABLE campaigns;--", "LinkedIn"]),
    "sql_in_filter": plan(filters=[{"dimension": "platform", "value": "x' OR '1'='1"}]),
    "unknown_tool_intent": plan(intent="run_sql"),
    "unknown_metric": plan(metrics=["lifetime_value"]),
    "unknown_dimension": plan(intent="breakdown", dimension="country"),
    "unfilterable_dimension": plan(filters=[{"dimension": "audience_interest", "value": "Tech"}]),
    "unexpected_key": json.dumps({**json.loads(plan()), "sql": "SELECT * FROM campaigns"}),
    "missing_key": json.dumps({k: v for k, v in json.loads(plan()).items() if k != "order"}),
    "invalid_operator": plan(intent="campaign_list", conditions=[{"metric": "spend", "operator": "LIKE", "value": 1}]),
    "unknown_condition_field": plan(intent="campaign_list", conditions=[{"metric": "ad_spend; --", "operator": ">", "value": 1}]),
    "non_finite_value": plan(intent="campaign_list").replace('"conditions": []', '"conditions": [{"metric": "spend", "operator": ">", "value": NaN}]'),
    "extreme_limit": plan(intent="breakdown", dimension="platform", limit=10 ** 9),
    "negative_limit": plan(intent="breakdown", dimension="platform", limit=-5),
    "bool_as_limit": plan(intent="breakdown", dimension="platform", limit=True),
    "html_js_value": plan(filters=[{"dimension": "platform", "value": "<script>alert(1)</script>"}]),
    "unknown_chart_type": plan(intent="breakdown", dimension="platform", visualization="heatmap"),
    "bad_month": plan(intent="period_comparison", months=[13]),
    "year_outside_data": plan(years=[1999]),
    "string_metrics": plan(metrics="revenue"),
    "too_many_metrics": plan(metrics=["revenue", "spend", "profit", "roas"]),
    "oversized": plan(entities=["x" * 5000]),
    "not_an_object": json.dumps([json.loads(plan())]),
}


def test_security_malicious_plans_rejected_by_validator():
    for name, text in MALICIOUS.items():
        try:
            qp.validate_plan(text)
        except qp.PlanError:
            continue
        raise AssertionError(f"accepted malicious plan: {name}")


def test_security_malicious_plans_never_reach_the_data_layer():
    for name, text in MALICIOUS.items():
        with patch.object(qr, "_execute", side_effect=AssertionError(f"data accessed: {name}")), \
                patch.object(dt, "get_repository", side_effect=AssertionError(f"repository used: {name}")):
            body, calls = ask("Is January revenue increasing over time?", text)
        assert calls == 1 and body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None, name
        assert "<script" not in body["answer"] and "DROP TABLE" not in body["answer"], name


def test_security_validator_uses_dataset_spelling_only():
    p = qp.validate_plan(plan(intent="compare_entities", dimension="platform", entities=["tiktok", "GOOGLE ADS"],
                              filters=[{"dimension": "device", "value": "mobile"}]))
    assert p.entities == ["TikTok", "Google Ads"] and p.filters == {"device": "Mobile"}
    assert qp.validate_plan(plan(intent="breakdown", dimension="platform", limit=50)).limit == qp.MAX_ROWS


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} PLANNER TESTS PASSED ===")
