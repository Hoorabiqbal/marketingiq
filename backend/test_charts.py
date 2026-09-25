"""
Tests for dynamic AI Analyst charts (chart_builder.py, /api/chat `chart` field).

Every chart number must equal the existing data tools' DuckDB result; chart requests make no
LLM call (Groq runs on a fake transport that fails the test if reached); the spec validator
rejects anything outside the contract.

Run:  python test_charts.py      (or: pytest test_charts.py)
"""
import copy
import logging
from unittest.mock import patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)

from fastapi.testclient import TestClient

import chart_builder as cb
import data_tools as dt
import main
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)


def refuse(request):
    raise AssertionError("Groq must not be called for a chart")


def ask(question, filters=None, respond=refuse):
    fake = fake_groq.install(main.llm_adapter, respond)
    try:
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": question}],
                                           "filters": filters or {}})
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    assert r.status_code == 200, r.text
    return r.json(), len(fake.requests)


def chart(question, filters=None):
    body, groq_calls = ask(question, filters)
    assert body["route"] == "CHART" and body["chart"], body
    assert groq_calls == 0
    return body["chart"]


def expect_invalid(spec, fragment=""):
    try:
        cb.validate_chart(spec)
    except cb.ChartSpecError as e:
        assert fragment in str(e), e
        return
    raise AssertionError(f"accepted: {spec}")


def test_monthly_revenue_line_chart():
    c = chart("Show monthly revenue as a line chart.")
    series = dt.trend_over_time("revenue")["series"]
    assert c["type"] == "line" and c["x_key"] == "month"
    assert c["series"] == [{"key": "revenue", "label": "Revenue", "unit": "$"}]
    assert [(r["month"], r["revenue"]) for r in c["data"]] == [(p["month"], p["value"]) for p in series]


def test_monthly_spend_line_chart():
    c = chart("Plot monthly spend.")
    assert c["type"] == "line" and c["series"][0]["key"] == "spend"
    assert [r["spend"] for r in c["data"]] == [p["value"] for p in dt.trend_over_time("spend")["series"]]


def test_revenue_by_platform_bar_chart():
    c = chart("Show revenue by platform as a bar chart.")
    expected = dt.rank_dimension("platform", "revenue", limit=100)["results"]
    assert c["type"] == "bar" and c["x_key"] == "name" and c["x_label"] == "Platform"
    assert [(r["name"], r["revenue"]) for r in c["data"]] == [(e["name"], e["revenue"]) for e in expected]


def test_roas_by_platform_bar_chart():
    c = chart("Create a bar chart of ROAS by platform.")
    expected = dt.rank_dimension("platform", "roas", limit=100)["results"]
    assert c["series"] == [{"key": "roas", "label": "ROAS", "unit": "x"}]
    assert {r["name"]: r["roas"] for r in c["data"]} == {e["name"]: e["roas"] for e in expected}


def test_platform_revenue_share_pie_chart():
    c = chart("Show platform revenue share as a pie chart.")
    assert c["type"] == "pie" and len(c["series"]) == 1
    assert abs(sum(r["revenue"] for r in c["data"]) - dt.get_totals()["revenue"]) < 1
    # A ratio can't be a pie; a plain category comparison stays a bar.
    body, calls = ask("Pie chart of ROAS by platform")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 0
    assert chart("Chart revenue by platform")["type"] == "bar"


def test_top_5_campaigns_by_revenue():
    c = chart("Bar chart of the top 5 campaigns by revenue.")
    expected = dt.filter_campaigns([], sort_by="revenue", order="desc", limit=5)["campaigns"]
    assert c["type"] == "bar" and c["x_key"] == "campaign" and len(c["data"]) == 5
    assert [(r["campaign"], r["revenue"]) for r in c["data"]] == [(e["id"], e["revenue"]) for e in expected]


def test_revenue_and_spend_multi_series():
    c = chart("Compare revenue and spend over time.")
    assert c["type"] == "line" and [s["key"] for s in c["series"]] == ["revenue", "spend"]
    revenue = {p["month"]: p["value"] for p in dt.trend_over_time("revenue")["series"]}
    spend = {p["month"]: p["value"] for p in dt.trend_over_time("spend")["series"]}
    assert {r["month"]: (r["revenue"], r["spend"]) for r in c["data"]} == {m: (revenue[m], spend[m]) for m in revenue}
    # Different units in one chart would mislead: a clarification, not a chart.
    body, calls = ask("Compare revenue and conversions over time.")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 0


def test_invalid_metric():
    body, calls = ask("Chart something for me")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 0
    assert "Which metric would you like me to visualize?" in body["answer"]
    body, _ = ask("Plot monthly ROAS")  # no monthly tool for a ratio: said, not invented
    assert body["chart"] is None and "not ROAS" in body["answer"]
    spec = copy.deepcopy(chart("Plot monthly spend."))
    spec["series"][0]["key"] = "happiness"
    expect_invalid(spec, "unknown metric")


def test_invalid_chart_type():
    body, calls = ask("Make a scatter plot of spend and revenue")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 0
    spec = copy.deepcopy(chart("Plot monthly spend."))
    spec["type"] = "scatter"
    expect_invalid(spec, "unknown chart type")


def test_non_finite_values_rejected():
    base = chart("Plot monthly spend.")
    for bad in (float("nan"), float("inf"), float("-inf"), "12", None, True):
        spec = copy.deepcopy(base)
        spec["data"][3]["spend"] = bad
        expect_invalid(spec, "non-numeric or non-finite")


def test_months_stay_chronological():
    c = chart("Show monthly revenue as a line chart.")
    months = [r["month"] for r in c["data"]]
    assert months[0] == "2024-01" and months == sorted(months, key=lambda m: (int(m[:4]), int(m[5:])))
    spec = copy.deepcopy(c)
    spec["data"][0], spec["data"][1] = spec["data"][1], spec["data"][0]
    expect_invalid(spec, "chronological")


def test_top_n_is_capped():
    c = chart("Bar chart of top 50 campaigns by profit")
    assert len(c["data"]) == cb.MAX_CATEGORIES
    spec = copy.deepcopy(c)
    spec["data"].append({"campaign": "EXTRA", "profit": 1.0})
    expect_invalid(spec, "rows")


def test_chart_requests_make_no_groq_calls():
    questions = ["Show monthly revenue as a line chart.", "Plot monthly spend.", "Show revenue by platform as a bar chart.",
                 "Create a bar chart of ROAS by platform.", "Show platform revenue share as a pie chart.",
                 "Bar chart of the top 5 campaigns by revenue.", "Compare revenue and spend over time.",
                 "Chart something for me", "Pie chart of ROAS by platform"]
    for q in questions:
        _, calls = ask(q)
        assert calls == 0, q
    # "Show ..." questions keep their DIRECT_DATABASE text answer and carry the same chart.
    body, calls = ask("Show monthly revenue.")
    assert body["route"] == "DIRECT_DATABASE" and body["answer"].startswith("Monthly revenue:") and calls == 0
    assert body["chart"]["type"] == "line" and len(body["chart"]["data"]) == len(dt.trend_over_time("revenue")["series"])


def test_direct_database_question_still_works():
    body, calls = ask("What is total revenue?")
    assert body["route"] == "DIRECT_DATABASE" and body["chart"] is None and calls == 0
    assert f"${dt.get_totals()['revenue']:,.2f}" in body["answer"]
    body, _ = ask("Show me total revenue")  # "show" alone is not a chart request
    assert body["route"] == "DIRECT_DATABASE" and body["chart"] is None


def test_llm_required_explanation_still_works():
    respond = fake_groq.reply("TikTok has the highest ROAS of the platforms shown.")
    body, calls = ask("Why is TikTok performing better than other platforms?", respond=respond)
    assert body["route"] == "LLM_REQUIRED" and body["chart"] is None and calls == 1
    assert "TikTok" in body["answer"]


def test_executable_content_rejected():
    base = chart("Show revenue by platform as a bar chart.")
    attacks = [("title", "<script>alert(1)</script>"), ("title", "javascript:alert(1)"),
               ("x_label", "<img src=x onerror=alert(1)>"), ("title", "Revenue\x00")]
    for key, value in attacks:
        spec = copy.deepcopy(base)
        spec[key] = value
        expect_invalid(spec, "unsafe")
    spec = copy.deepcopy(base)
    spec["data"][0]["name"] = "<b onmouseover=alert(1)>x</b>"
    expect_invalid(spec, "unsafe")
    spec = copy.deepcopy(base)
    spec["render"] = "function(){fetch('//evil')}"
    expect_invalid(spec, "fields must be exactly")
    spec = copy.deepcopy(base)
    spec["series"][0]["formatter"] = "() => 1"
    expect_invalid(spec, "exactly key, label, unit")
    spec = copy.deepcopy(base)
    spec["data"][0]["onclick"] = 1.0
    expect_invalid(spec, "exactly the x key")


def test_controlled_response_when_data_fails():
    with patch.object(qr, "_execute", side_effect=qr.DataUnavailableError()):
        body, calls = ask("Plot monthly spend.")
    assert body["route"] == "DATA_ERROR" and body["chart"] is None and calls == 0
    assert "Traceback" not in body["answer"]


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} CHART TESTS PASSED ===")
