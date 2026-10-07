"""
Tests for charts in the AI Analyst chat: the local model's plan asks for a chart type, DuckDB supplies
every value, and chart_builder.validate_chart() checks every spec (data + metadata, never code).

The model is the fake llama-server (fake_llama.py) with a scripted plan per question; every chart value
is compared with the data tools' own results. (Ported from the router chart-path version: the specs,
value checks and security checks are unchanged; each request now costs one planning call.)

Run:  python test_charts.py      (or: pytest test_charts.py)
"""
import copy
import html
import logging
from unittest.mock import patch

import fake_llama  # noqa: F401  (first: points the app's model URL at a closed port)

from fastapi.testclient import TestClient

import analytics_executor as ax
import chart_builder as cb
import data_tools as dt
import main
import query_planner as qp
from fake_llama import Model

client = TestClient(main.app)

PLANS = {
    "Show monthly revenue as a line chart.": {"intent": "trend", "metrics": ["revenue"], "grain": "month", "visualization": "line"},
    "Plot monthly spend.": {"intent": "trend", "metrics": ["spend"], "grain": "month", "visualization": "line"},
    "Show revenue by platform as a bar chart.": {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform",
                                                 "visualization": "bar"},
    "Create a bar chart of ROAS by platform.": {"intent": "breakdown", "metrics": ["roas"], "dimension": "platform",
                                                "visualization": "bar"},
    "Show platform revenue share as a pie chart.": {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform",
                                                    "visualization": "pie"},
    "Pie chart of ROAS by platform": {"intent": "breakdown", "metrics": ["roas"], "dimension": "platform",
                                      "visualization": "pie"},
    "Chart revenue by platform": {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform",
                                  "visualization": "bar"},
    "Bar chart of the top 5 campaigns by revenue.": {"intent": "campaign_list", "metrics": ["revenue"], "limit": 5,
                                                     "visualization": "bar"},
    "Compare revenue and spend over time.": {"intent": "trend", "metrics": ["revenue", "spend"], "grain": "month"},
    "Compare revenue and conversions over time.": {"intent": "trend", "metrics": ["revenue", "conversions"], "grain": "month"},
    "Chart something for me": {"intent": "not_answerable", "reason": "ambiguous"},
    "Make a heatmap chart of spend by platform": {"intent": "breakdown", "metrics": ["spend"], "dimension": "platform",
                                                  "visualization": "heatmap"},
    "Bar chart of top 50 campaigns by profit": {"intent": "campaign_list", "metrics": ["profit"], "limit": 50,
                                                "visualization": "bar"},
    "Show monthly revenue.": {"intent": "trend", "metrics": ["revenue"], "grain": "month"},
    "What is total revenue?": {"intent": "total", "metrics": ["revenue"]},
    "Show me total revenue": {"intent": "total", "metrics": ["revenue"]},
    "TikTok performing better than other platforms": {"intent": "breakdown", "metrics": ["roas"], "dimension": "platform"},
}


def ask(question, filters=None):
    with Model(PLANS) as m:
        r = client.post("/api/chat", json={"messages": [{"role": "user", "content": question}],
                                           "filters": filters or {}})
    assert r.status_code == 200, r.text
    return r.json(), m.calls


def chart(question, filters=None):
    body, calls = ask(question, filters)
    assert body["route"] == "ANALYSIS" and body["chart"], body
    assert calls == 1
    return body["chart"]


def complete(series):
    """Monthly points of complete months: the partial last month (data ends 2026-01-30) is never charted."""
    return [p for p in series if p["month"] != "2026-01"]


def expect_invalid(spec, fragment=""):
    try:
        cb.validate_chart(spec)
    except cb.ChartSpecError as e:
        assert fragment in str(e), e
        return
    raise AssertionError(f"accepted: {spec}")


def test_monthly_revenue_line_chart():
    c = chart("Show monthly revenue as a line chart.")
    series = complete(dt.trend_over_time("revenue")["series"])
    assert c["type"] == "line" and c["x_key"] == "month"
    assert c["series"] == [{"key": "revenue", "label": "Revenue", "unit": "$"}]
    assert [(r["month"], r["revenue"]) for r in c["data"]] == [(p["month"], p["value"]) for p in series]


def test_monthly_spend_line_chart():
    c = chart("Plot monthly spend.")
    assert c["type"] == "line" and c["series"][0]["key"] == "spend"
    assert [r["spend"] for r in c["data"]] == [p["value"] for p in complete(dt.trend_over_time("spend")["series"])]


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
    # A ratio can't be a pie: drawn as a bar chart, and the answer says why.
    body, calls = ask("Pie chart of ROAS by platform")
    assert body["chart"]["type"] == "bar" and calls == 1 and "pie chart would mislead" in body["answer"]
    assert chart("Chart revenue by platform")["type"] == "bar"


def test_top_5_campaigns_by_revenue():
    c = chart("Bar chart of the top 5 campaigns by revenue.")
    expected = dt.filter_campaigns([], sort_by="revenue", order="desc", limit=5)["campaigns"]
    assert c["type"] == "bar" and c["x_key"] == "campaign" and len(c["data"]) == 5
    assert [(r["campaign"], r["revenue"]) for r in c["data"]] == [(e["id"], e["revenue"]) for e in expected]


def test_revenue_and_spend_multi_series():
    c = chart("Compare revenue and spend over time.")
    assert c["type"] == "line" and [s["key"] for s in c["series"]] == ["revenue", "spend"]
    revenue = {p["month"]: p["value"] for p in complete(dt.trend_over_time("revenue")["series"])}
    spend = {p["month"]: p["value"] for p in complete(dt.trend_over_time("spend")["series"])}
    assert {r["month"]: (r["revenue"], r["spend"]) for r in c["data"]} == {m: (revenue[m], spend[m]) for m in revenue}
    # Different units never share one chart.
    body, calls = ask("Compare revenue and conversions over time.")
    charts = body.get("charts") or ([body["chart"]] if body["chart"] else [])
    assert calls == 1 and all(len({s["unit"] for s in c["series"]}) == 1 for c in charts)


def test_invalid_metric():
    # A request the model can't map gets the standard clarification and no chart.
    body, calls = ask("Chart something for me")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 1
    assert body["answer"] == html.escape(qp.FAILED)
    spec = copy.deepcopy(chart("Plot monthly spend."))
    spec["series"][0]["key"] = "happiness"
    expect_invalid(spec, "unknown metric")


def test_invalid_chart_type():
    # A chart type outside the plan contract is rejected by validate_plan(): nothing runs.
    body, calls = ask("Make a heatmap chart of spend by platform")
    assert body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None and calls == 1
    spec = copy.deepcopy(chart("Plot monthly spend."))
    spec["type"] = "heatmap"
    expect_invalid(spec, "unknown chart type")
    spec["type"] = "scatter"  # a known type, but not for a monthly series
    expect_invalid(spec, "not months")


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


def test_chart_requests_cost_one_local_planning_call():
    for q in ("Show monthly revenue as a line chart.", "Plot monthly spend.", "Show revenue by platform as a bar chart.",
              "Create a bar chart of ROAS by platform.", "Show platform revenue share as a pie chart.",
              "Bar chart of the top 5 campaigns by revenue.", "Compare revenue and spend over time.",
              "Pie chart of ROAS by platform"):
        _, calls = ask(q)
        assert calls == 1, q
    # "Show ..." answers that are a series carry the same chart.
    body, calls = ask("Show monthly revenue.")
    assert body["route"] == "ANALYSIS" and calls == 1 and body["chart"]["type"] == "line"
    assert len(body["chart"]["data"]) == len(complete(dt.trend_over_time("revenue")["series"]))


def test_plain_question_has_no_chart():
    body, calls = ask("What is total revenue?")
    assert body["route"] == "ANALYSIS" and body["chart"] is None and calls == 1
    assert f"${dt.get_totals()['revenue']:,.2f}" in body["answer"]
    body, _ = ask("Show me total revenue")  # "show" alone is not a chart request for a single figure
    assert body["route"] == "ANALYSIS" and body["chart"] is None


def test_explanation_has_no_chart():
    body, calls = ask("Why is TikTok performing better than other platforms?")
    assert body["route"] == "EXPLANATION" and body["chart"] is None and calls == 1
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
    with patch.object(ax, "execute", side_effect=ax.DataUnavailableError()):
        body, calls = ask("Plot monthly spend.")
    assert body["route"] == "DATA_ERROR" and body["chart"] is None and calls == 1
    assert "Traceback" not in body["answer"]


if __name__ == "__main__":
    logging.getLogger("marketingiq.analytics").setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} CHART TESTS PASSED ===")
