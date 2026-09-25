"""
Tests for general analytics (schema catalog, semantic parser, analytics engine) and advanced charts.

Groq runs on a fake transport that fails the test if reached, unless a test supplies the planner's
reply. Every chart value is checked against the data layer's own DuckDB results.

Run:  python test_general_analytics.py      (or: pytest test_general_analytics.py)
"""
import json
import logging
import uuid

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)
import numpy as np

from fastapi.testclient import TestClient

import chart_builder as cb
import data_tools as dt
import main
import query_planner as qp
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)


def refuse(request):
    raise AssertionError("Groq must not be called")


class Chat:
    def __init__(self):
        self.session_id = uuid.uuid4().hex

    def ask(self, question, reply=None):
        fake = fake_groq.install(main.llm_adapter, fake_groq.reply(reply) if reply is not None else refuse)
        try:
            r = client.post("/api/chat", json={"messages": [{"role": "user", "content": question}], "filters": {},
                                               "session_id": self.session_id})
        finally:
            fake_groq.block_real_calls(main.llm_adapter)
        assert r.status_code == 200, r.text
        return r.json(), len(fake.requests)


def ask(question, reply=None):
    return Chat().ask(question, reply)


def charts(body):
    return body.get("charts") or ([body["chart"]] if body["chart"] else [])


def series_values(chart, label=None):
    """{x: value} for the series with this label (default: the only/first series)."""
    s = next(s for s in chart["series"] if label is None or s["label"] == label)
    return {row[chart["x_key"]]: row[s["key"]] for row in chart["data"]}


def by(dimension, metric, **filters):
    return {r["group"]: r[metric] for r in dt.metric_table(filters, group_by=dimension)}


def monthly(metric, **filters):
    return {r["group"]: r[metric] for r in dt.metric_table(filters, group_by="month")}


COMPLETE = [m for m in monthly("revenue") if m != "2026-01"]  # January 2026 is partial in the data


# --- general questions and charts -------------------------------------------------------------

def test01_revenue_by_device():
    body, calls = ask("Show revenue by device.")
    assert calls == 0 and body["route"] == "DIRECT_DATABASE"
    for name, value in by("device", "revenue").items():
        assert f"{name}: ${value:,.2f}" in body["answer"]


def test02_visualize_conversions_by_industry_vertical():
    body, calls = ask("Visualize conversions by industry vertical.")
    assert calls == 0 and series_values(body["chart"]) == by("vertical", "conversions")


def test03_compare_two_platforms_monthly_revenue():
    body, calls = ask("Compare Google Ads and TikTok monthly revenue.")
    assert calls == 0 and body["route"] == "DIRECT_DATABASE"
    g, t = monthly("revenue", platform="Google Ads"), monthly("revenue", platform="TikTok")
    assert f"Google Ads revenue: ${g['2024-01']:,.2f} in 2024-01" in body["answer"]
    assert f"TikTok revenue: ${t['2024-01']:,.2f} in 2024-01" in body["answer"]


def test04_compare_revenue_and_spend_across_platforms_then_visualize():
    c = Chat()
    body, calls = c.ask("Compare revenue and spend across platforms.")
    assert calls == 0 and body["chart"] is None
    rev, spend = by("platform", "revenue"), by("platform", "spend")
    assert f"Google Ads: ${rev['Google Ads']:,.2f}, Spend ${spend['Google Ads']:,.2f}" in body["answer"]
    body, calls = c.ask("Visualize it")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "bar" and [s["key"] for s in chart["series"]] == ["revenue", "spend"]
    assert series_values(chart, "Revenue") == rev and series_values(chart, "Spend") == spend  # grouped, one unit


def test05_relationship_between_spend_and_revenue():
    body, calls = ask("Show the relationship between spend and revenue.")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "scatter" and [s["key"] for s in chart["series"]] == ["spend", "revenue"]
    df = dt.get_dataframe()
    r = round(float(np.corrcoef(df["ad_spend"], df["revenue"])[0, 1]), 3)
    assert f"r = {r:.3f}" in body["answer"] and "doesn&#x27;t show that one causes the other" in body["answer"]
    rows = df.set_index("campaign_id")
    for p in chart["data"][:25]:
        assert p["spend"] == round(float(rows.loc[p["campaign"], "ad_spend"]), 2)
        assert p["revenue"] == round(float(rows.loc[p["campaign"], "revenue"]), 2)
    assert len(chart["data"]) <= cb.MAX_SCATTER_POINTS


def test06_top_10_campaigns_by_revenue_chart():
    body, calls = ask("Top 10 campaigns by revenue as a chart.")
    rows = dt.filter_campaigns([], sort_by="revenue", order="desc", limit=10)["campaigns"]
    assert calls == 0 and series_values(body["chart"]) == {r["id"]: r["revenue"] for r in rows}


# --- conversation memory ---------------------------------------------------------------------

def test07_chain_revenue_chart_spend_tiktok_monthly():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("Make this a chart.")
    assert calls == 0 and series_values(body["chart"]) == by("platform", "revenue")
    body, calls = c.ask("Now spend.")
    assert calls == 0 and series_values(body["chart"]) == by("platform", "spend")
    body, calls = c.ask("Only TikTok.")
    assert calls == 0 and f"Spend: ${dt.get_totals({'platform': 'TikTok'})['spend']:,.2f}" in body["answer"]
    body, calls = c.ask("Monthly.")
    tiktok = monthly("spend", platform="TikTok")
    assert calls == 0 and f"${tiktok['2024-01']:,.2f} in 2024-01" in body["answer"] and "platform = TikTok" in body["answer"]


def test08_chain_compare_conversions_over_time_chart():
    c = Chat()
    c.ask("Compare Google Ads and TikTok revenue.")
    body, calls = c.ask("Now conversions.")
    assert calls == 0 and "Conversions" in body["answer"]
    body, calls = c.ask("Show this over time.")
    assert calls == 0 and "TikTok conversions:" in body["answer"]
    body, calls = c.ask("Make it a chart.")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "line" and [s["label"] for s in chart["series"]] == ["Google Ads", "TikTok"]
    for name in ("Google Ads", "TikTok"):
        expected = monthly("conversions", platform=name)
        assert series_values(chart, name) == {m: expected[m] for m in COMPLETE}


def test09_multiple_entities_over_time_is_multi_series_line():
    body, calls = ask("Chart monthly revenue for Google Ads, TikTok and Facebook.")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "line" and len(chart["series"]) == 3
    for s in chart["series"]:
        expected = monthly("revenue", platform=s["label"])
        assert series_values(chart, s["label"]) == {m: expected[m] for m in COMPLETE}


def test10_multiple_compatible_metrics_grouped():
    body, calls = ask("Chart revenue and profit by device.")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "bar" and len(chart["series"]) == 2 and "charts" not in body
    assert series_values(chart, "Profit") == by("device", "profit")


def test11_platform_by_device_grouped_stacked_percent():
    c = Chat()
    body, calls = c.ask("Chart conversions across platform and device.")
    chart = body["chart"]
    assert calls == 0 and chart["type"] == "bar" and chart["x_label"] == "Platform"
    assert {s["label"] for s in chart["series"]} == {"Desktop", "Mobile", "Tablet"}
    for device in ("Desktop", "Mobile", "Tablet"):
        assert series_values(chart, device) == by("platform", "conversions", device=device)
    body, _ = c.ask("Use a stacked chart instead")
    assert body["chart"]["type"] == "stacked_bar"
    body, _ = c.ask("Make it 100% stacked")
    assert body["chart"]["type"] == "stacked_percent"  # values stay absolute; shares are drawn from them
    assert series_values(body["chart"], "Mobile") == by("platform", "conversions", device="Mobile")
    body, _ = c.ask("Now show ROAS")  # a ratio can't be stacked: grouped bars, explained
    assert body["chart"]["type"] == "bar" and "Stacking needs" in body["answer"]


# --- safety -------------------------------------------------------------------------------------

def test12_incompatible_units_are_separate_charts():
    body, calls = ask("Compare revenue, ROAS and CTR for every platform as a chart.")
    cs = charts(body)
    assert calls == 0 and len(cs) == 3 and body["chart"] == cs[0]
    assert [c["series"][0]["unit"] for c in cs] == ["$", "x", "%"]
    assert all(len({s["unit"] for s in c["series"]}) == 1 for c in cs)
    assert "different units" in body["answer"]
    assert series_values(cs[1]) == by("platform", "roas") and series_values(cs[2]) == by("platform", "ctr")
    mixed = {"type": "bar", "title": "Mixed", "x_key": "name", "x_label": "Platform",
             "series": [{"key": "revenue", "label": "Revenue", "unit": "$"}, {"key": "ctr", "label": "CTR", "unit": "%"}],
             "data": [{"name": "A", "revenue": 1.0, "ctr": 2.0}]}
    for spec, fragment in ((mixed, "different units"), ({**mixed, "type": "stacked_bar", "series": [
            {"key": "roas", "label": "ROAS", "unit": "x"}], "data": [{"name": "A", "roas": 1.0}]}, "additive"),
            ({**mixed, "type": "pie", "series": [{"key": "ctr", "label": "CTR", "unit": "%"}],
              "data": [{"name": "A", "ctr": 1.0}]}, "additive")):
        try:
            cb.validate_chart(spec)
            raise AssertionError("accepted")
        except cb.ChartSpecError as e:
            assert fragment in str(e)


def test13_fresh_chart_this_asks():
    body, calls = ask("Chart this")
    assert calls == 0 and body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None


def test14_unknown_dataset_field():
    for q in ("Show revenue by country.", "Show customer lifetime value by platform."):
        body, calls = ask(q)
        assert calls == 0 and body["route"] == "UNSUPPORTED" and "not available" in body["answer"], q


def test15_catalog_fields_beyond_the_dashboard():
    body, calls = ask("Show bounce rate by day of week as a chart")
    chart = body["chart"]
    assert calls == 0 and chart["series"][0]["unit"] == "%" and series_values(chart) == by("day_of_week", "bounce_rate")


def test16_semantic_planner_uses_the_extended_plan():
    plan = {"intent": "breakdown", "metrics": ["conversions"], "dimension": "placement", "dimension2": "device",
            "entities": [], "exclude": [], "filters": [], "months": [], "years": [], "grain": "none",
            "conditions": [], "order": "desc", "limit": 0, "visualization": "stacked_bar", "reason": "none"}
    body, calls = ask("Chart how gadget types split within each ad slot for sign-ups", json.dumps(plan))
    chart = body["chart"]
    assert calls == 1 and chart["type"] == "stacked_bar"
    assert series_values(chart, "Mobile") == by("placement", "conversions", device="Mobile")
    for bad in ({**plan, "dimension2": "country"}, {**plan, "dimension2": "placement"},
                {**plan, "exclude": ["DROP TABLE"]}, {**plan, "visualization": "heatmap"}):
        try:
            qp.validate_plan(json.dumps(bad))
            raise AssertionError(f"accepted {bad}")
        except qp.PlanError:
            pass


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} GENERAL ANALYTICS TESTS PASSED ===")
