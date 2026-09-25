"""
Tests for AI Analyst conversation memory (conversation.py) and general data-driven charts.

Groq runs on a fake transport (fake_groq.py) that counts requests; a test either supplies the
planner's JSON reply or fails if Groq is reached. Every chart value is checked against the data
tools' own DuckDB results.

Run:  python test_conversation.py      (or: pytest test_conversation.py)
"""
import json
import logging
import uuid

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)

from fastapi.testclient import TestClient

import conversation
import data_tools as dt
import main
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)


def plan(**fields):
    base = {"intent": "total", "metrics": ["revenue"], "dimension": "none", "entities": [], "filters": [],
            "months": [], "years": [], "grain": "none", "conditions": [], "order": "none", "limit": 0,
            "visualization": "none", "reason": "none"}
    base.update(fields)
    return json.dumps(base)


JANUARY = plan(intent="period_comparison", months=[1], grain="year")


def refuse(request):
    raise AssertionError("Groq must not be called")


class Chat:
    """One browser conversation: a fresh session id, like the dashboard generates."""

    def __init__(self):
        self.session_id = uuid.uuid4().hex

    def ask(self, question, reply=None, session=True):
        fake = fake_groq.install(main.llm_adapter, fake_groq.reply(reply) if reply is not None else refuse)
        try:
            body = {"messages": [{"role": "user", "content": question}], "filters": {}}
            if session:
                body["session_id"] = self.session_id
            r = client.post("/api/chat", json=body)
        finally:
            fake_groq.block_real_calls(main.llm_adapter)
        assert r.status_code == 200, r.text
        self.last_requests = fake.requests
        return r.json(), len(fake.requests)

    @property
    def state(self):
        return main.sessions.get(self.session_id)


def platform(metric, **filters):
    return {r["name"]: r[metric] for r in dt.rank_dimension("platform", metric, limit=100, filters=filters)["results"]}


def chart_values(chart, key=None):
    key = key or chart["series"][0]["key"]
    return {row[chart["x_key"]]: row[key] for row in chart["data"]}


def month_total(metric, year, month, **filters):
    return dt.get_totals({**filters, "month_from": f"{year}-{month:02d}", "month_to": f"{year}-{month:02d}"})[metric]


# --- the nine required conversations ---------------------------------------------------

def test_conversation1_turn_this_into_a_chart():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("Turn this into a chart.")
    assert calls == 0 and body["route"] == "CHART" and body["chart"]["type"] == "bar"
    assert chart_values(body["chart"]) == platform("revenue")


def test_conversation2_make_it_a_bar_chart_then_pie_then_spend():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("Make it a bar chart.")
    assert calls == 0 and body["chart"]["type"] == "bar" and chart_values(body["chart"]) == platform("revenue")
    body, calls = c.ask("Now make it a pie chart.")  # revenue is additive: a pie is valid
    assert calls == 0 and body["chart"]["type"] == "pie" and chart_values(body["chart"]) == platform("revenue")
    body, calls = c.ask("Now show spend instead.")  # same dimension, new metric, still charted
    assert calls == 0 and body["chart"]["series"][0]["key"] == "spend" and chart_values(body["chart"]) == platform("spend")
    body, calls = c.ask("Make it a pie chart of ROAS")  # ROAS isn't additive: explained, shown as bars
    assert calls == 0 and body["chart"]["type"] == "bar" and "pie chart would mislead" in body["answer"]
    assert chart_values(body["chart"]) == platform("roas")


def test_conversation3_january_then_february():
    c = Chat()
    body, calls = c.ask("Is January revenue increasing over time?", JANUARY)
    assert calls == 1 and "January 2024" in body["answer"]
    body, calls = c.ask("What about February?")
    assert calls == 0 and "February Revenue by year" in body["answer"] and "January" not in body["answer"]
    for year in (2024, 2025):
        assert f"February {year}: {conversation.qp._fmt('revenue', month_total('revenue', year, 2))}" in body["answer"]


def test_conversation4_same_entities_new_metric():
    c = Chat()
    c.ask("Compare Google Ads and TikTok revenue.")
    body, calls = c.ask("Now compare spend.")
    assert calls == 0 and body["route"] == "DIRECT_DATABASE"
    spend = {r["name"]: r["spend"] for r in dt.compare_entities("platform", ["Google Ads", "TikTok"])["results"]}
    for name, value in spend.items():
        assert f"{name}: Spend {conversation.qp._fmt('spend', value)}" in body["answer"]
    assert "revenue" not in body["answer"].lower()


def test_conversation5_same_trend_other_entity():
    c = Chat()
    c.ask("Show monthly revenue for TikTok.")
    body, calls = c.ask("Now Google Ads.")
    assert calls == 0 and body["chart"]["type"] == "line"  # the first answer was charted: kept
    values = chart_values(body["chart"])
    for month in ("2024-01", "2025-06", "2025-12"):
        assert values[month] == month_total("revenue", int(month[:4]), int(month[5:]), platform="Google Ads")
    assert "platform = Google Ads" in body["answer"]


def test_conversation6_february_chart_not_january():
    c = Chat()
    c.ask("Show January revenue across years.", JANUARY)
    c.ask("What about February?")
    body, calls = c.ask("Turn that into a chart.")
    assert calls == 0 and body["route"] == "CHART" and body["chart"]["type"] == "bar"
    assert body["chart"]["title"] == "February Revenue by Year"
    assert chart_values(body["chart"]) == {f"February {y}": month_total("revenue", y, 2) for y in (2024, 2025)}


def test_conversation7_fresh_session_has_nothing_to_chart():
    body, calls = Chat().ask("Turn this into a chart.")
    assert calls == 0 and body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None
    assert "What would you like me to chart?" in body["answer"]


def test_conversation8_greeting():
    c = Chat()
    for text in ("Hi", "Hello!", "Thanks", "Great", "Who are you?", "What can you do?"):
        body, calls = c.ask(text)
        assert calls == 0 and body["route"] == "CONVERSATION" and body["chart"] is None, text
    assert c.state.plan is None


def test_conversation9_personal_topic_redirect_without_memory_pollution():
    c = Chat()
    c.ask("Show revenue by platform.")
    remembered = c.state.plan
    body, calls = c.ask("Can you give me marriage advice?")
    assert calls == 0 and body["route"] == "UNSUPPORTED" and "campaign dataset" in body["answer"]
    state = c.state
    assert state.plan is remembered and "marriage" not in json.dumps([state.question, list(state.turns)])
    body, _ = c.ask("Turn this into a chart.")  # the analysis context survived the detour
    assert chart_values(body["chart"]) == platform("revenue")


# --- more follow-ups ---------------------------------------------------------------------

def test_focus_then_compare_with_named_entity():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("What about TikTok?")
    tiktok = platform("revenue")["TikTok"]
    assert calls == 0 and f"Revenue: {conversation.qp._fmt('revenue', tiktok)}" in body["answer"]
    assert "platform = TikTok" in body["answer"]
    body, calls = c.ask("Compare it with Google Ads.")
    assert calls == 0
    for name, value in platform("revenue").items():
        if name in ("TikTok", "Google Ads"):
            assert f"{name}: Revenue {conversation.qp._fmt('revenue', value)}" in body["answer"]


def test_why_follow_up_uses_the_grounded_explanation_path_once():
    c = Chat()
    c.ask("Which platform has the highest revenue?")
    body, calls = c.ask("Why?", reply="Google Ads has the highest revenue of the platforms shown.")
    assert calls == 1 and body["route"] == "LLM_REQUIRED"
    sent = json.loads(c.last_requests[0].content)
    assert "revenue by platform" in sent["messages"][1]["content"].lower()  # "why" was resolved, not guessed
    body, calls = c.ask("Visualize it")  # the remembered analysis is still what "it" means
    assert calls == 0 and chart_values(body["chart"]) == platform("revenue")
    assert Chat().ask("Why?")[0]["route"] == "NEEDS_CLARIFICATION"


def test_unresolvable_follow_up_goes_to_planner_with_plan_not_transcript():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("And what does that look like overall?", reply=plan(intent="total"))
    assert calls == 1
    prompt = json.loads(c.last_requests[0].content)["messages"][1]["content"]
    assert '"dimension": "platform"' in prompt and "Show revenue by platform" not in prompt


def test_no_automatic_chart_for_ordinary_questions():
    body, calls = Chat().ask("Which platform has the highest revenue?")
    assert calls == 0 and body["chart"] is None


def test_stateless_requests_keep_the_old_contract():
    history = [{"role": "user", "content": "Show revenue by platform."}, {"role": "assistant", "content": "..."}]
    r = client.post("/api/chat", json={"messages": [*history, {"role": "user", "content": "Turn this into a chart."}],
                                       "filters": {}}).json()
    assert r["route"] == "NEEDS_CLARIFICATION" and "complete question" in r["answer"]
    bad = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hi"}], "session_id": "x"}).json()
    assert bad["route"] == "UNSUPPORTED"  # an invalid id is ignored: stateless


def test_memory_is_bounded():
    now = [0.0]
    store = conversation.SessionStore(max_sessions=3, ttl_s=100, clock=lambda: now[0])
    ids = [uuid.uuid4().hex for _ in range(5)]
    for i in ids:
        store.get(i)
    assert len(store) == 3 and ids[0] not in store._sessions and ids[-1] in store._sessions
    now[0] = 1000
    store.get(ids[0])
    assert len(store) == 1  # expired sessions are dropped
    s = store.get(ids[0])
    for i in range(50):
        s.turns.append({"route": "X", "kind": "none"})
    assert len(s.turns) == conversation.MAX_TURNS
    assert store.get("not a valid id!") is None and store.get("a" * 65) is None


# --- general data-driven charts (not limited to dashboard visuals) --------------------------------

def _chart(question, reply=None, calls_expected=0):
    body, calls = Chat().ask(question, reply)
    assert calls == calls_expected and body["chart"], (question, body)
    return body["chart"]


def _by(dimension, metric, **filters):
    return {r["name"]: r[metric] for r in dt.rank_dimension(dimension, metric, limit=100, filters=filters)["results"]}


def test_chart01_revenue_by_device():
    assert chart_values(_chart("Chart revenue by device.")) == _by("device", "revenue")


def test_chart02_spend_by_objective():
    assert chart_values(_chart("Plot spend by objective.")) == _by("objective", "spend")


def test_chart03_conversions_by_industry_vertical():
    assert chart_values(_chart("Show conversions by industry vertical as a chart.")) == _by("vertical", "conversions")


def test_chart04_monthly_revenue_for_tiktok():
    c = _chart("Show monthly revenue for TikTok as a line chart.")
    assert c["type"] == "line"
    assert chart_values(c) == {p["month"]: p["value"] for p in dt.trend_over_time("revenue", {"platform": "TikTok"})["series"]}


def test_chart05_google_ads_vs_tiktok_spend():
    c = _chart("Compare Google Ads and TikTok spend as a bar chart.")
    assert c["type"] == "bar"
    assert chart_values(c) == {r["name"]: r["spend"] for r in dt.compare_entities("platform", ["Google Ads", "TikTok"])["results"]}


def test_chart06_january_revenue_across_years():
    c = _chart("Chart January revenue across years.", plan(intent="period_comparison", months=[1], grain="year",
                                                           visualization="bar"), calls_expected=1)
    assert chart_values(c) == {f"January {y}": month_total("revenue", y, 1) for y in (2024, 2025)}


def test_chart07_profit_by_budget_tier():
    assert chart_values(_chart("Plot profit by budget tier.")) == _by("budget_tier", "profit")


def test_chart08_top_10_campaigns_by_revenue():
    c = _chart("Chart the top 10 campaigns by revenue.")
    rows = dt.filter_campaigns([], sort_by="revenue", order="desc", limit=10)["campaigns"]
    assert chart_values(c) == {r["id"]: r["revenue"] for r in rows}


def test_chart09_ctr_by_platform():
    assert chart_values(_chart("Visualize CTR by platform.")) == _by("platform", "ctr")


def test_chart10_previous_result_to_chart():
    c = Chat()
    c.ask("Which objective has the lowest CPA?")
    body, calls = c.ask("Turn this into a graph.")
    assert calls == 0 and chart_values(body["chart"]) == _by("objective", "cpa")


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} CONVERSATION TESTS PASSED ===")
