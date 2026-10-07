"""
Tests for the final local architecture (analyst.py): Qwen plans, validate_plan() gates, DuckDB computes,
findings / recommendations / charts are deterministic. No cloud LLM, no API key, no query router.

The model is the fake llama-server (fake_llama.py, real HTTP) with a scripted plan per question.
Spies count model calls, validate_plan() / execute_plan() and DuckDB queries; every figure is checked
against independent data-tool queries. eval_frozen.jsonl is not used.

Run:  python test_analyst.py      (or: pytest test_analyst.py)
"""
import html
import json
import logging
import os
import re
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import fake_llama  # noqa: F401  (first: points the app's model URL at a closed port)

from fastapi.testclient import TestClient

import analyst
import campaign_repository as cr
import chart_builder as cb
import data_tools as dt
import findings_engine
import grounding
import local_planner
import main
import query_planner as qp
import recommendation_engine
from fake_llama import Model

client = TestClient(main.app)
BACKEND = Path(__file__).resolve().parent
RUNTIME = ["main", "analyst", "local_planner", "query_planner", "analytics_executor", "analytics_engine",
           "findings_engine", "recommendation_engine", "campaign_lookup", "conversation", "chart_builder",
           "schema_catalog", "data_tools", "campaign_repository", "grounding"]

REV_PLATFORM = {"intent": "breakdown", "metrics": ["revenue"], "dimension": "platform"}
ROAS_PLATFORM = {"intent": "breakdown", "metrics": ["roas"], "dimension": "platform"}
CPA_DEVICE = {"intent": "breakdown", "metrics": ["cpa"], "dimension": "device"}
LOSS = {"intent": "campaign_list", "metrics": ["profit"], "conditions": [{"metric": "profit", "operator": "<", "value": 0}]}


class Spy:
    """Model + spies on validate_plan / execute_plan (in call order) and DuckDB queries."""

    def __init__(self, respond=fake_llama.refuse, **kw):
        self.model, self.stack, self.order = Model(respond, **kw), ExitStack(), []

    def __enter__(self):
        s = self.stack
        s.enter_context(self.model)
        real_validate, real_execute = qp.validate_plan, qp.execute_plan

        def validate(text):
            p = real_validate(text)
            self.order.append(("validate", id(p)))
            return p

        def execute(plan, filters):
            self.order.append(("execute", id(plan)))
            return real_execute(plan, filters)
        self.validate = s.enter_context(patch.object(qp, "validate_plan", side_effect=validate))
        self.execute = s.enter_context(patch.object(qp, "execute_plan", side_effect=execute))
        self.duckdb = s.enter_context(patch.object(cr.DuckDBCampaignRepository, "_run", autospec=True,
                                                   side_effect=cr.DuckDBCampaignRepository._run))
        return self

    def __exit__(self, *exc):
        self.stack.close()

    @property
    def calls(self):
        return self.model.calls

    def ask(self, question, session_id=None, history=(), filters=None):
        messages = [*history, {"role": "user", "content": question}]
        r = client.post("/api/chat", json={"messages": messages, "filters": filters or {}, "session_id": session_id})
        assert r.status_code == 200, r.text
        return r.json()


def text(body) -> str:
    return html.unescape(body["answer"].replace("<br>", "\n"))


def fmt(metric, value):
    return grounding.fmt(metric, value)


def sid():
    return str(uuid.uuid4())


def one_call_analysis(s, body):
    assert body["route"] == "ANALYSIS", body
    assert s.calls == 1 and s.duckdb.call_count >= 1
    executed = [pid for kind, pid in s.order if kind == "execute"]
    assert executed and all(("validate", pid) in s.order[:s.order.index(("execute", pid))] for pid in executed), \
        "a plan reached execution without validate_plan() returning it first"


# --------------------------------------------------------------------------- 1-8 analytics

def test_01_total():
    with Spy({"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}) as s:
        body = s.ask("What is total revenue?")
        one_call_analysis(s, body)
    assert fmt("revenue", dt.get_totals()["revenue"]) in text(body)


def test_02_breakdown():
    with Spy({"revenue by platform": REV_PLATFORM}) as s:
        body = s.ask("revenue by platform")
        one_call_analysis(s, body)
    for r in dt.rank_dimension("platform", metric="revenue", limit=20)["results"]:
        assert f"{r['name']}: {fmt('revenue', r['revenue'])}" in text(body)


def test_03_comparison():
    plan = {"intent": "compare_entities", "metrics": ["roas"], "dimension": "platform", "entities": ["Google Ads", "TikTok"]}
    with Spy({"compare Google Ads and TikTok": plan}) as s:
        body = s.ask("compare Google Ads and TikTok")
        one_call_analysis(s, body)
    for r in dt.compare_entities("platform", ["Google Ads", "TikTok"])["results"]:
        assert f"{r['name']}: ROAS {fmt('roas', r['roas'])}" in text(body)


def test_04_trend():
    with Spy({"monthly revenue trend": {"intent": "trend", "metrics": ["revenue"], "grain": "month"}}) as s:
        body = s.ask("monthly revenue trend")
        one_call_analysis(s, body)
    first = dt.trend_over_time("revenue")["series"][0]
    assert fmt("revenue", first["value"]) in text(body) and first["month"] in text(body)


def test_05_relationship():
    with Spy({"revenue vs conversions": {"intent": "relationship", "metrics": ["revenue", "conversions"]}}) as s:
        body = s.ask("revenue vs conversions")
        one_call_analysis(s, body)
    r = dt.campaign_points("revenue", "conversions")["correlation"]
    assert f"r = {r:.3f}" in text(body)


def test_06_campaign_list():
    with Spy({"top 5 campaigns by profit": {"intent": "campaign_list", "metrics": ["profit"], "limit": 5}}) as s:
        body = s.ask("top 5 campaigns by profit")
        one_call_analysis(s, body)
    for c in dt.filter_campaigns([], sort_by="profit", order="desc", limit=5)["campaigns"]:
        assert c["id"] in text(body) and fmt("profit", c["profit"]) in text(body)


def test_07_filters():
    plan = {"intent": "total", "metrics": ["revenue"], "filters": [{"dimension": "platform", "value": "TikTok"}],
            "years": [2025]}
    with Spy({"TikTok revenue in 2025": plan}) as s:
        body = s.ask("TikTok revenue in 2025")
        one_call_analysis(s, body)
    t = dt.get_totals({"platform": "TikTok", "month_from": "2025-01", "month_to": "2025-12"})
    assert fmt("revenue", t["revenue"]) in text(body)
    # the dashboard's own filters apply on top of the plan
    with Spy({"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}) as s:
        body = s.ask("What is total revenue?", filters={"device": "Mobile"})
    assert fmt("revenue", dt.get_totals({"device": "Mobile"})["revenue"]) in text(body)


def test_08_chart_from_chart_builder():
    plan = {**REV_PLATFORM, "visualization": "bar"}
    with Spy({"show revenue by platform as a bar chart": plan}) as s, \
            patch.object(cb, "validate_chart", wraps=cb.validate_chart) as validated:
        body = s.ask("show revenue by platform as a bar chart")
        one_call_analysis(s, body)
        assert validated.call_count >= 1
    chart = body["chart"]
    assert chart["type"] == "bar" and cb.validate_chart(chart) == chart
    rows = dt.rank_dimension("platform", metric="revenue", limit=20)["results"]
    assert {d["name"]: d["revenue"] for d in chart["data"]} == {r["name"]: r["revenue"] for r in rows}


def test_08b_model_cannot_supply_chart_code_or_numbers():
    for extra in ({"svg": "<svg/>"}, {"html": "<b>x</b>"}, {"data": [1, 2, 3]}):
        with Spy(lambda q, p: {**REV_PLATFORM, "visualization": "bar", **extra}) as s:
            body = s.ask("show revenue by platform as a bar chart")
            assert body["chart"] is None and s.validate.call_count == 0 and s.duckdb.call_count == 0


# --------------------------------------------------------------------------- 9 follow-ups and memory

def test_09_follow_up_chain_and_memory():
    """"Revenue by platform" -> "Now profit" -> "Why?" -> "What should we improve?"."""
    session = sid()
    plans = {"Revenue by platform": REV_PLATFORM, "Now profit": lambda prev: {**prev, "metrics": ["profit"]}}
    with Spy(plans) as s:
        s.ask("Revenue by platform", session)
        follow = s.ask("Now profit", session)
        assert s.calls == 2 and '"metrics": ["revenue"], "dimension": "platform"' in s.model.prompts[1]
        assert "<|im_start|>assistant\n{" not in s.model.prompts[1]          # no transcript, no earlier answers
        why = s.ask("Why?", session)
        advice = s.ask("What should we improve?", session)
        assert s.calls == 2                                                  # "why" and advice: no model call
    profit = dt.rank_dimension("platform", metric="profit", limit=20)["results"]
    assert follow["route"] == "ANALYSIS" and all(f"{r['name']}: {fmt('profit', r['profit'])}" in text(follow)
                                                 for r in profit)
    assert why["route"] == "EXPLANATION" and f"{profit[0]['name']} has the highest profit" in text(why)
    assert advice["route"] == "RECOMMENDATION" and "Recommended action:" in text(advice)
    assert main.sessions.get(session).plan.metrics == ["profit"]


def test_09b_new_request_overrides_and_failure_keeps_memory():
    session = sid()
    plans = {"revenue by platform": REV_PLATFORM,
             "TikTok revenue in 2025": {"intent": "total", "metrics": ["revenue"],
                                        "filters": [{"dimension": "platform", "value": "TikTok"}], "years": [2025]},
             "Compare impressions across browsers": {"intent": "breakdown", "metrics": ["impressions"],
                                                     "dimension": "browser"}}
    with Spy(plans) as s:
        s.ask("revenue by platform", session)
        s.ask("TikTok revenue in 2025", session)
        assert "Previous analysis" not in s.model.prompts[1]                 # a complete new request
        assert main.sessions.get(session).plan.filters == {"platform": "TikTok"}
        s.ask("Compare impressions across browsers", session)                 # rejected
    assert main.sessions.get(session).plan.filters == {"platform": "TikTok"}  # memory unchanged


def test_09c_reference_without_context():
    with Spy() as s:
        body = s.ask("make it a chart", sid())
        assert s.calls == 0 and body["route"] == "NEEDS_CLARIFICATION"
        body = s.ask("Why?", sid())
        assert s.calls == 0 and text(body) == analyst.NOTHING_TO_EXPLAIN


# --------------------------------------------------------------------------- 10-17 findings and recommendations

def test_10_deterministic_explanation():
    plan = {"intent": "compare_entities", "metrics": ["roas"], "dimension": "platform", "entities": ["TikTok", "Google Ads"]}
    with Spy({"TikTok ROAS lower than Google Ads": plan}) as s:
        body = s.ask("Why is TikTok ROAS lower than Google Ads?")
        assert body["route"] == "EXPLANATION" and s.calls == 1           # one planning call, no second call
    rows = dt.compare_entities("platform", ["TikTok", "Google Ads"])["results"]
    hi, lo = max(rows, key=lambda r: r["roas"]), min(rows, key=lambda r: r["roas"])
    assert f"{hi['name']} has a higher ROAS than {lo['name']}: {fmt('roas', hi['roas'])} vs {fmt('roas', lo['roas'])}." \
        in text(body)
    assert analyst.EXPLAIN_NOTE in text(body)


def test_11_deterministic_recommendation():
    with Spy({"ROAS across platforms": ROAS_PLATFORM}) as s:
        body = s.ask("How can we improve ROAS across platforms?")
        assert body["route"] == "RECOMMENDATION" and s.calls == 1
    out = text(body)
    assert out.index("Finding:") < out.index("Recommended action:")
    assert not re.search(r"\bwill (?:increase|improve|grow|raise)\b|\bguarantee", out)


def test_12_highest_and_lowest():
    rows = dt.rank_dimension("platform", metric="roas", limit=20)["results"]
    f = findings_engine.find(qp.Plan("breakdown", ["roas"], "platform"), {})
    assert (f.highest, f.lowest, f.best, f.worst) == (rows[0]["name"], rows[-1]["name"], rows[0]["name"], rows[-1]["name"])
    assert f"{rows[0]['name']} has the highest ROAS across {len(rows)} platforms at {fmt('roas', rows[0]['roas'])}" \
        in f.statements[0]
    cost = findings_engine.find(qp.Plan("breakdown", ["cpa"], "device"), {})
    cpa = dt.rank_dimension("device", metric="cpa", limit=20)["results"]       # highest first
    assert cost.worst == cpa[0]["name"] and cost.best == cpa[-1]["name"]       # lower CPA is better


def test_13_roas_recommendation_direction():
    rows = dt.rank_dimension("platform", metric="roas", limit=20)["results"]
    best, worst = rows[0]["name"], rows[-1]["name"]
    actions = " ".join(recommendation_engine.recommend(findings_engine.find(qp.Plan("breakdown", ["roas"], "platform"), {})))
    assert f"budget on {best} (the highest ROAS)" in actions and f"for {worst} (the lowest ROAS)" in actions
    assert f"budget on {worst}" not in actions
    # budget advice from a spend/revenue breakdown follows ROAS, not the biggest totals
    spend = recommendation_engine.recommend(findings_engine.find(qp.Plan("breakdown", ["spend", "revenue"], "platform"), {}))
    assert any(f"budget on {best} (the highest ROAS)" in a for a in spend)
    assert not any(f"budget on {worst}" in a for a in spend)


def test_14_cpa_recommendation_direction():
    rows = dt.rank_dimension("device", metric="cpa", limit=20)["results"]
    with Spy({"CPA across devices": CPA_DEVICE}) as s:
        body = s.ask("How can we reduce CPA across devices?")
    out = text(body)
    assert f"Prioritize CPA reduction on {rows[0]['name']} (the highest CPA)" in out
    assert f"Use {rows[-1]['name']} (the lowest CPA) as the benchmark" in out


def test_15_loss_making_means_profit_below_zero():
    revenue_proxy = {"intent": "campaign_list", "metrics": ["revenue"],
                     "conditions": [{"metric": "revenue", "operator": "<", "value": 0}]}
    expected = dt.filter_campaigns([{"field": "profit", "operator": "<", "value": 0}], limit=1)["matched_count"]
    for question in ("loss-making campaigns", "losing campaigns", "campaigns running at a loss",
                     "negative-profit campaigns", "unprofitable campaigns"):
        plan = qp.validate_plan(json.dumps({**qp.plan_to_raw(qp.Plan("total", ["revenue"])), **revenue_proxy}))
        fixed = local_planner.apply_loss_rule(plan, question)
        assert fixed.conditions == [local_planner.LOSS_CONDITION], question
    with Spy(lambda q, p: revenue_proxy) as s:   # the model's wrong proxy is corrected deterministically
        body = s.ask("Which campaigns are running at a loss?")
    assert f"{expected:,} campaigns have Profit < $0.00." in text(body), text(body)
    # a non-loss question keeps its plan untouched
    plain = qp.validate_plan(json.dumps({**qp.plan_to_raw(qp.Plan("total", ["revenue"])), **revenue_proxy}))
    assert local_planner.apply_loss_rule(plain, "campaigns with negative revenue") is plain


def test_16_negative_profit_recommendation():
    expected = dt.filter_campaigns([{"field": "profit", "operator": "<", "value": 0}], limit=1)["matched_count"]
    with Spy({"loss-making campaigns": LOSS}) as s:
        body = s.ask("What should we do about loss-making campaigns?")
        assert s.calls == 1
    out = text(body)
    assert f"{expected:,} campaigns" in out and dt.UNSUCCESSFUL_RULE in out
    assert "Review or pause persistently loss-making campaigns" in out


def test_17_trend_increase_and_decrease():
    for plan, metric in ((qp.Plan("trend", ["revenue"], grain="month"), "revenue"),
                         (qp.Plan("trend", ["spend"], grain="year"), "spend")):
        f = findings_engine.find(plan, {})
        if plan.grain == "year":
            years = [y for y in (2024, 2025)]
            vals = [dt.get_totals({"month_from": f"{y}-01", "month_to": f"{y}-12"})[metric] for y in years]
        else:
            months = [p for p in dt.trend_over_time(metric)["series"] if p["month"] != "2026-01"]
            vals = [months[0]["value"], months[-1]["value"]]
        assert f.direction == ("increase" if vals[-1] > vals[0] else "decrease"), (metric, vals)
        actions = " ".join(recommendation_engine.recommend(f))
        assert ("Investigate the periods contributing to the decline" in actions) == (f.direction == "decrease")
        assert ("Monitor whether the increase" in actions) == (f.direction == "increase")


# --------------------------------------------------------------------------- 18-19 rejection

def test_18_unsupported_dimension_is_rejected():
    with Spy({"Compare impressions across browsers": {"intent": "breakdown", "metrics": ["impressions"],
                                                      "dimension": "browser"}}) as s:
        body = s.ask("Compare impressions across browsers")
        assert body["route"] == "UNSUPPORTED" and s.duckdb.call_count == 0 and s.execute.call_count == 0
    assert text(body).startswith("Browser-level analysis isn't available in the current MarketingIQ dataset.")


def test_19_invalid_plans_never_reach_duckdb():
    outputs = ['{"intent": "total", "metrics": ["revenue"]', "Total revenue is $5.", "[1, 2]",
               ('{"intent":"total","metrics":["revenue"]}', "limit"),                        # cut off
               {"intent": "total", "metrics": ["revenue"], "sql": "SELECT 1"},              # unknown key
               {"intent": "total", "metrics": ["bounce_money"]},                            # unknown metric
               {"intent": "total", "metrics": ["revenue"], "filters": [{"dimension": "platform", "value": "MySpace"}]},
               {"intent": "total", "metrics": ["revenue"], "years": [2031]}]
    for out in outputs:
        for question in ("What is total revenue?", "Why is revenue so low?", "How can we improve revenue?"):
            with Spy(lambda q, p, out=out: out) as s:
                body = s.ask(question)
                assert s.calls == 1 and s.execute.call_count == 0 and s.duckdb.call_count == 0, (out, question, body)
                assert body["route"] in ("NEEDS_CLARIFICATION", "UNSUPPORTED"), (out, body)


def test_19b_model_unavailable_and_timeout():
    with Spy(url="http://127.0.0.1:9", timeout_s=1) as s:
        body = s.ask("What is total revenue?")
        assert body["route"] == "MODEL_UNAVAILABLE" and text(body) == local_planner.UNAVAILABLE
        assert s.duckdb.call_count == 0 and "Traceback" not in body["answer"]
    with Spy(lambda q, p: {"intent": "total", "metrics": ["revenue"]}, delay_s=1.5, timeout_s=0.3) as s:
        t0 = time.perf_counter()
        body = s.ask("What is total revenue?")
        assert time.perf_counter() - t0 < 1.4 and body["route"] == "MODEL_UNAVAILABLE"


def test_19c_execution_failure_is_safe():
    with Spy({"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}) as s, \
            patch.object(qp, "execute_plan", side_effect=KeyError("ctr")):
        body = s.ask("What is total revenue?")
    assert body["route"] == "DATA_ERROR" and text(body) == local_planner.EXECUTION_FAILED


# --------------------------------------------------------------------------- 20-21 side paths

def test_20_small_talk():
    for q, route in (("hello", "CONVERSATION"), ("thanks!", "CONVERSATION"), ("what can you do?", "CONVERSATION"),
                     ("can you give me relationship advice for my marriage?", "UNSUPPORTED")):
        with Spy() as s:
            body = s.ask(q, sid())
            assert body["route"] == route and s.calls == 0 and s.duckdb.call_count == 0, (q, body)


def test_21_campaign_id_lookup():
    session = sid()
    c = dt.get_campaign("CAMP_00012")["campaign"]
    with Spy() as s:
        body = s.ask("CAMP_00012 revenue", session)
        follow = s.ask("what platform is it on?", session)
        why = s.ask("why is camp_12 profitable?", session)
        assert s.calls == 0
    assert body["route"] == "CAMPAIGN" and f"CAMP_00012's revenue is {fmt('revenue', c['revenue'])}." in text(body)
    assert f"CAMP_00012 is on {c['platform']}." in text(follow)
    assert why["route"] == "CAMPAIGN" and "The data shows:" in text(why) and f"{c['platform']} average" in text(why)


# --------------------------------------------------------------------------- 22-27 architecture

def _runtime_source() -> str:
    return "\n".join((BACKEND / f"{m}.py").read_text(encoding="utf-8") for m in RUNTIME)


def test_22_no_groq_anywhere_in_the_runtime():
    src = _runtime_source().lower()
    assert "groq" not in src and "api.groq.com" not in src
    for gone in ("groq_provider", "llm_adapter", "llm_providers", "fake_groq", "local_explainer"):
        assert not (BACKEND / f"{gone}.py").exists(), gone
    req = (BACKEND / "requirements.txt").read_text().lower()
    assert "groq" not in req and "openai" not in req


def test_23_no_api_key_required():
    assert "GROQ_API_KEY" not in _runtime_source() and "API_KEY" not in _runtime_source()
    with patch.dict(os.environ, {k: "" for k in os.environ if "API_KEY" in k}):
        assert client.get("/api/health").json()["status"] == "ok"
        with Spy({"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}) as s:
            assert s.ask("What is total revenue?")["route"] == "ANALYSIS"


def test_24_old_query_router_is_gone():
    assert not (BACKEND / "query_router.py").exists() and not (BACKEND / "semantic_parser.py").exists()
    assert not (BACKEND / "chat_routing.py").exists()
    src = _runtime_source()
    for word in ("DIRECT_DATABASE", "LLM_REQUIRED", "query_router", "route_query", "classify_query"):
        assert word not in src, word
    paths = {r.path for r in main.app.routes}
    assert "/api/query" not in paths and {"/api/chat", "/api/health"} <= paths


def test_25_no_planner_switch():
    assert "MARKETINGIQ_PLANNER" not in _runtime_source()
    health = client.get("/api/health").json()
    assert health["planner"] == "local_qwen" and "llm_provider" not in health and "llm_configured" not in health


def test_26_27_no_second_model_call_for_explanations_or_advice():
    session = sid()
    with Spy({"revenue by platform": REV_PLATFORM, "ROAS across platforms": ROAS_PLATFORM,
              "CPA high across devices": CPA_DEVICE}) as s:
        s.ask("revenue by platform", session)
        for q in ("Why?", "explain this", "What should we improve?", "Any recommendations?"):
            s.ask(q, session)
        assert s.calls == 1                       # only the first question was planned
        s.ask("How can we improve ROAS across platforms?")
        s.ask("Why is CPA so high across devices?")
        assert s.calls == 3                       # one planning call each, never a second (text) call


def test_28_request_is_the_exact_training_prompt_greedy_thinking_off():
    with Spy({"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}) as s:
        s.ask("What is total revenue?")
        req = s.model.fake.requests[0]
        prompt = s.model.planner.lc.render_prompt("What is total revenue?")
    runtime = json.loads((local_planner.DEFAULT_TINY_DIR / "export" / "marketingiq_runtime.json").read_text())
    assert req["prompt"] == prompt                                              # the exact Phase 4 training prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")    # thinking OFF
    assert req["temperature"] == 0 and req["top_k"] == 1 and req["n_predict"] == runtime["max_new_tokens"]
    assert "stream" not in req or req["stream"] is False


if __name__ == "__main__":
    logging.getLogger("marketingiq.analytics").setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} ANALYST TESTS PASSED ===")
