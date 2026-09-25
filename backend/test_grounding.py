"""
Tests for the grounding layer (grounding.py): the typed LLM context, the numerical-claim
validator, the deterministic fallback, and how the LLM Adapter applies them to every provider.

No real provider call is ever made: Groq runs on a fake transport (fake_groq.py), so these tests
use no quota and need no real key. Historical bad answers come from benchmark_results/ (output
recorded from Gemini, Qwen and Groq in Phases 6-7): the validator is provider-independent.

Run:  python test_grounding.py      (or: pytest test_grounding.py)
"""
import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)
from fake_groq import FAKE_KEY

from fastapi.testclient import TestClient

import data_tools as dt
import grounding
import main
import query_router as qr
from llm_adapter import LLMAdapter, LLMProvider

GUARD = fake_groq.block_real_calls(main.llm_adapter)  # nothing below reaches api.groq.com
client = TestClient(main.app)
BENCH = Path(__file__).parent / "benchmark_results"
NNBSP = "\u202f"  # narrow no-break space, as Groq writes thousands separators


def ctx(question, filters=None):
    r = qr.route_query(question, filters or {})
    assert r["route"] == "LLM_REQUIRED", r
    return grounding.build_llm_context(question, {"filters_applied": r["filters_applied"],
                                                  "focus_metrics": r["focus_metrics"], "results": r["analysis"]})


def check(text, context, question=""):
    return grounding.validate_explanation(text, context, question)


def categories(text, context, question=""):
    return check(text, context, question).categories()


OVERALL_Q = "Give me a business interpretation of overall performance."
DEVICE_Q = "What insights can you draw about device performance?"
FILTER_Q = "Why do campaigns with spend over 10000 have lower ROAS?"
FATIGUE_Q = "Explain how creative age affects CTR."
TREND_Q = "Explain what the monthly revenue trend means."
OBJECTIVE_Q = "Why is CPA higher for some campaign objectives?"
PLATFORM_Q = "Why is TikTok performing better than other platforms?"
OVERALL, DEVICE, FILTER = ctx(OVERALL_Q), ctx(DEVICE_Q), ctx(FILTER_Q)
TOTALS = dt.get_totals()


def entity(context, name):
    comp = next(s for s in context["sections"] if s["type"] == "comparison")
    return next(e for e in comp["entities"] if e["name"] == name)


# ---------------------------------------------------------------------------
# Typed context: units, scopes and structure
# ---------------------------------------------------------------------------

def test_metric_units_describe_the_real_formulas():
    """The unit metadata must match what data_tools actually computes (it describes, never redefines)."""
    t = TOTALS
    assert t["roas"] == round(t["revenue"] / t["spend"], 3)                      # a ratio -> "x"
    assert t["ctr"] == round(t["clicks"] / t["impressions"] * 100, 3)             # already x100 -> "%"
    assert t["conversion_rate"] == round(t["conversions"] / t["clicks"] * 100, 3)
    assert t["roi_pct"] == round((t["revenue"] - t["spend"]) / t["spend"] * 100, 1)
    assert t["cpa"] == round(t["spend"] / t["conversions"], 2)
    units = OVERALL["metric_units"]
    assert units["roas"].startswith("x:") and "never a percentage" in units["roas"]
    assert units["ctr"].startswith("%:") and "already a percentage" in units["ctr"]
    assert units["conversion_rate"].startswith("%:") and units["roi_pct"].startswith("%:")
    assert units["revenue"] == units["spend"] == units["cpa"].split(":")[0] == "$"
    assert units["campaign_count"] == units["clicks"] == "count"
    # No currency code is invented: the dataset has none; the dashboard shows "$".
    assert "USD" not in json.dumps(OVERALL) and "no currency code" in OVERALL["currency"]


def test_comparison_structure():
    comp = next(s for s in DEVICE["sections"] if s["type"] == "comparison")
    assert comp["dimension"] == "device" and comp["ranked_by"] == "roas" and comp["order"] == "highest first"
    assert comp["entity_count"] == len(comp["entities"]) == 3
    assert list(comp["entities"][0])[0] == "name"  # each row starts with what it describes
    agg = next(s for s in DEVICE["sections"] if s["type"] == "aggregate")
    assert agg["of"] == "all campaigns in scope" and agg["metrics"] == TOTALS


def test_creative_age_unit_is_days():
    c = ctx(FATIGUE_Q)
    b = next(s for s in c["sections"] if s["type"] == "creative_age_buckets")
    assert b["unit"] == "days" and "not audience age" in b["meaning"]
    assert [x["creative_age_days"] for x in b["buckets"]] == dt.CREATIVE_AGE_LABELS
    assert dt.get_repository().has_column("creative_age_days")  # the unit comes from the real column
    assert "age_range" not in json.dumps(c)  # the old unit-less label is gone


def test_filtered_subset_has_aggregates_and_labelled_examples():
    s = next(x for x in FILTER["sections"] if x["type"] == "filtered_subset")
    assert s["conditions"] == [{"field": "spend", "operator": ">", "value": 10000.0, "unit": "$"}]
    # Aggregates are the whole matching group, computed with the same formulas as get_totals.
    df = dt.get_dataframe()
    sub = df[df["ad_spend"] > 10000]
    assert s["matched_campaign_count"] == s["aggregates"]["campaign_count"] == len(sub)
    assert s["aggregates"] == dt._metrics_from_rows(sub)
    ex = s["examples"]
    assert ex["count"] == len(ex["campaigns"]) <= qr.MAX_LLM_CAMPAIGN_ROWS
    assert "individual campaigns, not averages" in ex["selection"]
    assert "sorted by spend ascending" in ex["selection"]
    assert "aggregates" not in ex and "campaigns" not in s  # never one flat mixed structure
    assert any("example campaigns" in x for x in FILTER["analysis_scope"]["does_not_support"])
    # DIRECT_DATABASE results keep their exact shape (aggregates only on request).
    assert "aggregates" not in dt.filter_campaigns([{"field": "spend", "operator": ">", "value": 10000}])


def test_time_series_scope_and_units():
    c = ctx("Why did TikTok revenue change over time?")
    ts = next(s for s in c["sections"] if s["type"] == "time_series")
    assert ts["metric"] == "revenue" and ts["unit"] == "$" and ts["of"] == {"only": {"platform": "TikTok"}}
    assert ts["granularity"].startswith("calendar month")


def test_data_sufficiency_signal():
    scope = ctx("Explain the decline in CTR.")["analysis_scope"]
    assert scope["supports"] and scope["does_not_support"][0].startswith("proving causes")
    assert any("change over time in ctr" in x for x in scope["does_not_support"])  # no CTR time series exists
    assert not any("change over time in revenue" in x for x in ctx(TREND_Q)["analysis_scope"]["does_not_support"])


def test_active_filters_ignore_dashboard_placeholders():
    c = ctx(PLATFORM_Q, {"platform": "All Platforms", "objective": "All Objectives", "budget": "High"})
    assert c["active_filters"] == {"budget_tier": "High"} and c["population"] == "dashboard-filtered view"
    assert ctx(PLATFORM_Q, {"platform": "All Platforms"})["population"] == "all campaigns"


# ---------------------------------------------------------------------------
# Validator: required cases
# ---------------------------------------------------------------------------

def test_supported_revenue_passes():
    assert check("Total revenue was $284,157,706.47 on $43,455,661.79 of spend.", OVERALL).passed


def test_invented_revenue_fails():
    r = check("Revenue reached $211,005,852.77 in total.", OVERALL)
    assert not r.passed and r.categories() == ["unsupported_value"]
    assert not check("Revenue was $28,415,770.64.", OVERALL).passed  # a real value with invented digits


def test_equivalent_formatting_passes():
    for text in ("Revenue was $284.16M.", "Revenue was $284.2M.", "Revenue was 284.2 million.",
                 "Revenue was $284,157,706.", "Revenue was 284157706.47.", "Revenue was $0.28 billion.",
                 f"Revenue was ${'284' + NNBSP + '157' + NNBSP + '706.47'}.", "10,000 campaigns and 10000 campaigns."):
        assert check(text, OVERALL).passed, text
    assert not check("Revenue was $284.9M.", OVERALL).passed  # rounding, not approximation
    # Rounded or truncated at the written precision (seen live from Groq: "$10 001" for 10,001.88).
    spend = next(s for s in FILTER["sections"] if s["type"] == "filtered_subset")["examples"]["campaigns"][0]["spend"]
    assert spend == 10001.88
    for text in ("One campaign spent $10,001.", "One campaign spent $10,002.", f"One campaign spent $10{NNBSP}001."):
        assert check(text, FILTER, FILTER_Q).passed, text
    for text in ("One campaign spent $10,000.99.", "One campaign spent $10,003.", "ROAS was 6.55."):
        assert not check(text, FILTER if "spent" in text else OVERALL, FILTER_Q).passed, text


def test_roas_ratio_passes():
    for text in ("ROAS was 6.54x.", "Overall ROAS is 6.54.", "ROAS: 6.539", "a 6.54 \u00d7 ROAS",
                 "each $1 of spend returned $6.54 in revenue", "$6.54 in revenue for every $1 spent"):
        assert check(text, OVERALL).passed, text


def test_roas_as_percentage_fails():
    r = check("Overall ROAS was 6.54%.", OVERALL)
    assert not r.passed and r.categories() == ["unit_mismatch"]
    assert r.issues[0]["claimed_unit"] == "%" and "x" in r.issues[0]["data_units"]
    assert not check("ROAS was 6.54 percent.", OVERALL).passed


def test_correct_ctr_passes():
    for text in ("Overall CTR is 2.16%.", "CTR was 2.164 percent.", "a click-through rate of 2.16%",
                 "Conversion rate was 4.30%.", "ROI of 553.9%"):
        assert check(text, OVERALL).passed, text


def test_cpa_labelled_as_ctr_detected():
    tablet = entity(DEVICE, "Tablet")
    r = check(f"Tablet has a higher click-through rate ({tablet['cpa']}).", DEVICE)
    assert not r.passed and r.issues[0]["category"] == "metric_label_mismatch"
    assert r.issues[0]["label"] == "ctr" and r.issues[0]["data_metric"] == "cpa"
    assert check(f"Tablet CPA is ${tablet['cpa']}.", DEVICE).passed
    # Revenue presented as spend.
    assert categories(f"Desktop spent ${entity(DEVICE, 'Desktop')['revenue']:,}.", DEVICE) == ["metric_label_mismatch"]


def test_filter_aggregate_not_confused_with_example():
    s = next(x for x in FILTER["sections"] if x["type"] == "filtered_subset")
    example_roas = s["examples"]["campaigns"][0]["roas"]
    group_roas = s["aggregates"]["roas"]
    ok = (f"The {s['matched_campaign_count']:,} campaigns with spend over $10,000 have a ROAS of "
          f"{group_roas:.2f}x, versus {TOTALS['roas']:.2f}x overall. One example campaign has a ROAS of {example_roas}.")
    assert check(ok, FILTER, FILTER_Q).passed
    # An individual example presented as the group's average.
    assert categories(f"These campaigns have an average ROAS of {example_roas}.", FILTER, FILTER_Q) == ["example_as_aggregate"]
    # The overall figure presented as the subset's (historical Groq Q6 error).
    bad = (f"Among the {s['matched_campaign_count']:,} campaigns with spend > $10,000, the average ROAS is "
           f"{TOTALS['roas']} (overall dataset).")
    assert categories(bad, FILTER, FILTER_Q) == ["scope_mismatch"]
    assert check(f"Overall ROAS is {TOTALS['roas']}; {s['matched_campaign_count']:,} campaigns spent over $10,000.",
                 FILTER, FILTER_Q).passed


def test_harmless_structural_numbers_pass():
    text = ("What the data shows:\n1. Desktop leads on ROAS.\n2. Mobile trails.\n\n"
            "There are 3 observations and two reasons worth noting. First, see step 2 and option 3.\n"
            "The top 3 devices differ; #1 is Desktop, the 2nd is Tablet.")
    r = check(text, DEVICE)
    assert r.passed and r.numbers_checked == 0, r.issues
    # Inline enumeration counts only as a run from 1).
    assert check("Two possible reasons: 1) cheaper clicks 2) younger audiences.", DEVICE).passed
    assert not check("Desktop leads (7) on ROAS.", DEVICE).passed
    assert not check("Mobile trails 7) behind Desktop.", DEVICE).passed
    # ...but a structural-looking number that is really a claim is still checked.
    assert not check("Desktop has 7 campaigns.", DEVICE).passed
    platforms = ctx(PLATFORM_Q)
    assert check("The top 5 platforms differ.", platforms).passed      # within the 6 supplied platforms
    assert not check("The top 10 platforms differ.", platforms).passed  # only 6 exist (historical Qwen Q1)


# ---------------------------------------------------------------------------
# Validator: other observed error classes
# ---------------------------------------------------------------------------

def test_creative_age_read_as_years_or_audience_age():
    c = ctx(FATIGUE_Q)
    assert check("CTR is 2.64% for creatives aged 0-15 days and 1.47% at 61-90 days.", c, FATIGUE_Q).passed
    assert "unit_mismatch" in categories("0-15 yrs: 2.64%", c, FATIGUE_Q)
    assert "metric_label_mismatch" in categories("the lowest CTR of 1.47 in the 61-90 age group", c, FATIGUE_Q)


def test_value_attached_to_wrong_month_or_entity():
    c = ctx(TREND_Q)
    ts = next(s for s in c["sections"] if s["type"] == "time_series")
    p0, p1 = ts["points"][0], ts["points"][1]
    month = grounding._month_name(p0["month"])
    assert check(f"Revenue was ${p0['value']:,.2f} in {month}.", c, TREND_Q).passed
    assert categories(f"Revenue was ${p1['value']:,.2f} in {month}.", c, TREND_Q) == ["month_mismatch"]
    tiktok, linkedin = entity(ctx(PLATFORM_Q), "TikTok"), entity(ctx(PLATFORM_Q), "LinkedIn")
    assert check(f"TikTok's ROAS is {tiktok['roas']} while LinkedIn's is {linkedin['roas']}.", ctx(PLATFORM_Q)).passed
    assert check(f"ROAS ranges from {linkedin['roas']} on LinkedIn to {tiktok['roas']} on TikTok.", ctx(PLATFORM_Q)).passed
    assert categories(f"TikTok's ROAS is {linkedin['roas']}.", ctx(PLATFORM_Q)) == ["entity_mismatch"]


def test_derived_numbers_are_not_supported():
    c = ctx(PLATFORM_Q)
    assert categories("TikTok's ROAS is roughly 3.4 times higher than LinkedIn's.", c) == ["derived_value"]
    assert categories("Mobile makes up 54% of campaigns.", DEVICE) == ["unsupported_value"]
    assert categories("TikTok's CPA is 20% lower.", c) == ["derived_value"]


def test_dates_are_checked():
    assert check("From January 2024 to January 2026 revenue varied.", ctx(TREND_Q), TREND_Q).passed
    assert categories("Revenue peaked in 2019.", ctx(TREND_Q), TREND_Q) == ["unsupported_date"]


def test_unicode_hyphen_dates():
    """Groq writes "2024‑01" with U+2011; U+2010 and "-" must validate the same way."""
    c = ctx(TREND_Q)
    ts = next(s for s in c["sections"] if s["type"] == "time_series")
    p = ts["points"][0]
    for dash in ("-", "‐", "‑"):
        text = f"Revenue was ${p['value']:,.2f} in {p['month'].replace('-', dash)}."
        assert check(text, c, TREND_Q).passed, dash
        assert categories(f"Revenue peaked in 2019{dash}12.", c, TREND_Q) == ["unsupported_date"], dash


def test_comparisons_between_supported_numbers():
    c = ctx(PLATFORM_Q)
    tiktok, linkedin = entity(c, "TikTok"), entity(c, "LinkedIn")
    hi, lo = tiktok["roas"], linkedin["roas"]
    for text in (f"TikTok's ROAS ({hi}x) is higher than LinkedIn's ({lo}x).",
                 f"LinkedIn's ROAS ({lo}x) is lower than TikTok's ({hi}x).",
                 f"TikTok's ROAS of {hi}x is above LinkedIn's {lo}x.",
                 f"TikTok's ROAS ({hi}x) is higher than LinkedIn's ({lo}x), with a CPA of ${tiktok['cpa']}."):
        assert check(text, c).passed, text
    for text in (f"LinkedIn's ROAS ({lo}x) is higher than TikTok's ({hi}x).",
                 f"TikTok's ROAS ({hi}x) is lower than LinkedIn's ({lo}x).",
                 f"LinkedIn's ROAS of {lo}x is above TikTok's {hi}x."):
        assert "wrong_comparison" in categories(text, c), text
    # A threshold is not a comparison.
    s = next(x for x in FILTER["sections"] if x["type"] == "filtered_subset")
    assert check(f"The {s['matched_campaign_count']:,} campaigns with spend above $10,000 have a ROAS of "
                 f"{s['aggregates']['roas']}x, below the overall {TOTALS['roas']}x.", FILTER, FILTER_Q).passed


def test_recorded_q1_false_comparison_is_blocked():
    """Groq gpt-oss-120b, Groq-only quality test Q1: both numbers exist, the relation is false."""
    text = ("- Conversion rate was **3.439%**, slightly below Facebook (4.26%) and Instagram (4.154%) but above "
            "Google Ads (4.992%) and LinkedIn (4.995%) when measured as a percentage of clicks.")
    r = check(text, ctx(PLATFORM_Q), PLATFORM_Q)
    assert r.categories() == ["wrong_comparison"]
    assert {i["claim"] for i in r.issues} == {"3.439% above 4.992%", "3.439% above 4.995%"}
    # The same errors with the other side named instead of written as a number (real Groq
    # gpt-oss-120b answer from the fix's smoke test): compared with the named entities' values.
    real = ("- Its profit was **$31,278,715.96**, and ROI was **1018.4%**, both above all other platforms.\n"
            "- TikTok’s CPA was **$37.6**, lower than Facebook ($41.12), Instagram ($51.42), Twitter ($49.56), "
            "Google Ads ($80.77) and LinkedIn ($131.11).\n"
            "- Conversion rate was **3.439%**, slightly below Facebook and Instagram but above Google Ads and LinkedIn.")
    r = check(real, ctx(PLATFORM_Q), PLATFORM_Q)
    assert {i["claim"] for i in r.issues} == {"$31,278,715.96 above Facebook", "$31,278,715.96 above Google Ads",
                                              "3.439% above Google Ads", "3.439% above LinkedIn"}
    assert check(f"TikTok's ROAS of {entity(ctx(PLATFORM_Q), 'TikTok')['roas']}x is above all other platforms.",
                 ctx(PLATFORM_Q), PLATFORM_Q).passed


def test_superlatives_checked_against_comparison_rows():
    small = {"sections": [{"type": "comparison", "dimension": "platform", "entity_count": 3, "entities": [
        {"name": "TikTok", "profit": 10}, {"name": "Google Ads", "profit": 20}, {"name": "LinkedIn", "profit": 15}]}]}
    assert categories("TikTok has the highest profit.", small) == ["wrong_comparison"]
    assert check("Google Ads has the highest profit.", small).passed
    assert check("TikTok has the lowest profit.", small).passed
    assert categories("LinkedIn has the lowest profit.", small) == ["wrong_comparison"]
    c = ctx(PLATFORM_Q)
    for text in ("TikTok has the highest ROAS.", "The highest profit came from Google Ads.",
                 "TikTok has the lowest CPA.", "TikTok has the best CPA.", "LinkedIn has the worst ROAS.",
                 "Google Ads had the most profit.", "TikTok has the second-highest profit.",
                 "TikTok is one of the highest in spend, and has the highest ROAS.", "TikTok has the best spend.",
                 "Google Ads has the highest CPA among the top five platforms."):
        assert check(text, c, PLATFORM_Q).passed, text
    for text in ("Google Ads has the lowest ROAS.", "TikTok has the best conversion rate.",
                 "TikTok has the worst CPA.", "LinkedIn had the most profit.", "TikTok has the highest spend."):
        assert categories(text, c, PLATFORM_Q) == ["wrong_comparison"], text


def test_recorded_tiktok_highest_profit_is_blocked():
    """Recorded Groq failure: TikTok's profit described as above all other platforms (Google Ads and
    Facebook are higher). The equivalent superlative must be blocked too, with or without the figure."""
    c = ctx(PLATFORM_Q)
    profit = entity(c, "TikTok")["profit"]
    for text in ("TikTok has the highest profit of all platforms.",
                 f"TikTok had the highest profit (${profit:,.2f}).",
                 f"With ${profit:,.2f}, TikTok delivered the highest profit."):
        r = check(text, c, PLATFORM_Q)
        assert r.categories() == ["wrong_comparison"], (text, r.issues)
        assert {i["claim"] for i in r.issues} == {"TikTok highest profit"}


def test_safe_summary_is_itself_grounded():
    questions = [PLATFORM_Q, "Explain the decline in CTR.", TREND_Q, FILTER_Q, OVERALL_Q, DEVICE_Q, FATIGUE_Q, OBJECTIVE_Q]
    for q in questions:
        c = ctx(q)
        text = grounding.safe_summary(c)
        r = check(text, c, q)
        assert r.passed and r.causal_flags == 0 and r.numbers_checked > 3, (q, r.issues)
        assert text.startswith("What the data shows:") and grounding.WITHHELD_NOTE in text
    filtered = ctx(PLATFORM_Q, {"budget": "High"})
    assert "budget tier = High" in grounding.safe_summary(filtered)


# ---------------------------------------------------------------------------
# Adapter policy (provider-independent)
# ---------------------------------------------------------------------------

class Canned(LLMProvider):
    name, model = "canned", "none"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate_explanation(self, question, analysis, timeout_s):
        self.calls += 1
        return self.reply


def explain(reply, question=OVERALL_Q):
    provider = Canned(reply)
    r = qr.route_query(question, {}, explainer=LLMAdapter(provider))
    return r, provider


def test_unsupported_answer_is_replaced_without_another_llm_call():
    r, provider = explain("What the data shows: revenue was $211,005,852.77.")
    assert provider.calls == 1  # no retry / regeneration loop
    assert r["llm"]["status"] == "ok" and r["llm"]["grounding"]["action"] == "replaced_with_data_summary"
    assert r["llm"]["grounding"]["passed"] is False and r["llm"]["grounding"]["issues"] == ["unsupported_value"]
    assert "211,005,852.77" not in json.dumps(r)  # the unsupported answer is never exposed
    assert r["explanation"] == grounding.safe_summary(grounding.build_llm_context(
        OVERALL_Q, {"filters_applied": {}, "focus_metrics": r["focus_metrics"], "results": r["analysis"]}))


def test_causal_policy():
    hedged = "What the data shows: overall ROAS is 6.54.\nInterpretation: this may be driven by cheaper clicks."
    r, _ = explain(hedged)
    assert r["explanation"] == hedged and r["llm"]["grounding"]["action"] == "none"
    unhedged = "What the data shows: overall ROAS is 6.54.\nInterpretation: cheaper clicks drive the higher ROAS."
    r, _ = explain(unhedged)
    assert r["llm"]["grounding"]["action"] == "caveat_added"
    assert r["explanation"].startswith(unhedged) and r["explanation"].endswith(grounding.CAUSAL_CAVEAT)
    # Numbers are checked first: an unsupported number replaces the answer whatever else it says.
    r, _ = explain("ROAS is 6.54% because of TikTok.")
    assert r["llm"]["grounding"]["action"] == "replaced_with_data_summary"


def test_validator_error_never_exposes_an_unchecked_answer():
    with patch.object(grounding, "validate_explanation", side_effect=RuntimeError("bug")):
        r, _ = explain("Revenue was $1.")
    assert r["llm"]["grounding"]["action"] == "replaced_with_data_summary" and "$1." not in r["explanation"]


def test_grounding_log_is_sanitized():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    lg = logging.getLogger("marketingiq.llm_adapter")
    level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    secret_question = "Why is overall performance SECRETQUESTION good?"
    try:
        qr.route_query(secret_question, {}, explainer=LLMAdapter(Canned("SECRETANSWER revenue was $999.99.")))
    finally:
        lg.removeHandler(handler)
        lg.setLevel(level)
    events = [json.loads(r.getMessage()) for r in records if '"grounding_check"' in r.getMessage()]
    assert len(events) == 1
    e = events[0]
    assert e["provider"] == "canned" and e["passed"] is False and e["action"] == "replaced_with_data_summary"
    assert e["issues"] == [{"category": "unsupported_value", "claim": "$999.99"}]
    blob = json.dumps(e)
    assert "SECRETQUESTION" not in blob and "SECRETANSWER" not in blob and FAKE_KEY not in blob


# ---------------------------------------------------------------------------
# Every Groq answer goes through the validator; DIRECT_DATABASE never does
# ---------------------------------------------------------------------------

def groq_adapter(text):
    provider, fake = fake_groq.provider(fake_groq.reply(text))
    return LLMAdapter(provider), fake


def test_groq_responses_are_validated():
    good = f"What the data shows: overall ROAS is {TOTALS['roas']:.2f}x."
    adapter, fake = groq_adapter(good)
    with patch.object(grounding, "validate_explanation", wraps=grounding.validate_explanation) as spy:
        r = qr.route_query(OVERALL_Q, {}, explainer=adapter)
    assert len(fake.requests) == 1 and spy.call_count == 1 and r["explanation"] == good
    assert r["llm"]["provider"] == "groq" and r["llm"]["grounding"]["passed"] is True
    adapter, fake = groq_adapter("ROAS is 6.54%.")  # a ratio written as a percentage
    r = qr.route_query(OVERALL_Q, {}, explainer=adapter)
    assert len(fake.requests) == 1 and r["llm"]["grounding"]["issues"] == ["unit_mismatch"]
    assert "6.54%" not in r["explanation"]

    tablet = entity(DEVICE, "Tablet")
    adapter, fake = groq_adapter(f"Tablet CPA is ${tablet['cpa']}.")
    r = qr.route_query(DEVICE_Q, {}, explainer=adapter)
    assert len(fake.requests) == 1 and r["llm"]["grounding"]["passed"] is True
    adapter, fake = groq_adapter(f"Tablet has a higher click-through rate ({tablet['cpa']}).")
    r = qr.route_query(DEVICE_Q, {}, explainer=adapter)
    assert len(fake.requests) == 1 and r["llm"]["grounding"]["issues"] == ["metric_label_mismatch"]
    # The answer is validated against exactly the typed context Groq was given.
    sent = fake.user_prompt()
    assert '"metric_units"' in sent and '"analysis_scope"' in sent


def test_direct_database_skips_llm_and_grounding():
    with patch.object(grounding, "validate_explanation", side_effect=AssertionError("validator called")) as v, \
            patch.object(grounding, "build_llm_context", side_effect=AssertionError("context built")) as b:
        for q in ("What is total revenue?", "Which platform has the highest ROAS?", "Show monthly revenue.",
                  "What is the average CTR?", "Compare TikTok and LinkedIn ROAS.",
                  "Show campaigns with spend over 10000."):
            body = client.post("/api/query", json={"query": q}).json()
            assert body["route"] == "DIRECT_DATABASE" and "llm" not in body, (q, body)
            chat = client.post("/api/chat", json={"messages": [{"role": "user", "content": q}], "filters": {}}).json()
            assert chat["route"] == "DIRECT_DATABASE", (q, chat)
    assert v.call_count == b.call_count == 0 and GUARD.requests == []
    # Still millisecond-level: time the router alone (no HTTP), after warm-up.
    qr.route_query("What is total revenue?")
    started = time.perf_counter()
    for _ in range(20):
        qr.route_query("What is total revenue?")
    assert (time.perf_counter() - started) / 20 < 0.05


# ---------------------------------------------------------------------------
# Context-size protection and no full dataset
# ---------------------------------------------------------------------------

def test_context_size_protection_still_works():
    provider = Canned("ok")
    r = qr.route_query(PLATFORM_Q, {}, explainer=LLMAdapter(provider, max_analysis_bytes=500))
    assert r["llm"]["error"] == "analysis_too_large" and provider.calls == 0
    assert r["llm"]["analysis_bytes"] > 500
    big = {"results": [{"tool": "x", "result": {"rows": ["x" * 100] * 1000}}]}
    assert LLMAdapter(provider).explain("Why?", big)["error"] == "analysis_too_large" and provider.calls == 0
    for q in (PLATFORM_Q, FILTER_Q, TREND_Q, FATIGUE_Q, OVERALL_Q):
        size = len(json.dumps(ctx(q), separators=(",", ":")).encode())
        assert size < 4_000, (q, size)  # typed context stays small (limit is 16 KB)


def test_no_full_dataset_reaches_providers():
    provider = Canned("ok")
    adapter = LLMAdapter(provider)
    sent = []
    provider.generate_explanation = lambda q, a, t: sent.append(a) or "ok"
    r = qr.route_query("Why do campaigns with spend over 100 perform differently?", {}, explainer=adapter)
    fs = next(s for s in sent[0]["sections"] if s["type"] == "filtered_subset")
    assert fs["matched_campaign_count"] > 1000 and len(fs["examples"]["campaigns"]) <= qr.MAX_LLM_CAMPAIGN_ROWS
    payload = json.dumps(sent[0])
    assert payload.count('"id":') <= qr.MAX_LLM_CAMPAIGN_ROWS and len(payload) < os.path.getsize(main.CSV_PATH) / 100
    df_block = {"results": [{"tool": "x", "result": {"rows": dt.get_dataframe()}}]}
    assert adapter.explain("Why?", df_block)["error"] == "invalid_analysis"


# ---------------------------------------------------------------------------
# Historical bad provider answers (benchmark_results/, recorded Phases 6-7) as regression fixtures
# ---------------------------------------------------------------------------

# (file, provider) -> {question number: expected issue categories}; unlisted answered questions pass.
HISTORICAL = {
    ("phase6_qwen_vs_gemini.json", "qwen"): {
        1: {"unsupported_value"},                        # "top 10 platforms" (6 exist)
        3: {"metric_label_mismatch"},                    # creative age read as audience age
        4: {"unsupported_value"},                        # invented revenue ($211,005,852.77)
        8: {"metric_label_mismatch"},                    # CPA 74.57 presented as CTR
    },
    ("phase7_gemini_vs_groq_gpt-oss-20b.json", "gemini"): {
        6: {"unsupported_value"},                        # "5 of the 10" sampled below 1 (it is 4)
        7: {"unsupported_value"},                        # "$1.76" (the value is $1.752)
    },
    ("phase7_gemini_vs_groq_gpt-oss-20b.json", "groq"): {
        1: {"wrong_comparison"},                         # conversion rate 3.439% "higher than" overall 4.296%
        3: {"unit_mismatch"},                            # creative-age bounds written as percentages
        4: {"month_mismatch"},                           # value attached to the wrong month
        5: {"derived_value"},                            # "3.4 times higher"
        6: {"scope_mismatch"},                           # overall ROAS attributed to the >$10K subset
        8: {"unsupported_value"},                        # "54% of campaigns" (computed)
        9: {"unit_mismatch", "metric_label_mismatch"},   # creative age read as audience age in years
    },
    ("phase7_groq_gpt-oss-120b.json", "groq"): {
        1: {"wrong_comparison"},                         # conversion rate 3.439% "slightly above" overall 4.296%
        4: {"unsupported_value"},                        # invented monthly low ($7,441,396)
        6: {"unsupported_value"},                        # "exceed 4" / "below 1" (not supplied)
        7: {"unsupported_value", "unit_mismatch"},       # computed margin 84.7%, "over $6"
    },
}
# Correct answers that must NOT be flagged but whose unhedged causal wording gets the caveat.
HISTORICAL_CAUSAL = {("phase7_gemini_vs_groq_gpt-oss-20b.json", "groq"): {1, 7, 10}}


def test_historical_bad_answers():
    questions = [q["question"] for q in
                 json.loads((BENCH / "phase7_gemini_vs_groq_gpt-oss-20b.json").read_text("utf-8"))["questions"]]
    contexts = {q: ctx(q) for q in questions}
    for (file, provider), expected in HISTORICAL.items():
        answers = json.loads((BENCH / file).read_text("utf-8"))["providers"][provider]["answers"]
        for i, a in enumerate(answers, 1):
            if a["status"] != "ok":
                continue
            q = questions[i - 1]
            r = check(a["text"], contexts[q], q)
            assert set(r.categories()) == expected.get(i, set()), (file, provider, i, r.issues)
            if i in HISTORICAL_CAUSAL.get((file, provider), set()):
                assert r.causal_flags, (file, provider, i)


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} GROUNDING TESTS PASSED ===")
