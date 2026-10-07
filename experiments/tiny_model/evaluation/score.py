"""
Field-level scoring of one planner output against a gold plan, for Groq and the tiny model alike.

Both sides go through what production does: to_canonical() (a no-op for Groq's full plans) ->
validate_plan() -> the post-processing query_planner.answer() applies -> execute_plan() on the local
DuckDB data (no LLM). Plans are compared by what they would do, not by how they are spelled:

- the validator canonicalizes values ("tiktok" -> "TikTok") and defaults ("none" order -> desc);
- a yearly trend with no breakdown and a period_comparison run the same code (_periods), and a
  period_comparison's grain doesn't change its result: both count as period_comparison by year;
- limits count as executed (campaign lists default to 10 rows, rankings to the row cap), and order
  only where it changes the result (breakdown, change_ranking, campaign_list);
- visualization is the chart type(s) production actually builds for the plan ("pie" of a
  non-additive metric is drawn as a bar, change rankings are never charted).

`execution_match` compares the whole executed answer (text lines + charts): the strictest check
that the model's plan gives the user the same result as the gold plan. A plan the validator
rejects scores 0 on every field, except that refusing a period outside the data counts as the
correct refusal for an unavailable_period gold plan.
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
import plan_format as pf  # noqa: E402

FIELDS = ["intent", "metrics", "dimension", "entities", "filters", "time", "conditions", "order_limit",
          "visualization", "dimension2_exclude"]
ORDERED = ("breakdown", "change_ranking", "campaign_list")
CAMPAIGN_ROWS = 10  # query_planner._campaigns default


def _validated(plan_text: str, question: str):
    """(Plan or None, not_answerable reason or None, error or None)."""
    qp = pf.backend()
    try:
        plan = qp.validate_plan(pf.to_canonical(plan_text))
    except qp.PlanError as e:
        if str(e).startswith("not answerable ("):
            return None, str(e)[len("not answerable ("):-1], None
        if str(e) == "year outside the data":
            return None, "unavailable_period", None
        return None, None, str(e)
    except (ValueError, TypeError) as e:
        return None, None, f"not JSON: {e}"
    return pf.postprocess(plan, question), None, None


def execute(plan):
    """(comparable answer, [chart types]) from production's execute_plan on the local data."""
    qp = pf.backend()
    try:
        out = qp.execute_plan(copy.deepcopy(plan), {})
    except (qp.PlanError, qp.qr.QueryRouterError) as e:
        return ("error", type(e).__name__, str(e)), ["error"]
    charts = out.charts or ([out.chart] if out.chart else [])
    return (out.status, out.lines, json.dumps(charts, sort_keys=True, default=str)), [c["type"] for c in charts]


def _fields(p, chart_types: list) -> dict:
    import analytics_engine  # backend module, importable once pf.backend() has run
    qp = pf.backend()
    intent = p.intent
    if intent == "trend" and p.grain == "year" and not analytics_engine.handles(p):
        intent = "period_comparison"
    grain = "year" if intent == "period_comparison" else p.grain
    if intent in ORDERED:
        order_limit = (p.order, p.limit or (CAMPAIGN_ROWS if intent == "campaign_list" else qp.MAX_ROWS))
    else:
        order_limit = None
    return {"intent": intent,
            "metrics": list(p.metrics) if p.intent == "relationship" else sorted(p.metrics),  # x/y order matters
            "dimension": p.dimension, "entities": sorted(p.entities), "filters": sorted(p.filters.items()),
            "time": (p.months, p.years, grain),
            "conditions": sorted(json.dumps(c, sort_keys=True) for c in p.conditions),
            "order_limit": order_limit, "visualization": chart_types,
            "dimension2_exclude": (p.dimension2, sorted(p.exclude))}


def score(gold: dict, output: str, question: str) -> dict:
    """gold: sparse gold plan. output: the raw text the planner returned."""
    try:
        json.loads(output)
        json_valid = True
    except (ValueError, TypeError):
        json_valid = False
    g_plan, g_reason, g_err = _validated(json.dumps(gold), question)
    if g_err:
        raise ValueError(f"gold plan is invalid ({g_err}); run validate_dataset.py first")
    p_plan, p_reason, p_err = _validated(output, question)
    result = {"json_valid": json_valid, "validator_accepted": p_err is None,
              "gold_answerable": g_plan is not None, "error": p_err}
    if g_plan is None:                      # unsupported / ambiguous / off-topic: must be refused, same reason
        result["refused"] = p_reason is not None
        result["reason"] = p_reason == g_reason
        result["exact"] = result["execution_match"] = result["reason"]
        return result
    result["refused"] = p_reason is not None  # a false refusal
    g_out, g_charts = execute(g_plan)
    gf = _fields(g_plan, g_charts)
    if p_plan is not None:
        p_out, p_charts = execute(p_plan)
        pf_ = _fields(p_plan, p_charts)
    for f in FIELDS:
        result[f] = p_plan is not None and pf_[f] == gf[f]
    result["exact"] = all(result[f] for f in FIELDS)
    result["execution_match"] = p_plan is not None and p_out == g_out
    return result
