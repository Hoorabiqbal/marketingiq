"""
Deterministic findings: a validated plan's DuckDB results -> factual statements. No LLM.

    validated plan (+ dashboard filters)
      -> data tools on DuckDB (metric_table, get_totals, filter_campaigns, campaign_points)
      -> structured comparisons computed here: highest / lowest, gaps, ratios, change over time,
         correlation, loss-making counts
      -> Findings (statements for the user + facts for recommendation_engine)

Every figure is read from a structured result and formatted with the same formatter the answers
use; nothing is parsed back out of text. Findings say what the data shows, never why: the dataset
is observational, so no statement here claims a cause.
"""
import copy
import math
from dataclasses import dataclass, field

import analytics_engine
import analytics_executor as ax
import data_tools as dt
import grounding
import query_planner as qp
import schema_catalog as catalog

LOWER_IS_BETTER = grounding.LOWER_IS_BETTER        # cpa, cpc
NEUTRAL = {"spend", "campaign_count"}               # more is neither better nor worse
MAX_TREND_ENTITIES = 6


@dataclass
class Findings:
    statements: list = field(default_factory=list)  # plain text, each a complete factual sentence
    kind: str = "none"          # ranking / comparison / total / context / trend / relationship / campaigns / summary
    metric: str = None
    dimension: str = None
    best: str = None            # better end by the metric's direction (lowest for CPA/CPC)
    worst: str = None
    highest: str = None         # by value
    lowest: str = None
    direction: str = None       # trend: increase / decrease / flat
    lowest_period: str = None
    correlation: float = None
    loss_campaigns: int = None
    roas_best: str = None       # efficiency context for ranking advice (by ROAS of the same rows)
    roas_worst: str = None
    top_spender: str = None
    top_spender_roas_rank: int = None
    entity_count: int = 0
    x: str = None               # relationship metrics
    y: str = None
    focus: str = None           # total with one entity filter: that entity and its rank
    focus_rank: int = None


def label(metric: str) -> str:
    """Metric name inside a sentence ("revenue", "ROAS")."""
    return catalog.lower_label(metric)


def Label(metric: str) -> str:
    """Metric name starting a sentence ("Revenue", "ROAS")."""
    return catalog.label(metric)


PLURALS = {"age_group": "age groups", "budget_tier": "budget tiers", "creative_format": "creative formats",
           "creative_emotion": "creative emotions", "income_bracket": "income brackets",
           "audience_interest": "audience interests", "operating_system": "operating systems",
           "retargeting": "retargeting groups", "day_of_week": "days of the week", "creative_size": "creative sizes",
           "ad_copy_length": "ad copy lengths", "call_to_action": "calls to action", "purchase_intent": "purchase-intent levels"}


def plural(dimension: str) -> str:
    return PLURALS.get(dimension, dimension.replace("_", " ") + "s") if dimension else "segments"


def poss(name) -> str:
    name = str(name)
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def fmt(metric: str, value) -> str:
    return qp._fmt(metric, value)


def _higher_is_better(metric: str):
    if metric in LOWER_IS_BETTER:
        return False
    return None if metric in NEUTRAL else True


def _scope_filters(plan, filters: dict) -> dict:
    return {**(filters or {}), **plan.filters, **qp._period_filter(plan)}


def _rows(base: dict, dimension: str, metric: str, names=None, exclude=()) -> list:
    rows = [r for r in dt.metric_table(base, group_by=dimension) if r.get(metric) is not None]
    rows = [r for r in rows if str(r["group"]) not in set(exclude)]
    if names:
        wanted = set(names)
        rows = [r for r in rows if str(r["group"]) in wanted]
    return sorted(rows, key=lambda r: r[metric], reverse=True)


def _ratio_text(high: float, low: float) -> str:
    """"X is N times Y", only when both are positive and the ratio is meaningful."""
    if high is None or low is None or low <= 0 or high <= 0 or high == low:
        return ""
    r = high / low
    return f"{r:.1f} times" if r >= 1.15 else ""


def _efficiency(f: Findings, rows: list, dimension: str, compared: bool = False):
    """ROAS ranking of the same rows (every metric_table row carries it): what budget advice uses.
    compared: the rows are only the named values, so statements say "of those compared"."""
    eff = sorted((r for r in rows if r.get("roas") is not None), key=lambda r: r["roas"], reverse=True)
    if len(eff) >= 2:
        f.roas_best, f.roas_worst = str(eff[0]["group"]), str(eff[-1]["group"])
        top = max(rows, key=lambda r: r.get("spend") or 0)
        f.top_spender = str(top["group"])
        f.top_spender_roas_rank = [str(r["group"]) for r in eff].index(f.top_spender) + 1
        if f.metric == "roas":
            return
        if compared and len(eff) == 2:
            f.statements.append(f"By ROAS, {eff[0]['group']} is the more efficient of the two "
                                f"({fmt('roas', eff[0]['roas'])} vs {fmt('roas', eff[1]['roas'])}).")
        else:
            group = f"the {len(eff)} compared" if compared else f"the {plural(dimension)}"
            f.statements.append(
                f"By ROAS, {eff[0]['group']} is the most efficient of {group} "
                f"({fmt('roas', eff[0]['roas'])}) and {eff[-1]['group']} the least ({fmt('roas', eff[-1]['roas'])}).")


def _ranking(plan, base, f: Findings, names=None):
    metric, dim = plan.metrics[0], plan.dimension
    rows = _rows(base, dim, metric, names, plan.exclude)
    if len(rows) < 2:
        return _summary(plan, base, f)
    f.kind, f.metric, f.dimension, f.entity_count = ("comparison" if names else "ranking"), metric, dim, len(rows)
    hi, lo = rows[0], rows[-1]
    f.highest, f.lowest = str(hi["group"]), str(lo["group"])
    better = _higher_is_better(metric)
    f.best, f.worst = ((f.highest, f.lowest) if better is not False else (f.lowest, f.highest)) if better is not None \
        else (None, None)
    if len(rows) == 2:
        f.statements.append(f"{hi['group']} has a higher {label(metric)} than {lo['group']}: "
                            f"{fmt(metric, hi[metric])} vs {fmt(metric, lo[metric])}.")
    else:
        of_n = f"of the {len(rows)} {plural(dim)} compared" if names else f"across {len(rows)} {plural(dim)}"
        f.statements.append(f"{hi['group']} has the highest {label(metric)} {of_n} at {fmt(metric, hi[metric])}, "
                            f"while {lo['group']} has the lowest at {fmt(metric, lo[metric])}.")
    ratio = _ratio_text(hi[metric], lo[metric])
    if ratio:
        f.statements.append(f"{poss(hi['group'])} {label(metric)} is {ratio} {poss(lo['group'])}.")
    if better is False:
        f.statements.append(f"For {label(metric)}, lower is better, so {f.best} is the most cost-efficient "
                            f"and {f.worst} the least.")
    if metric in ("revenue", "spend", "profit", "conversions", "clicks", "impressions") or len(plan.metrics) > 1:
        _efficiency(f, rows, dim, compared=bool(names))
    if metric == "spend" and f.top_spender and f.top_spender_roas_rank:
        f.statements.append(f"{f.top_spender} receives the most spend; its ROAS ranks {f.top_spender_roas_rank} "
                            f"of {len(rows)}.")


def _total(plan, filters, f: Findings):
    """One figure. With exactly one entity filter, the same metric for every value of that dimension
    gives the context ("LinkedIn ROAS is the lowest of 6 platforms")."""
    base = _scope_filters(plan, filters)
    t = ax.run("get_totals", base)
    metric = plan.metrics[0]
    f.metric = metric
    if not t.get("campaign_count"):
        f.statements.append("No campaigns match that selection.")
        return
    f.statements.append(f"{Label(metric)} is {fmt(metric, t[metric])} across {t['campaign_count']:,} campaigns"
                        + (f" ({', '.join(f'{k} = {v}' for k, v in plan.filters.items())})." if plan.filters else "."))
    if len(plan.filters) != 1:
        f.kind = "total"
        return
    key, value = next(iter(plan.filters.items()))
    dim = next((d for d, k in ax.ENTITY_FILTER_KEYS.items() if k == key), None)
    if dim is None:
        f.kind = "total"
        return
    context = {k: v for k, v in base.items() if k != key}
    rows = _rows(context, dim, metric)
    names = [str(r["group"]) for r in rows]
    if value not in names or len(rows) < 2:
        f.kind = "total"
        return
    rank = names.index(value) + 1
    f.kind, f.dimension, f.entity_count = "context", dim, len(rows)
    f.highest, f.lowest = names[0], names[-1]
    better = _higher_is_better(metric)
    if better is not None:
        f.best, f.worst = (names[0], names[-1]) if better else (names[-1], names[0])
    place = ("the highest" if rank == 1 else "the lowest" if rank == len(rows) else f"number {rank} from the top")
    f.statements.append(f"That is {place} of {len(rows)} {plural(dim)}: the highest is {rows[0]['group']} at "
                        f"{fmt(metric, rows[0][metric])} and the lowest is {rows[-1]['group']} at "
                        f"{fmt(metric, rows[-1][metric])}.")
    if better is False:
        f.statements.append(f"For {label(metric)}, lower is better.")
    f.focus, f.focus_rank = value, rank


def _trend(plan, base, f: Findings):
    metric = plan.metrics[0]
    f.kind, f.metric = "trend", metric
    if plan.grain == "year" or plan.intent == "period_comparison":
        periods = [p for p in qp._series_periods(plan) if qp._complete(*p)]
        series = []
        for p in periods:
            t = ax.run("get_totals", {**base, "month_from": f"{p[0]}-{p[1]:02d}", "month_to": f"{p[0]}-{p[2]:02d}"})
            if t.get("campaign_count") and t.get(metric) is not None:
                series.append((qp._period_label(*p), t[metric]))
        unit = "complete periods"
    else:
        rows = [r for r in dt.metric_table(base, group_by="month")
                if r.get(metric) is not None and qp._complete(int(str(r["group"])[:4]), int(str(r["group"])[5:7]),
                                                              int(str(r["group"])[5:7]))]
        series = sorted(((str(r["group"]), r[metric]) for r in rows), key=lambda x: x[0])
        unit = "complete months"
    if len(series) < 2:
        f.statements.append(f"There aren't enough complete periods to describe a {label(metric)} trend.")
        return
    (p0, v0), (p1, v1) = series[0], series[-1]
    f.direction = "increase" if v1 > v0 else "decrease" if v1 < v0 else "flat"
    change = f" ({(v1 - v0) / abs(v0) * 100:+.1f}%)" if v0 else ""
    verb = {"increase": "rose", "decrease": "fell", "flat": "stayed"}[f.direction]
    f.statements.append(f"{Label(metric)} {verb} from {fmt(metric, v0)} in {p0} to {fmt(metric, v1)} in {p1}{change}, "
                        f"over {len(series)} {unit}.")
    hi = max(series, key=lambda x: x[1])
    lo = min(series, key=lambda x: x[1])
    f.lowest_period = lo[0]
    if len(series) > 2:
        f.statements.append(f"The highest was {fmt(metric, hi[1])} in {hi[0]} and the lowest {fmt(metric, lo[1])} in {lo[0]}.")
    steps = qp.direction([v for _, v in series])
    if steps == "mixed" and len(series) > 2:
        f.statements.append("It moved in both directions between periods rather than steadily.")


def _trend_entities(plan, base, f: Findings):
    """Trend for several named values (or a dimension): each value's first and last complete month."""
    metric, dim = plan.metrics[0], plan.dimension
    key = ax.ENTITY_FILTER_KEYS.get(dim)
    names = plan.entities or [str(r["group"]) for r in _rows(base, dim, metric)][:MAX_TREND_ENTITIES]
    f.kind, f.metric, f.dimension = "trend", metric, dim
    if key is None:
        return _summary(plan, base, f)
    changes = []
    for name in names[:MAX_TREND_ENTITIES]:
        rows = sorted((r for r in dt.metric_table({**base, key: name}, group_by="month") if r.get(metric) is not None
                       and qp._complete(int(str(r["group"])[:4]), int(str(r["group"])[5:7]), int(str(r["group"])[5:7]))),
                      key=lambda r: str(r["group"]))
        if len(rows) >= 2:
            a, b = rows[0], rows[-1]
            changes.append((name, a[metric], b[metric], str(a["group"]), str(b["group"])))
    for name, a, b, m0, m1 in changes:
        verb = "rose" if b > a else "fell" if b < a else "stayed"
        f.statements.append(f"{name}: {label(metric)} {verb} from {fmt(metric, a)} in {m0} to {fmt(metric, b)} in {m1}.")
    if changes:
        up = [c for c in changes if c[2] > c[1]]
        f.direction = "increase" if len(up) > len(changes) / 2 else "decrease" if len(up) < len(changes) / 2 else "flat"


def _relationship(plan, base, f: Findings):
    x, y = plan.metrics
    f.kind, f.x, f.y = "relationship", x, y
    if plan.dimension:
        rows = [r for r in dt.metric_table(base, group_by=plan.dimension) if r.get(x) is not None and r.get(y) is not None]
        n, r_value, what = len(rows), None, plural(plan.dimension)
        if n >= 3:
            import numpy as np
            xs, ys = np.array([r[x] for r in rows], float), np.array([r[y] for r in rows], float)
            r_value = round(float(np.corrcoef(xs, ys)[0, 1]), 3) if xs.std() > 0 and ys.std() > 0 else None
    else:
        res = dt.campaign_points(x, y, base)
        n, r_value, what = res["count"], res["correlation"], "campaigns"
    if r_value is None or (isinstance(r_value, float) and math.isnan(r_value)):
        f.statements.append(f"There are too few {what} to measure how {label(x)} and {label(y)} move together.")
        return
    f.correlation = r_value
    f.statements.append(f"{Label(x)} and {label(y)} show {analytics_engine._strength(r_value)} across {n:,} {what} "
                        f"(correlation r = {r_value:.3f}).")
    f.statements.append("A correlation shows how the two move together, not that one causes the other.")


def _campaigns(plan, base, f: Findings):
    f.kind = "campaigns"
    conds = plan.conditions
    sort_by = plan.metrics[0] if plan.metrics[0] in ("spend", "revenue", "profit", "roas", "cpa", "conversion_rate") \
        else (conds[0]["field"] if conds else "profit")
    order = "asc" if any(c["field"] == "profit" and c["operator"] in ("<", "<=") for c in conds) else plan.order
    res = ax.run("filter_campaigns", base, conditions=conds, sort_by=sort_by, order=order, limit=3,
                 include_aggregates=True)
    total = ax.run("get_totals", base).get("campaign_count") or 0
    n = res["matched_count"]
    loss = any(c["field"] == "profit" and c["operator"] in ("<", "<=") and c["value"] <= 0 for c in conds)
    crit = " and ".join(f"{label(c['field'])} {c['operator']} {fmt(c['field'], c['value'])}" for c in conds) or None
    if loss:
        f.loss_campaigns = n
        share = f" ({n / total * 100:.1f}% of {total:,})" if total else ""
        f.statements.append(f"{n:,} campaigns{share} are loss-making ({dt.UNSUCCESSFUL_RULE}).")
        agg = res.get("aggregates") or {}
        if n and agg.get("profit") is not None:
            f.statements.append(f"Together they had a profit of {fmt('profit', agg['profit'])} on spend of "
                                f"{fmt('spend', agg.get('spend'))}.")
        if res["campaigns"]:
            c = res["campaigns"][0]
            f.statements.append(f"The largest loss is {c['id']} ({c['platform']}) with a profit of {fmt('profit', c['profit'])}.")
    elif crit:
        f.statements.append(f"{n:,} of {total:,} campaigns have {crit}.")
    if not loss and res["campaigns"]:
        c = res["campaigns"][0]
        f.statements.append(f"The {'lowest' if order == 'asc' else 'highest'} by {label(sort_by)} is {c['id']} "
                            f"({c['platform']}) at {fmt(sort_by, c[sort_by])}.")
    f.metric = sort_by


def _summary(plan, base, f: Findings):
    """Shapes without a dedicated finding: the deterministic DuckDB answer itself, unchanged."""
    f.kind = "summary"
    try:
        f.statements.extend(line for line in qp.execute_plan(plan, {k: v for k, v in base.items()
                                                                     if k not in ("month_from", "month_to")}).lines
                            if line)
    except qp.PlanError as e:
        f.statements.append(e.public)


def find(plan, filters: dict) -> Findings:
    """Findings for one validated plan. Raises analytics_executor.AnalyticsError for data-layer
    failures (the caller's safe error path)."""
    f = Findings(metric=plan.metrics[0] if plan.metrics else None, dimension=plan.dimension)
    try:
        base = _scope_filters(plan, filters)
    except qp.PlanError as e:
        f.statements.append(e.public)
        return f
    if plan.intent == "relationship" and len(plan.metrics) == 2:
        _relationship(plan, base, f)
    elif plan.intent == "campaign_list":
        _campaigns(plan, base, f)
    elif plan.intent in ("trend", "period_comparison") and (plan.entities or plan.dimension):
        _trend_entities(plan, base, f)
    elif plan.intent in ("trend", "period_comparison"):
        _trend(plan, base, f)
    elif plan.intent == "breakdown" and ax.ENTITY_FILTER_KEYS.get(plan.dimension) in plan.filters:
        # A breakdown filtered to one value of its own dimension has one row: that value in context instead.
        _total(copy.replace(plan, intent="total", dimension=None), filters, f)
    elif plan.intent == "breakdown" and plan.dimension:
        _ranking(plan, base, f)
    elif plan.intent == "compare_entities" and plan.dimension and len(plan.entities) >= 2:
        _ranking(plan, base, f, names=plan.entities)
    elif plan.intent == "total":
        _total(plan, filters, f)
    else:
        _summary(plan, base, f)
    return f
