"""
Deterministic, schema-aware parsing of analytical questions whose SHAPE the Query Router has no tool
plan for: several entities over time, several metrics across a breakdown, two breakdowns
(platform x device), relationships between two measures, and fields only the schema catalog knows
(bounce rate, day of week, ...). Words are mapped with the catalog's aliases; the result is a
query_planner.Plan checked by the same validator as the Groq planner's plans. 0 LLM calls.

Simple shapes (one metric by one breakdown, totals, one entity, thresholds, campaign lists, period
comparisons, explanations) return None and keep their existing paths.
"""
import json
import re

import chart_builder as cb
import data_tools as dt
import query_planner as qp
import query_router as qr
import schema_catalog as catalog

LEGACY_METRICS = {"revenue", "spend", "profit", "roas", "roi_pct", "cpa", "cpc", "ctr", "conversion_rate",
                  "conversions", "clicks", "impressions", "campaign_count"}
DEFAULT_METRICS = ["revenue", "spend", "roas"]
_c = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
RELATION_RE = _c(r"\b(?:relationship|relation|relate[sd]?|correlat\w*|scatter|association|associated)\b")
AXIS_RE = _c(r"\b(?:by|across|per|for each|between)\s+(?:the\s+)?(?:each\s+)?$")
CONTRIBUTION_RE = _c(r"\b(?:contribut\w*|composition|make[s]? up|share of|split of|mix)\b")


def requested_chart(text: str):
    """The chart type the user named, "auto" for a plain visualization request, else None."""
    share = r"(?:100\s*%|\bpercent(?:age)?\b|\bproportions?\b)"
    if re.search(share + r".*\bstack|\bstack\w*\b.*" + share, text, re.I):
        return "stacked_percent"
    if re.search(r"\bstack(?:ed)?\b", text, re.I):
        return "stacked_bar"
    if re.search(r"\bscatter\b", text, re.I):
        return "scatter"
    if re.search(r"\b(?:pie|donut|doughnut)\b", text, re.I):
        return "pie"
    if re.search(r"\b(?:grouped|side[- ]by[- ]side|bars?|columns?)\b", text, re.I):
        return "bar"
    if re.search(r"\blines?\b", text, re.I):
        return "line"
    if cb.VIZ_RE.search(text) or re.search(r"\b(?:charts?|graphs?|visuali[sz]\w*|plot)\b", text, re.I):
        return "auto"
    return None


def default_chart(intent: str, requested, contribution: bool = False):
    if requested != "auto":
        return requested
    if intent == "relationship":
        return "scatter"
    if intent == "trend":
        return "line"
    return "stacked_bar" if contribution else "bar"


def _axis_dimension(text: str, dims: list) -> str:
    """The breakdown written right after "by" / "across" / "per" is the x axis."""
    for name, pattern in catalog.DIMENSION_ALIAS_RES:
        if name not in dims:
            continue
        for m in pattern.finditer(text):
            if AXIS_RE.search(text[max(0, m.start() - 20):m.start()]):
                return name
    return dims[0]


def plan_for(question: str):
    text = " ".join(question.lower().split())
    if (qr.EXPLANATION_RE.search(text) or qr.UNAVAILABLE_RE.search(text) or qr.THRESHOLD_RE.search(text)
            or qr.BARE_MONTH_RE.search(qr.MONTH_RE.sub(" ", text)) or qr.YEAR_GRAIN_RE.search(text)
            or qr.UNCERTAIN_RE.search(qr.THRESHOLD_RE.sub(" ", text))):
        return None
    f = qr._parse(question, qr._get_entity_index())
    metrics = catalog.find_metrics(text)
    entity_dims = {d for d, _ in f.entities}
    dims = [d for d in catalog.find_dimensions(text) if d not in entity_dims]
    same = [v for d, v in f.entities if f.entities and d == f.entities[0][0]]
    trend = bool(qr.TREND_RE.search(text))
    relationship = bool(RELATION_RE.search(text)) and len(metrics) >= 2
    catalog_only = (any(m not in LEGACY_METRICS for m in metrics)
                    or any(d in dt.EXTRA_DIMENSION_COLUMNS for d in dims))
    complex_shape = (relationship or len(dims) >= 2 or (len(same) >= 2 and trend)
                     or (len(metrics) >= 2 and (dims or len(same) >= 2)) or catalog_only)
    if not complex_shape or (qr.CAMPAIGN_RE.search(text) and not relationship and not dims):
        return None

    raw = {"intent": "total", "metrics": (metrics or DEFAULT_METRICS)[:qp.MAX_METRICS], "dimension": "none",
           "entities": [], "filters": [], "months": [], "years": [], "grain": "none", "conditions": [],
           "order": "desc", "limit": 0, "visualization": "none", "reason": "none", "dimension2": "none", "exclude": []}
    if f.period:
        raw["years"] = [int(f.period[:4])]
        if len(f.period) == 7:
            raw["months"] = [int(f.period[5:7])]
    entity_dim = f.entities[0][0] if f.entities else None
    if relationship:
        raw.update(intent="relationship", metrics=metrics[:2])
        if dims:
            raw["dimension"] = dims[0]
    elif trend:
        raw.update(intent="trend", grain="month")
        if len(same) >= 2:
            raw.update(dimension=entity_dim, entities=same)
        elif dims:
            raw["dimension"] = dims[0]
    elif len(dims) >= 2 or (dims and len(same) >= 2):
        axis = _axis_dimension(text, dims)
        series_dim = entity_dim if len(same) >= 2 else next(d for d in dims if d != axis)
        raw.update(intent="breakdown", dimension=axis, dimension2=series_dim)
        if len(same) >= 2:
            raw["entities"] = same
    elif len(same) >= 2:
        raw.update(intent="compare_entities", dimension=entity_dim, entities=same)
    elif dims:
        raw.update(intent="breakdown", dimension=dims[0])
    used = {raw["dimension"], raw["dimension2"]} | ({entity_dim} if raw["entities"] else set())
    raw["filters"] = [{"dimension": d, "value": v} for d, v in f.entities
                      if d not in used and d in qp.FILTER_DIMENSIONS]
    if raw["intent"] in ("breakdown", "compare_entities") and raw["metrics"]:
        raw["order"] = qr._rank_order(text, raw["metrics"][0]) if raw["metrics"][0] in qr.RANK_METRICS else "desc"
    top = qr.TOP_N_RE.search(text)
    if top:
        raw["limit"] = min(int(top.group(1)), qp.MAX_LIMIT)
    requested = requested_chart(text)
    if requested is None and raw["intent"] == "relationship" and cb.SHOW_RE.search(text):
        requested = "auto"  # "show me the relationship between ..." is a request to see it
    if requested:
        raw["visualization"] = default_chart(raw["intent"], requested, bool(CONTRIBUTION_RE.search(text)))
        if raw["visualization"] == "grouped_bar":
            raw["visualization"] = "bar"
    try:
        return qp.validate_plan(json.dumps(raw))
    except qp.PlanError:
        return None
