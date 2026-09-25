"""
General analytics planner: questions the deterministic Query Router can't map.

    question -> LLM Adapter .plan() (one Groq call, strict JSON-schema output)
             -> validate_plan()   allowlists + real dataset values; nothing unvalidated runs
             -> existing data tools (get_totals, rank_dimension, compare_entities, filter_campaigns,
                trend_over_time) on DuckDB
             -> deterministic answer text (+ an optional chart_builder spec)

The LLM only turns language into a restricted plan. It never writes SQL, never sees the data and
never supplies a number: every figure, change, direction and ranking below is computed here from
data-tool results, so the answer needs no second LLM call and no grounding pass (it contains no
LLM-written text). Periods that the data covers only partly are reported, never compared.
"""
import calendar
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date

import chart_builder as cb
import data_tools as dt
import grounding
import query_router as qr
import schema_catalog as catalog

logger = logging.getLogger("marketingiq.query_router")

INTENTS = ["total", "breakdown", "trend", "period_comparison", "change_ranking", "compare_entities",
           "campaign_list", "relationship", "not_answerable"]
METRICS = list(catalog.METRICS)                          # schema_catalog.py: the allowlist
DIMENSIONS = list(dt.all_dimension_columns())
FILTER_DIMENSIONS = list(qr.ENTITY_FILTER_KEYS)          # dimension -> dashboard filter key
CONDITION_METRICS = sorted(dt.NUMERIC_FIELD_COLUMNS)
OPERATORS = [">", "<", ">=", "<="]
GRAINS = ["none", "month", "year"]
VISUALIZATIONS = ["none", "line", "bar", "grouped_bar", "stacked_bar", "stacked_percent", "pie", "scatter"]
REASONS = ["none", "unknown_metric", "unknown_dimension", "ambiguous", "unavailable_period", "not_about_data"]
MAX_METRICS, MAX_ENTITIES, MAX_FILTERS, MAX_CONDITIONS = 3, 10, 5, 3
MAX_ROWS = cb.MAX_CATEGORIES         # rows shown / charted
MAX_LIMIT = 100                      # larger requested limits are rejected, 21-100 are capped
MAX_PLAN_CHARS = 4000
SAFE_VALUE_RE = re.compile(r"^[\w .,&+/()'%-]{1,60}$")
FAILED = qr.PLANNER_HINT

_NONE_SENTINELS = {"dimension": "none", "grain": "none", "visualization": "none", "reason": "none"}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "metrics", "dimension", "entities", "filters", "months", "years", "grain",
                 "conditions", "order", "limit", "visualization", "reason"],
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "metrics": {"type": "array", "items": {"type": "string", "enum": METRICS}},
        "dimension": {"type": "string", "enum": DIMENSIONS + ["none"]},
        "entities": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["dimension", "value"],
            "properties": {"dimension": {"type": "string", "enum": FILTER_DIMENSIONS}, "value": {"type": "string"}}}},
        "months": {"type": "array", "items": {"type": "integer"}},
        "years": {"type": "array", "items": {"type": "integer"}},
        "grain": {"type": "string", "enum": GRAINS},
        "conditions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["metric", "operator", "value"],
            "properties": {"metric": {"type": "string", "enum": CONDITION_METRICS},
                           "operator": {"type": "string", "enum": OPERATORS}, "value": {"type": "number"}}}},
        "order": {"type": "string", "enum": ["desc", "asc", "none"]},
        "limit": {"type": "integer"},
        "visualization": {"type": "string", "enum": VISUALIZATIONS},
        "reason": {"type": "string", "enum": REASONS},
        "dimension2": {"type": "string", "enum": DIMENSIONS + ["none"]},
        "exclude": {"type": "array", "items": {"type": "string"}},
    },
}
CORE_FIELDS = set(SCHEMA["required"])
EXTENDED_FIELDS = {"dimension2", "exclude"}   # optional in a plan (older plans have 13 fields)
SCHEMA["required"] = SCHEMA["required"] + sorted(EXTENDED_FIELDS)  # strict mode: Groq always sends them

LABELS = cb.LABELS


class PlanError(ValueError):
    """The plan broke the contract. `public` is what the user may be told."""

    def __init__(self, detail: str, public: str = FAILED):
        super().__init__(detail)
        self.public = public


@dataclass
class Plan:
    intent: str
    metrics: list
    dimension: str = None
    entities: list = field(default_factory=list)
    filters: dict = field(default_factory=dict)       # dashboard filter key -> dataset value
    months: list = field(default_factory=list)
    years: list = field(default_factory=list)
    grain: str = None
    conditions: list = field(default_factory=list)    # data_tools filter_campaigns conditions
    order: str = "desc"
    limit: int = None
    visualization: str = None
    reason: str = None
    notes: list = field(default_factory=list)
    dimension2: str = None                            # second breakdown (series), e.g. platform x device
    exclude: list = field(default_factory=list)       # values of the dimension left out ("remove TikTok")


@dataclass
class PlannedAnswer:
    lines: list                      # plain text; the caller escapes it
    chart: dict = None
    status: str = "ok"               # ok / not_answerable / invalid_plan / llm_error / no_data
    plan: Plan = None
    charts: list = None              # when a result needs several charts (incompatible units)


# ---------------------------------------------------------------------------
# Dataset facts (values and date bounds), read once from the data layer
# ---------------------------------------------------------------------------

_facts = None


def _dataset():
    global _facts
    if _facts is None:
        fields = dt.list_available_fields()
        values = {d.name: {v.lower(): v for v in d.values} for d in catalog.dimensions().values()}
        start, end = (date.fromisoformat(d) for d in fields["date_range"])
        _facts = {"values": values, "start": start, "end": end}
    return _facts


def system_prompt() -> str:
    facts = _dataset()
    dims = "\n".join(f"- {d}: {', '.join(sorted(facts['values'][d].values()))}" if d in facts["values"]
                     else f"- {d} (many values; breakdowns only)" for d in DIMENSIONS)
    return f"""You translate one question about the MarketingIQ advertising dataset into a JSON analytics plan.
You never answer the question and never write numbers from the data: the backend computes every figure.

Dataset: advertising campaigns with start dates {facts['start']} to {facts['end']}.
{catalog.prompt_text()}
"Performance" with no metric means metrics revenue, spend, roas. A metric or field not listed:
intent not_answerable, reason unknown_metric (or unknown_dimension).
Intents:
- total: one figure for a metric, optionally filtered or for a period.
- breakdown: a metric by one dimension ("revenue by platform", "which platform had the highest ROAS").
- trend: a metric over time; grain month (default) or year.
- period_comparison: the same calendar month(s) compared across years (January = months [1]; years []
  means every year in the data), or whole years compared (grain year, months []). Use it for "is
  January revenue increasing over time", "year over year", "across years", "2024 vs 2025".
- change_ranking: which value of a dimension changed or improved the most over time.
- compare_entities: two or more named values of one dimension (entities: exact values).
- campaign_list: individual campaigns, e.g. conditions [{{"metric": "spend", "operator": ">", "value": 10000}}].
- relationship: how two metrics move together (metrics [x, y]); per campaign, or per value of a dimension.
Up to 3 metrics. dimension2 = a second breakdown shown as series ("conversions by device and platform":
dimension device, dimension2 platform). With several entities over time use intent trend with entities.
exclude = dimension values to leave out. Chart types: line, bar, grouped_bar, stacked_bar (parts of a
whole), stacked_percent (100% shares), pie (one additive metric), scatter (relationship).
- not_answerable: the question can't be expressed with these fields (set reason: unknown_metric,
  unknown_dimension, ambiguous, unavailable_period or not_about_data).
Fields: filters narrow the data (e.g. device Mobile, platform TikTok); years/months only when the
question names them; order desc = highest / most improved first (none = default); limit = N of "top N",
else 0; visualization = none unless the user explicitly asks for a chart or graph. Unused fields: [] or
"none" or 0."""


# ---------------------------------------------------------------------------
# Validation: the only gate between LLM output and the data layer
# ---------------------------------------------------------------------------

def _require(ok: bool, detail: str, public: str = FAILED):
    if not ok:
        raise PlanError(detail, public)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _value_of(dimension: str, raw) -> str:
    """The dataset's own spelling of a value, or PlanError. The LLM's string is never used as is."""
    _require(isinstance(raw, str) and SAFE_VALUE_RE.match(raw) is not None, "unsafe value")
    canonical = _dataset()["values"].get(dimension, {}).get(raw.strip().lower())
    _require(canonical is not None, f"unknown {dimension} value",
             f"\"{raw.strip()}\" isn't a {dimension.replace('_', ' ')} in the MarketingIQ data.")
    return canonical


def _data_end() -> str:
    return f"{_dataset()['end']:%d %B %Y}".lstrip("0")


def _period_message() -> str:
    f = _dataset()
    return f"The MarketingIQ data covers campaigns starting {f['start']:%d %B %Y} to {f['end']:%d %B %Y}."


def validate_plan(text) -> Plan:
    """Parse and check raw planner output. Strict: unknown keys, wrong types, values outside the
    allowlists or the dataset, and malformed JSON are rejected, never repaired."""
    _require(isinstance(text, str) and 0 < len(text) <= MAX_PLAN_CHARS, "missing or oversized plan")
    try:
        raw = json.loads(text)
    except ValueError:
        raise PlanError("plan is not JSON")
    _require(isinstance(raw, dict), "plan is not an object")
    _require(CORE_FIELDS <= set(raw) <= CORE_FIELDS | EXTENDED_FIELDS, "unexpected or missing plan fields")

    intent, metrics = raw["intent"], raw["metrics"]
    _require(intent in INTENTS, "unknown intent")
    reason = raw["reason"]
    _require(reason in REASONS, "unknown reason")
    if intent == "not_answerable":
        public = {"unknown_metric": "That metric isn't in the MarketingIQ data. Available metrics: revenue, "
                                    "spend, profit, ROAS, ROI, CPA, CPC, CTR, conversion rate, conversions, "
                                    "clicks and impressions.",
                  "unknown_dimension": "That breakdown isn't in the MarketingIQ data. Available breakdowns: "
                                       + ", ".join(d.replace("_", " ") for d in DIMENSIONS) + ".",
                  "unavailable_period": _period_message()}.get(reason, FAILED)
        raise PlanError(f"not answerable ({reason})", public)

    _require(isinstance(metrics, list) and 1 <= len(metrics) <= MAX_METRICS, "bad metrics")
    _require(all(isinstance(m, str) and m in METRICS for m in metrics), "unknown metric")
    _require(len(set(metrics)) == len(metrics), "duplicate metrics")

    dimension = raw["dimension"]
    _require(isinstance(dimension, str) and dimension in DIMENSIONS + ["none"], "unknown dimension")
    dimension = None if dimension == "none" else dimension

    dimension2 = raw.get("dimension2", "none")
    _require(isinstance(dimension2, str) and dimension2 in DIMENSIONS + ["none"], "unknown dimension2")
    dimension2 = None if dimension2 == "none" else dimension2
    if dimension2:
        _require(dimension is not None and dimension2 != dimension, "dimension2 needs a different dimension")
        if dimension2 not in FILTER_DIMENSIONS:  # series are built by filtering on dimension2's values
            _require(dimension in FILTER_DIMENSIONS, "neither dimension can be split",
                     f"I can't cross {dimension.replace('_', ' ')} with {dimension2.replace('_', ' ')}.")
            dimension, dimension2 = dimension2, dimension

    entities = raw["entities"]
    _require(isinstance(entities, list) and len(entities) <= MAX_ENTITIES, "bad entities")
    entity_dimension = dimension
    if entities:
        if dimension2 and all(isinstance(e, str) and e.strip().lower() in _dataset()["values"].get(dimension2, {})
                              for e in entities):
            entity_dimension = dimension2  # series restricted to these values
        _require(entity_dimension is not None and entity_dimension in _dataset()["values"],
                 "entities without a listable dimension")
        entities = list(dict.fromkeys(_value_of(entity_dimension, e) for e in entities))
    exclude = raw.get("exclude", [])
    _require(isinstance(exclude, list) and len(exclude) <= MAX_ENTITIES, "bad exclude")
    if exclude:
        _require(entity_dimension is not None, "exclude without a dimension")
        exclude = list(dict.fromkeys(_value_of(entity_dimension, e) for e in exclude))

    filters = {}
    _require(isinstance(raw["filters"], list) and len(raw["filters"]) <= MAX_FILTERS, "bad filters")
    for flt in raw["filters"]:
        _require(isinstance(flt, dict) and set(flt) == {"dimension", "value"}, "bad filter")
        _require(flt["dimension"] in FILTER_DIMENSIONS, "unknown filter dimension")
        filters[qr.ENTITY_FILTER_KEYS[flt["dimension"]]] = _value_of(flt["dimension"], flt["value"])

    facts = _dataset()
    months, years = raw["months"], raw["years"]
    _require(isinstance(months, list) and len(months) <= 12 and all(_int(m) and 1 <= m <= 12 for m in months),
             "bad months")
    months = sorted(set(months))
    _require(not months or months == list(range(months[0], months[-1] + 1)), "months must be one contiguous range")
    _require(isinstance(years, list) and len(years) <= 5 and all(_int(y) for y in years), "bad years")
    years = sorted(set(years))
    _require(all(facts["start"].year <= y <= facts["end"].year for y in years), "year outside the data",
             _period_message())

    grain = raw["grain"]
    _require(grain in GRAINS, "unknown grain")
    grain = None if grain == "none" else grain

    conditions = []
    _require(isinstance(raw["conditions"], list) and len(raw["conditions"]) <= MAX_CONDITIONS, "bad conditions")
    for c in raw["conditions"]:
        _require(isinstance(c, dict) and set(c) == {"metric", "operator", "value"}, "bad condition")
        _require(c["metric"] in CONDITION_METRICS and c["operator"] in OPERATORS, "unknown condition field/operator")
        v = c["value"]
        _require(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) < 1e12,
                 "bad condition value")
        conditions.append({"field": c["metric"], "operator": c["operator"], "value": float(v)})

    _require(raw["order"] in ("desc", "asc", "none"), "bad order")
    limit = raw["limit"]
    _require(_int(limit) and 0 <= limit <= MAX_LIMIT, "bad limit")
    notes = []
    if limit > MAX_ROWS:
        notes.append(f"Showing at most {MAX_ROWS} rows.")
    limit = min(limit, MAX_ROWS) or None
    viz = raw["visualization"]
    _require(viz in VISUALIZATIONS, "unknown visualization")

    needs = {"breakdown": dimension is not None, "change_ranking": dimension is not None and len(metrics) == 1,
             "compare_entities": dimension is not None and len(entities) >= 2,
             "campaign_list": metrics[0] in cb.CAMPAIGN_METRICS or bool(conditions),
             "period_comparison": bool(months) or grain == "year",
             "relationship": len(metrics) == 2 and (dimension is not None
                                                    or all(m in dt.CAMPAIGN_FIELD_COLUMNS for m in metrics))}
    _require(needs.get(intent, True), f"incomplete {intent} plan")
    averages = [m for m in metrics if catalog.METRICS[m].aggregation == "avg"]
    _require(not (averages and intent in ("period_comparison", "change_ranking", "campaign_list")),
             "average metric in a period plan",
             f"Year-over-year and change rankings aren't available for {catalog.lower_label(averages[0])} yet; "
             "try it by month or by a breakdown." if averages else FAILED)
    order = "asc" if raw["order"] == "asc" else "desc"
    viz = {"none": None, "grouped_bar": "bar"}.get(viz, viz)
    return Plan(intent, metrics, dimension, entities, filters, months, years, grain, conditions, order,
                limit, viz, None if reason == "none" else reason, notes, dimension2, exclude)


_FILTER_DIMENSION = {key: dim for dim, key in qr.ENTITY_FILTER_KEYS.items()}


def plan_to_raw(plan: Plan) -> dict:
    """A Plan back in the planner's JSON form, so any plan (e.g. one edited by a conversation
    follow-up) can be re-checked by validate_plan() before it runs."""
    return {"intent": plan.intent, "metrics": list(plan.metrics), "dimension": plan.dimension or "none",
            "entities": list(plan.entities),
            "filters": [{"dimension": _FILTER_DIMENSION[k], "value": v} for k, v in plan.filters.items()
                        if k in _FILTER_DIMENSION],
            "months": list(plan.months), "years": list(plan.years), "grain": plan.grain or "none",
            "conditions": [{"metric": c["field"], "operator": c["operator"], "value": c["value"]} for c in plan.conditions],
            "order": plan.order, "limit": plan.limit or 0, "visualization": plan.visualization or "none",
            "reason": "none", "dimension2": plan.dimension2 or "none", "exclude": list(plan.exclude)}


# ---------------------------------------------------------------------------
# Execution: existing data tools only; every conclusion computed here
# ---------------------------------------------------------------------------

def _fmt(metric: str, value) -> str:
    return grounding.fmt(metric, value)


def _last_day(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _complete(year: int, m_from: int, m_to: int) -> bool:
    f = _dataset()
    return f["start"] <= date(year, m_from, 1) and _last_day(year, m_to) <= f["end"]


def _in_data(year: int, m_from: int, m_to: int) -> bool:
    f = _dataset()
    return date(year, m_from, 1) <= f["end"] and _last_day(year, m_to) >= f["start"]


def _period_label(year: int, m_from: int, m_to: int) -> str:
    if (m_from, m_to) == (1, 12):
        return str(year)
    if m_from == m_to:
        return f"{calendar.month_name[m_from]} {year}"
    return f"{calendar.month_abbr[m_from]}-{calendar.month_abbr[m_to]} {year}"


def _change(a: float, b: float) -> str:
    return f"{(b - a) / abs(a) * 100:+.1f}%" if a else "n/a"


def direction(values: list) -> str:
    """Deterministic: increasing / decreasing / unchanged / mixed across consecutive periods."""
    steps = [b - a for a, b in zip(values, values[1:])]
    if all(s == 0 for s in steps):
        return "unchanged"
    if all(s > 0 for s in steps):
        return "increasing"
    if all(s < 0 for s in steps):
        return "decreasing"
    return "mixed"


def _run(tool: str, filters: dict, **kwargs):
    call = qr.ToolCall(tool, kwargs)
    return qr._execute(call, filters)


def _period_filter(plan: Plan) -> dict:
    """One calendar range from the plan's years/months (for totals, breakdowns and lists)."""
    f = _dataset()
    if not plan.years and not plan.months:
        return {}
    y0, y1 = (plan.years[0], plan.years[-1]) if plan.years else (f["start"].year, f["end"].year)
    m0, m1 = (plan.months[0], plan.months[-1]) if plan.months else (1, 12)
    if plan.months and not plan.years and y0 != y1:
        raise PlanError("month without a year", "Which year do you mean? " + _period_message())
    return {"month_from": f"{y0}-{m0:02d}", "month_to": f"{y1}-{m1:02d}"}


def _scope_line(plan: Plan, period: dict) -> str:
    parts = [f"{k} = {v}" for k, v in plan.filters.items()]
    if period:
        parts.append(f"period {period['month_from']} to {period['month_to']}")
    return f"Scope: {', '.join(parts)}." if parts else ""


def _partial_note(period: dict) -> str:
    if not period:
        return ""
    y0, m0 = map(int, period["month_from"].split("-"))
    y1, m1 = map(int, period["month_to"].split("-"))
    f = _dataset()
    if date(y0, m0, 1) < f["start"] or _last_day(y1, m1) > f["end"]:
        return f"Note: the data only partly covers this period. {_period_message()}"
    return ""


def _chart(spec) -> dict:
    try:
        return cb.validate_chart(spec)
    except (cb.ChartSpecError, KeyError, TypeError, ValueError):
        return None


def execute_plan(plan: Plan, filters: dict) -> PlannedAnswer:
    import analytics_engine  # generic shapes (several metrics/series, extra fields); imported here: it uses this module
    if analytics_engine.handles(plan):
        answer = analytics_engine.execute(plan, filters)
        answer.lines += plan.notes
        answer.plan = plan
        return answer
    base = {**(filters or {}), **plan.filters}
    handler = {"total": _total, "breakdown": _breakdown, "trend": _trend, "period_comparison": _periods,
               "change_ranking": _change_ranking, "compare_entities": _compare, "campaign_list": _campaigns}[plan.intent]
    answer = handler(plan, base)
    answer.lines += plan.notes
    answer.plan = plan
    return answer


def _total(plan, base):
    period = _period_filter(plan)
    t = _run("get_totals", {**base, **period})
    if not t["campaign_count"]:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    lines = [f"{LABELS[m]}: {_fmt(m, t[m])}" for m in plan.metrics]
    lines.append(f"Based on {t['campaign_count']:,} campaigns.")
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, period), _partial_note(period)) if x])


def _breakdown(plan, base):
    period = _period_filter(plan)
    metric = plan.metrics[0]
    res = _run("rank_dimension", {**base, **period}, dimension=plan.dimension, metric=metric, order=plan.order,
               limit=plan.limit or MAX_ROWS)
    rows = res["results"]
    if not rows:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    dim = plan.dimension.replace("_", " ")
    word = "Lowest" if plan.order == "asc" else "Highest"
    lines = [f"{word} {LABELS[metric]} by {dim}: {rows[0]['name']} ({_fmt(metric, rows[0][metric])})."]
    for i, r in enumerate(rows, 1):
        extra = "".join(f", {LABELS[m]} {_fmt(m, r[m])}" for m in plan.metrics[1:])
        lines.append(f"{i}. {r['name']}: {_fmt(metric, r[metric])}{extra}")
    chart = None
    if plan.visualization:
        kind = "pie" if plan.visualization == "pie" else "bar"
        if kind == "pie" and metric not in cb.PIE_METRICS:
            lines.append(f"{LABELS[metric]} values don't add up to a total, so a pie chart would mislead; "
                         "shown as a bar chart instead.")
            kind = "bar"
        elif plan.visualization == "line":
            lines.append("Shown as a bar chart: line charts are used for trends over time.")
        title = f"{LABELS[metric]} Share by {plan.dimension.replace('_', ' ').title()}" if kind == "pie" else None
        chart = _chart(cb.rank_spec(rows, plan.dimension, metric, kind, title))
        if chart is None and kind == "pie":
            lines.append("A pie chart would mislead here (too many slices or negative values); shown as a bar chart.")
            chart = _chart(cb.rank_spec(rows, plan.dimension, metric, "bar"))
        if chart is None:
            lines.append("A chart of this breakdown would be misleading, so the figures are shown as text.")
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, period), _partial_note(period)) if x], chart)


def _compare(plan, base):
    period = _period_filter(plan)
    res = _run("compare_entities", {**base, **period}, dimension=plan.dimension, names=plan.entities)
    rows = res["results"]
    if not rows:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    lines = [f"{r['name']}: " + ", ".join(f"{LABELS[m]} {_fmt(m, r[m])}" for m in plan.metrics) for r in rows]
    m = plan.metrics[0]
    ranked = sorted((r for r in rows if r.get(m) is not None), key=lambda r: r[m], reverse=True)
    if len(ranked) >= 2 and ranked[0][m] != ranked[1][m]:
        lines.insert(0, f"{ranked[0]['name']} has the highest {LABELS[m]} of those compared "
                        f"({_fmt(m, ranked[0][m])}); {ranked[-1]['name']} has the lowest ({_fmt(m, ranked[-1][m])}).")
    chart = _chart(cb.rank_spec(ranked, plan.dimension, m)) if plan.visualization else None
    if chart and plan.visualization in ("pie", "line"):
        lines.append("Shown as a bar chart: " + ("a few named values aren't parts of one whole."
                                                 if plan.visualization == "pie" else "line charts are for trends over time."))
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, period), _partial_note(period)) if x], chart)


def _campaigns(plan, base):
    period = _period_filter(plan)
    sort_by = plan.metrics[0] if plan.metrics[0] in cb.CAMPAIGN_METRICS else plan.conditions[0]["field"]
    limit = plan.limit or 10
    res = _run("filter_campaigns", {**base, **period}, conditions=plan.conditions, sort_by=sort_by,
               order=plan.order, limit=limit)
    lines = []
    if plan.conditions:
        crit = " and ".join(f"{LABELS.get(c['field'], c['field'])} {c['operator']} {_fmt(c['field'], c['value'])}"
                            for c in plan.conditions)
        lines.append(f"{res['matched_count']:,} campaigns have {crit}.")
    if res["campaigns"]:
        lines.append(f"{'Lowest' if plan.order == 'asc' else 'Highest'} {len(res['campaigns'])} by {LABELS.get(sort_by, sort_by)}:")
        lines += [f"{i}. {c['id']} ({c['platform']}): {LABELS.get(sort_by, sort_by)} {_fmt(sort_by, c[sort_by])}"
                  for i, c in enumerate(res["campaigns"], 1)]
    elif not plan.conditions:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    chart = None
    if plan.visualization and sort_by in cb.CAMPAIGN_METRICS and len(res["campaigns"]) > 1:
        chart = _chart(cb.campaign_spec(res["campaigns"], sort_by, plan.order))
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, period), _partial_note(period)) if x], chart)


def _series_periods(plan):
    """(year, m_from, m_to) periods for a comparison: the same months each year, or whole years."""
    f = _dataset()
    years = plan.years or list(range(f["start"].year, f["end"].year + 1))
    m0, m1 = (plan.months[0], plan.months[-1]) if plan.months else (1, 12)
    return [(y, m0, m1) for y in years if _in_data(y, m0, m1)]


def _periods(plan, base):
    periods = _series_periods(plan)
    if not periods:
        return PlannedAnswer([_period_message()], status="no_data")
    values = [(p, _run("get_totals", {**base, "month_from": f"{p[0]}-{p[1]:02d}", "month_to": f"{p[0]}-{p[2]:02d}"}))
              for p in periods]
    complete = [(p, t) for p, t in values if _complete(*p) and t["campaign_count"]]
    partial = [(p, t) for p, t in values if not _complete(*p) and t["campaign_count"]]
    what = (calendar.month_name[plan.months[0]] if len(plan.months) == 1 else
            f"{calendar.month_abbr[plan.months[0]]}-{calendar.month_abbr[plan.months[-1]]}" if plan.months else "")
    lines = []
    for m in plan.metrics:
        series = [(p, t[m]) for p, t in complete if t[m] is not None]
        lines.append(f"{what} {LABELS[m]} by year:".strip())
        for i, (p, v) in enumerate(series):
            delta = f" ({_change(series[i - 1][1], v)} vs {_period_label(*series[i - 1][0])})" if i else ""
            lines.append(f"- {_period_label(*p)}: {_fmt(m, v)}{delta}")
        if len(series) >= 2:
            d = direction([v for _, v in series])
            first, last = series[0], series[-1]
            verdict = {"increasing": "Yes, it is increasing", "decreasing": "No, it is decreasing",
                       "unchanged": "It is unchanged", "mixed": "It has moved in both directions"}[d]
            lines.append(f"{verdict}: {LABELS[m]} went from {_fmt(m, first[1])} ({_period_label(*first[0])}) to "
                         f"{_fmt(m, last[1])} ({_period_label(*last[0])}), {_change(first[1], last[1])} across "
                         f"{len(series)} complete periods.")
        else:
            lines.append(f"Only {len(series)} complete period is in the data, so there is no trend to measure.")
    for p, t in partial:
        figures = ", ".join(f"{LABELS[m]} {_fmt(m, t[m])}" for m in plan.metrics)
        lines.append(f"Not compared: {_period_label(*p)} is incomplete (the data ends on {_data_end()}); "
                     f"partial figure: {figures}.")
    scope = _scope_line(plan, {})
    chart = None
    if plan.visualization and complete:
        if plan.visualization in ("pie", "line"):
            lines.append("Shown as a bar chart: each bar is one period, and a pie can't show change over time."
                         if plan.visualization == "pie" else "Shown as a bar chart, one bar per period.")
        m = plan.metrics[0]
        rows = [{"name": _period_label(*p), m: t[m]} for p, t in complete if t[m] is not None]
        chart = _chart({"type": "bar", "title": f"{what} {LABELS[m]} by Year".strip(), "x_key": "name", "x_label": "Period",
                        "series": cb._series([m]), "data": rows})
    return PlannedAnswer([x for x in (*lines, scope) if x], chart)


def _trend(plan, base):
    if plan.grain == "year":
        return _periods(plan, base)
    period = _period_filter(plan)
    months = [p["month"] for p in _run("trend_over_time", {**base, **period}, metric="revenue")["series"]]
    if not months:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    totals = [(mo, _run("get_totals", {**base, "month_from": mo, "month_to": mo})) for mo in months]
    full = [(mo, t) for mo, t in totals if _complete(int(mo[:4]), int(mo[5:]), int(mo[5:]))]
    lines = []
    for m in plan.metrics:
        series = [(mo, t[m]) for mo, t in full if t[m] is not None]
        if len(series) < 2:
            lines.append(f"There aren't enough complete months to describe the {LABELS[m]} trend.")
            continue
        hi = max(series, key=lambda x: x[1])
        lo = min(series, key=lambda x: x[1])
        (m0, v0), (m1, v1) = series[0], series[-1]
        lines.append(f"{LABELS[m]}: {_fmt(m, v0)} in {m0} and {_fmt(m, v1)} in {m1} ({_change(v0, v1)}); "
                     f"highest {_fmt(m, hi[1])} in {hi[0]}, lowest {_fmt(m, lo[1])} in {lo[0]} "
                     f"({len(series)} complete months).")
    dropped = [mo for mo, _ in totals if mo not in dict(full)]
    if dropped:
        lines.append(f"Not included: {', '.join(dropped)} (incomplete; the data ends on {_data_end()}).")
    chart = None
    if plan.visualization and full:
        ms = [m for m in plan.metrics if all(t[m] is not None for _, t in full)]
        units = {cb.UNITS[m] for m in ms}
        if plan.visualization == "pie":
            lines.append("A pie chart can't show change over time; shown as a line chart instead.")
        if ms and len(units) == 1:
            results = {m: {"series": [{"month": mo, "value": t[m]} for mo, t in full]} for m in ms}
            chart = _chart(cb.trend_spec(results, "bar" if plan.visualization == "bar" and len(ms) == 1 else "line"))
        elif ms:
            lines.append("Those metrics use different units, so they aren't drawn on one chart.")
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, period)) if x], chart)


def _change_ranking(plan, base):
    f = _dataset()
    years = [y for y in (plan.years or range(f["start"].year, f["end"].year + 1)) if _complete(y, 1, 12)]
    metric = plan.metrics[0]
    if len(years) < 2:
        return PlannedAnswer([f"Measuring change needs at least two complete years. {_period_message()}"],
                             status="no_data")
    y0, y1 = years[0], years[-1]
    by_year = {y: {r["name"]: r[metric] for r in _run("rank_dimension", {**base, "month_from": f"{y}-01",
                                                                         "month_to": f"{y}-12"},
                                                     dimension=plan.dimension, metric=metric, limit=1000)["results"]}
               for y in (y0, y1)}
    names = [n for n in by_year[y0] if n in by_year[y1] and by_year[y0][n] is not None and by_year[y1][n] is not None]
    if not names:
        return PlannedAnswer(["No campaigns match that selection."], status="no_data")
    lower_better = metric in grounding.LOWER_IS_BETTER
    # "Improved" = up for most metrics, down for costs (CPA, CPC).
    improvement = {n: (by_year[y0][n] - by_year[y1][n]) if lower_better else (by_year[y1][n] - by_year[y0][n])
                   for n in names}
    ranked = sorted(names, key=lambda n: improvement[n], reverse=(plan.order == "desc"))[:plan.limit or MAX_ROWS]
    top = ranked[0]
    word = "improved the most" if plan.order == "desc" else "improved the least (or declined the most)"
    lines = [f"{top} {word} in {LABELS[metric]}: {_fmt(metric, by_year[y0][top])} in {y0} to "
             f"{_fmt(metric, by_year[y1][top])} in {y1} ({_change(by_year[y0][top], by_year[y1][top])}).",
             f"{LABELS[metric]} by {plan.dimension.replace('_', ' ')}, {y0} vs {y1}, ranked by absolute change "
             f"(complete years only{'; lower is better' if lower_better else ''}):"]
    lines += [f"{i}. {n}: {_fmt(metric, by_year[y0][n])} to {_fmt(metric, by_year[y1][n])} "
              f"({_change(by_year[y0][n], by_year[y1][n])})" for i, n in enumerate(ranked, 1)]
    if not _complete(f["end"].year, 1, 12):
        lines.append(f"{f['end'].year} is not included: it is incomplete (the data ends on {_data_end()}).")
    if plan.visualization:
        lines.append("Change rankings are shown as text.")
    return PlannedAnswer([x for x in (*lines, _scope_line(plan, {})) if x])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def answer(question: str, filters: dict, adapter, previous: Plan = None) -> PlannedAnswer:
    """One planning call at most, then data tools only. Never raises for LLM or plan problems.
    `previous`: the conversation's last analysis, sent as its compact plan (never the transcript) so
    references like "that" or "instead" can be resolved."""
    if adapter is None or not getattr(adapter.provider, "configured", False):
        return PlannedAnswer([FAILED], status="llm_error")
    prompt = question
    if previous is not None:
        prompt = (f"Previous analysis in this conversation, as a plan (resolve words like this, that, same, "
                  f"instead or it against it): {json.dumps(plan_to_raw(previous))}\nQuestion: {question}")
    result = adapter.plan(prompt, system_prompt(), SCHEMA)
    if result["status"] != "ok":
        return PlannedAnswer([FAILED], status="llm_error")
    try:
        plan = validate_plan(result["text"])
    except PlanError as e:
        status = "not_answerable" if str(e).startswith("not answerable") else "invalid_plan"
        logger.warning(json.dumps({"event": "plan_rejected", "status": status, "detail": str(e)[:80]}))
        return PlannedAnswer([e.public], status=status)
    if not cb.VIZ_RE.search(question):
        plan.visualization = None  # a chart only when the user asked for one, whatever the plan says
    try:
        out = execute_plan(plan, filters)
    except PlanError as e:
        return PlannedAnswer([e.public], status="invalid_plan", plan=plan)
    logger.info(json.dumps({"event": "plan_executed", "intent": plan.intent, "metrics": plan.metrics,
                            "dimension": plan.dimension, "status": out.status, "chart": bool(out.chart)}))
    return out
