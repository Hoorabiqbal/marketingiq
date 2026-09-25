"""
Dynamic charts for the AI Analyst chat: DuckDB -> existing data tools -> validated chart spec.

    explicit visualization request ("chart", "plot", "graph", "visualize", "line/bar/pie chart")
    or "compare <metrics> over time"
        -> Query Router parsing (metrics, dimensions, entities, period)
        -> existing data tools (trend_over_time, rank_dimension, filter_campaigns), via the router
        -> chart spec (data + metadata only) -> validate_chart() -> the chat response

No LLM is ever involved: every number comes from the data tools, and a request that can't be
mapped safely gets a clarification instead of a guessed chart. The spec never carries code or
HTML; the frontend draws it from data (site/dashboard.html, renderChatChart).
"""
import math
import re
from dataclasses import dataclass, field

import grounding
import query_router as qr

CHART_TYPES = {"line", "bar", "pie"}
MAX_CATEGORIES = 20   # bars / top-N items in one chat chart
MAX_PIE_SLICES = 10
MAX_SERIES = 3
MAX_POINTS = 120      # monthly points (the dataset spans 25 months)
X_KEYS = {"month": "Month", "campaign": "Campaign", "name": None}  # None: the dimension's own label
UNITS = {m: unit for m, (unit, _) in grounding.METRICS.items()}
LABELS = {"campaign_count": "Campaigns", "spend": "Spend", "revenue": "Revenue", "profit": "Profit",
          "conversions": "Conversions", "clicks": "Clicks", "impressions": "Impressions", "roas": "ROAS",
          "roi_pct": "ROI", "cpa": "CPA", "cpc": "CPC", "ctr": "CTR", "conversion_rate": "Conversion Rate"}
# Additive, never-negative-by-definition totals: the only metrics a pie can split into parts.
PIE_METRICS = {"spend", "revenue", "conversions", "clicks", "impressions", "campaign_count", "profit"}
CAMPAIGN_METRICS = {"spend", "revenue", "profit", "roas", "cpa", "conversion_rate"}  # filter_campaigns columns

_c = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
VIZ_RE = _c(r"\b(?:charts?|chart(?:ed|ing)|plot(?:s|ted|ting)?|graphs?|graph(?:ed|ing)|visuali[sz]\w*|diagrams?"
            r"|line (?:chart|graph)|bar (?:chart|graph)|column chart|pie(?: chart)?)\b")
SHOW_RE = _c(r"^\s*(?:please\s+)?(?:show|display|draw)\b")
COMPARE_OVER_TIME_RE = _c(r"\b(?:compare|comparison|vs\.?|versus|against)\b")
PIE_RE = _c(r"\bpie\b|\bdonut\b|\bdoughnut\b")
UNSUPPORTED_TYPE_RE = _c(r"\b(?:scatter|heat ?maps?|bubble|radar|histograms?|3d|3-d|maps?|area chart|funnel chart"
                         r"|treemap|sankey|gauge)\b")
SAFE_TEXT_RE = re.compile(r"^[^<>`\x00-\x1f\\]{1,160}$")
MONTH_KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
SCRIPT_RE = _c(r"javascript:|data:|on\w+\s*=|<script")

WHICH_METRIC = ("Which metric would you like me to visualize? For example: \"Show monthly revenue as a line "
                "chart\", \"Bar chart of ROAS by platform\" or \"Top 5 campaigns by revenue\".")
TYPES_HINT = "I can draw line charts (monthly trends), bar charts (comparisons and rankings) and pie charts (shares)."


class ChartSpecError(ValueError):
    """The chart spec broke the contract; it is never sent to the frontend."""


@dataclass
class ChartResult:
    chart: dict = None          # a validated spec, or None
    message: str = None         # plain text: a caption (with a chart) or a clarification (without)
    route: str = "CHART"        # CHART / NEEDS_CLARIFICATION / UNSUPPORTED
    unmapped: bool = False      # the request itself couldn't be mapped (the analytics planner may try)
    plan: dict = None           # the charted request as query_planner.Plan fields (conversation memory)
    filters_applied: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)


def wants_chart(question: str) -> bool:
    """Explicit visualization language, or "compare <two+ metrics> over time". A plain "show" is not
    enough ("Show me total revenue" stays a text answer); "why"-questions stay explanations."""
    text = question or ""
    if qr.EXPLANATION_RE.search(text.lower()):
        return False
    if VIZ_RE.search(text):
        return True
    if COMPARE_OVER_TIME_RE.search(text) and qr.TREND_RE.search(text.lower()):
        f = qr._parse(text, qr._get_entity_index())
        return len([m for m in qr._metric_names(f) if m in qr.TREND_METRICS]) >= 2
    return False


# ---------------------------------------------------------------------------
# Spec construction (from existing data-tool results)
# ---------------------------------------------------------------------------

def _series(metrics: list) -> list:
    return [{"key": m, "label": LABELS[m], "unit": UNITS[m]} for m in metrics]


def _month_order(month: str) -> tuple:
    y, m = month.split("-")
    return int(y), int(m)


def trend_spec(results: dict, chart_type: str = "line", title_suffix: str = "") -> dict:
    """results: metric -> trend_over_time() result. Months are merged and kept in calendar order."""
    metrics = list(results)
    by_month = [{p["month"]: p["value"] for p in results[m]["series"]} for m in metrics]
    months = sorted(set.intersection(*(set(s) for s in by_month)), key=_month_order)
    data = [{"month": mo, **{m: s[mo] for m, s in zip(metrics, by_month)}} for mo in months]
    title = "Monthly " + " and ".join(LABELS[m] for m in metrics) + title_suffix
    return {"type": chart_type, "title": title, "x_key": "month", "x_label": "Month",
            "series": _series(metrics), "data": data}


def rank_spec(rows: list, dimension: str, metric: str, chart_type: str = "bar", title: str = None) -> dict:
    dim_label = dimension.replace("_", " ").title()
    return {"type": chart_type, "title": title or f"{LABELS[metric]} by {dim_label}", "x_key": "name",
            "x_label": dim_label, "series": _series([metric]),
            "data": [{"name": str(r["name"]), metric: r[metric]} for r in rows]}


def campaign_spec(rows: list, metric: str, order: str) -> dict:
    word = "Bottom" if order == "asc" else "Top"
    return {"type": "bar", "title": f"{word} {len(rows)} Campaigns by {LABELS[metric]}", "x_key": "campaign",
            "x_label": "Campaign", "series": _series([metric]),
            "data": [{"campaign": str(r["id"]), metric: r[metric]} for r in rows]}


def from_direct_result(r: dict):
    """A chart for a DIRECT_DATABASE answer that is already a series or a ranking ("Show monthly
    revenue."), built from the same tool result: no extra query. None when it isn't chartable."""
    res, inp = r.get("result") or {}, r.get("tool_input") or {}
    try:
        if r.get("tool") == "trend_over_time" and len(res.get("series") or []) > 1 \
                and res.get("metric") in qr.TREND_METRICS:
            return validate_chart(trend_spec({res["metric"]: res}))
        if r.get("tool") == "rank_dimension" and 1 < len(res.get("results") or []) <= MAX_CATEGORIES \
                and res.get("metric") in LABELS:
            return validate_chart(rank_spec(res["results"], res["dimension"], res["metric"]))
        if r.get("tool") == "filter_campaigns" and not inp.get("conditions") \
                and inp.get("sort_by") in CAMPAIGN_METRICS and 1 < len(res.get("campaigns") or []) <= MAX_CATEGORIES:
            return validate_chart(campaign_spec(res["campaigns"], inp["sort_by"], inp.get("order", "desc")))
    except (ChartSpecError, KeyError, TypeError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# Explicit chart requests
# ---------------------------------------------------------------------------

def _requested_type(text: str):
    if PIE_RE.search(text):
        return "pie"
    if re.search(r"\b(?:bar|column)\s+(?:chart|graph)s?\b|\bbars\b", text, re.IGNORECASE):
        return "bar"
    if re.search(r"\bline\s+(?:chart|graph)s?\b|\blines\b", text, re.IGNORECASE):
        return "line"
    return None


def _label(metric: str) -> str:
    return LABELS[metric] if metric in ("roas", "roi_pct", "cpa", "cpc", "ctr") else LABELS[metric].lower()


def build_chart(question: str, filters: dict = None) -> ChartResult:
    """Plan and build one chart for an explicit chart request. Raises qr.QueryRouterError on data
    failures (the caller turns it into a controlled message); never calls an LLM."""
    filters = qr._validate(question, filters)
    if qr.UNAVAILABLE_RE.search(question.lower()):
        return ChartResult(route="UNSUPPORTED", message="That information is not available in the current "
                                                        "MarketingIQ dataset. " + qr.SUPPORTED_HINT)
    if UNSUPPORTED_TYPE_RE.search(question):
        return ChartResult(route="NEEDS_CLARIFICATION", message=TYPES_HINT + " Which of those would you like?")
    f = qr._parse(question, qr._get_entity_index())
    text, metrics, requested = f.text, qr._metric_names(f), _requested_type(question)
    dimension = f.dimensions[0] if f.dimensions else None
    scope = qr._entity_scope(f)
    top_n = qr.TOP_N_RE.search(text)
    notes = []

    years = [int(f.period[:4])] if f.period else []
    months = [int(f.period[5:7])] if f.period and len(f.period) == 7 else []

    def done(spec, caption, used_scope=None, **plan):
        spec = validate_chart(spec)
        # What was charted, as query_planner.Plan fields: the conversation memory's follow-ups start here.
        plan = {"entities": [], "filters": dict(used_scope or {}), "months": [], "years": [], "grain": None,
                "conditions": [], "order": "desc", "limit": None, "dimension": None, "visualization": spec["type"],
                **plan}
        return ChartResult(chart=spec, message=" ".join([caption, *notes]), filters_applied=filters,
                           scope=used_scope or {}, plan=plan)

    def ask(message, unmapped=False):
        return ChartResult(route="NEEDS_CLARIFICATION", message=message, unmapped=unmapped)

    if not metrics:
        return ask(WHICH_METRIC, unmapped=True)

    campaigns = qr.CAMPAIGN_RE.search(text) and (top_n or qr.SUPERLATIVE_RE.search(text)) and not dimension
    trend = not campaigns and (f.period or qr.TREND_RE.search(text) or (requested == "line" and not dimension))

    if trend:
        wrong = [m for m in metrics if m not in qr.TREND_METRICS]
        if wrong:
            return ask("Monthly figures are available for revenue, spend, profit and conversions, "
                       f"not {_label(wrong[0])}.", unmapped=True)
        if requested == "pie":
            return ask("A pie chart can't show change over time. Try \"Show monthly "
                       f"{_label(metrics[0])} as a line chart\".")
        metrics = metrics[:MAX_SERIES]
        if len({UNITS[m] for m in metrics}) > 1:
            return ask(f"{' and '.join(LABELS[m] for m in metrics)} use different units, so one chart would be "
                       "misleading. Ask for them as separate charts.")
        results = {m: qr._execute(qr.ToolCall("trend_over_time", {"metric": m}, period=f.period,
                                              extra_filters=scope or None), filters) for m in metrics}
        if not any(res["series"] for res in results.values()):
            return ask("There is no monthly data for that selection.")
        suffix = (f" ({f.period})" if f.period else "") + (" — " + ", ".join(scope.values()) if scope else "")
        spec = trend_spec(results, "bar" if requested == "bar" and len(metrics) == 1 else "line", suffix)
        first, last = spec["data"][0]["month"], spec["data"][-1]["month"]
        return done(spec, f"{spec['title']}, {first} to {last} ({len(spec['data'])} months).", scope,
                    intent="trend", metrics=metrics, grain="month", years=years, months=months)

    if campaigns:
        metric = next((m for m in metrics if m in CAMPAIGN_METRICS), None)
        if metric is None:
            return ask("Top-campaign charts support revenue, spend, profit, ROAS, CPA and conversion rate.")
        if requested == "pie":
            return ask("Individual campaigns aren't parts of one whole, so a pie chart would mislead. "
                       "Try a bar chart instead.")
        if requested == "line":
            notes.append("Shown as a bar chart: line charts are used for monthly trends.")
        n = min(int(top_n.group(1)), MAX_CATEGORIES) if top_n else 10
        if top_n and int(top_n.group(1)) > MAX_CATEGORIES:
            notes.append(f"Charts show at most {MAX_CATEGORIES} campaigns.")
        order = qr._rank_order(text, metric)
        res = qr._execute(qr.ToolCall("filter_campaigns", {"conditions": [], "sort_by": metric, "order": order,
                                                           "limit": max(n, 1)}, extra_filters=scope or None), filters)
        if not res["campaigns"]:
            return ask("No campaigns match the current filters.")
        spec = campaign_spec(res["campaigns"], metric, order)
        return done(spec, f"{spec['title']}.", scope, intent="campaign_list", metrics=[metric], order=order, limit=n)

    if dimension:
        metric = metrics[0]
        # Entities of the charted dimension pick bars; others scope the data ("... for TikTok").
        picked = [v for d, v in f.entities if d == dimension]
        scope = {k: v for k, v in scope.items() if k != qr.ENTITY_FILTER_KEYS.get(dimension)}
        pie = requested == "pie" or (requested is None and qr.SHARE_RE.search(text))
        if pie:
            if metric not in PIE_METRICS:
                return ask(f"{LABELS[metric]} values don't add up to a total, so a pie chart would mislead. "
                           f"Try \"Bar chart of {_label(metric)} by {dimension.replace('_', ' ')}\".")
            res = qr._execute(qr.ToolCall("rank_dimension", {"dimension": dimension, "metric": metric,
                                                             "order": "desc", "limit": 1000},
                                          extra_filters=scope or None), filters)
            rows = res["results"]
            if len(rows) > MAX_PIE_SLICES:
                return ask(f"There are {len(rows)} {dimension.replace('_', ' ')} values, too many for a readable "
                           "pie chart. Try a bar chart instead.")
            if any(r[metric] < 0 for r in rows):
                return ask(f"Some {dimension.replace('_', ' ')} values have negative {_label(metric)}, so they "
                           "can't be shown as shares of a total. Try a bar chart instead.")
            dim_label = dimension.replace("_", " ").title()
            spec = rank_spec(rows, dimension, metric, "pie", f"{LABELS[metric]} Share by {dim_label}")
            return done(spec, f"{spec['title']}: each slice is its share of the total.", scope,
                        intent="breakdown", metrics=[metric], dimension=dimension)
        if requested == "line":
            notes.append("Shown as a bar chart: line charts are used for monthly trends.")
        order = qr._rank_order(text, metric)
        limit = min(int(top_n.group(1)), MAX_CATEGORIES) if top_n else MAX_CATEGORIES
        res = qr._execute(qr.ToolCall("rank_dimension", {"dimension": dimension, "metric": metric, "order": order,
                                                         "limit": 1000}, extra_filters=scope or None), filters)
        rows = [r for r in res["results"] if not picked or r["name"] in picked]
        if not rows:
            return ask("No campaigns match the current filters.")
        if len(rows) > limit:
            if not top_n:
                notes.append(f"Showing the {'lowest' if order == 'asc' else 'top'} {limit} of {len(rows)}.")
            rows = rows[:limit]
        spec = rank_spec(rows, res["dimension"], res["metric"])
        plan = ({"intent": "compare_entities", "entities": picked} if len(picked) >= 2 else
                {"intent": "breakdown", "limit": limit if top_n else None})
        return done(spec, f"{spec['title']}, {'lowest' if order == 'asc' else 'highest'} first.", scope,
                    metrics=[res["metric"]], dimension=res["dimension"], order=order, **plan)

    # Two or more named values of one dimension, no breakdown word: "Google Ads vs TikTok spend chart".
    same_dim = [(d, v) for d, v in f.entities if f.entities and d == f.entities[0][0]]
    if len(same_dim) >= 2:
        dim, names, metric = same_dim[0][0], [v for _, v in same_dim], metrics[0]
        if requested == "pie":
            notes.append("Shown as a bar chart: a few named values aren't parts of one whole.")
        elif requested == "line":
            notes.append("Shown as a bar chart: line charts are used for monthly trends.")
        scope = {k: v for k, v in scope.items() if k != qr.ENTITY_FILTER_KEYS.get(dim)}
        res = qr._execute(qr.ToolCall("compare_entities", {"dimension": dim, "names": names},
                                      extra_filters=scope or None), filters)
        rows = sorted((r for r in res["results"] if r.get(metric) is not None), key=lambda r: r[metric], reverse=True)
        if not rows:
            return ask("No campaigns match the current filters.")
        spec = rank_spec(rows, dim, metric, "bar", f"{LABELS[metric]}: " + " vs ".join(r["name"] for r in rows))
        return done(spec, f"{spec['title']}.", scope, intent="compare_entities", metrics=[metric], dimension=dim,
                    entities=names)

    return ask(f"How should I break down {_label(metrics[0])}? For example: \"Line chart of monthly "
               f"{_label(metrics[0])}\" or \"Bar chart of {_label(metrics[0])} by platform\".", unmapped=True)


# ---------------------------------------------------------------------------
# Validation: the only gate between a spec and the frontend
# ---------------------------------------------------------------------------

def _text(value, what: str) -> str:
    if not isinstance(value, str) or not SAFE_TEXT_RE.match(value) or SCRIPT_RE.search(value):
        raise ChartSpecError(f"unsafe or missing {what}")
    return value


def validate_chart(spec) -> dict:
    """Deterministic contract check. Returns the spec unchanged, or raises ChartSpecError."""
    if not isinstance(spec, dict):
        raise ChartSpecError("spec must be an object")
    required = {"type", "title", "x_key", "x_label", "series", "data"}
    if not required <= set(spec) or set(spec) - required:
        raise ChartSpecError(f"fields must be exactly {sorted(required)}")
    chart_type = spec["type"]
    if chart_type not in CHART_TYPES:
        raise ChartSpecError(f"unknown chart type {chart_type!r}")
    _text(spec["title"], "title")
    _text(spec["x_label"], "x_label")
    x_key = spec["x_key"]
    if x_key not in X_KEYS:
        raise ChartSpecError(f"unknown x_key {x_key!r}")
    if chart_type == "line" and x_key != "month":
        raise ChartSpecError("line charts are for monthly series")

    series = spec["series"]
    if not isinstance(series, list) or not 1 <= len(series) <= MAX_SERIES:
        raise ChartSpecError("series must be a list of 1-3 entries")
    keys = []
    for s in series:
        if not isinstance(s, dict) or set(s) != {"key", "label", "unit"}:
            raise ChartSpecError("series entries need exactly key, label, unit")
        if s["key"] not in LABELS or s["label"] != LABELS[s["key"]] or s["unit"] != UNITS[s["key"]]:
            raise ChartSpecError(f"unknown metric or unit {s.get('key')!r}")
        keys.append(s["key"])
    if len(set(keys)) != len(keys):
        raise ChartSpecError("duplicate series")
    if len({s["unit"] for s in series}) > 1:
        raise ChartSpecError("series with different units")
    if chart_type == "pie" and (len(series) != 1 or keys[0] not in PIE_METRICS):
        raise ChartSpecError("a pie chart needs one additive metric")

    data = spec["data"]
    limit = MAX_POINTS if x_key == "month" else (MAX_PIE_SLICES if chart_type == "pie" else MAX_CATEGORIES)
    if not isinstance(data, list) or not 1 <= len(data) <= limit:
        raise ChartSpecError(f"data must have 1-{limit} rows")
    row_keys, previous = {x_key, *keys}, None
    for row in data:
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ChartSpecError("data rows must have exactly the x key and the series keys")
        x = _text(row[x_key], "category label")
        if x_key == "month":
            m = MONTH_KEY_RE.match(x)
            if not m:
                raise ChartSpecError(f"bad month {x!r}")
            current = (int(m.group(1)), int(m.group(2)))
            if previous and current <= previous:
                raise ChartSpecError("months must be in chronological order")
            previous = current
        for k in keys:
            v = row[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ChartSpecError(f"non-numeric or non-finite value for {k}")
            if chart_type == "pie" and v < 0:
                raise ChartSpecError("negative pie slice")
    if x_key != "month" and len({row[x_key] for row in data}) != len(data):
        raise ChartSpecError("duplicate categories")
    if chart_type == "pie" and not sum(row[keys[0]] for row in data) > 0:
        raise ChartSpecError("pie total must be positive")
    return spec
