"""
Chart specs for the AI Analyst chat: validated plan results (DuckDB) -> validated chart spec.

    query_planner / analytics_engine results -> rank_spec / trend_spec / campaign_spec / ...
        -> validate_chart() -> the chat response

No LLM is ever involved: the model only asks for a chart type in its plan; every number comes from
the data tools. The spec never carries code or HTML; the frontend draws it from data
(site/dashboard.html, renderChatChart).
"""
import math
import re

import grounding  # noqa: F401
import schema_catalog as catalog

CHART_TYPES = {"line", "bar", "pie", "stacked_bar", "stacked_percent", "scatter"}
MAX_CHART_SERIES = 12   # grouped / stacked / multi-line series in one chart
MAX_SCATTER_POINTS = 1000
GENERIC_KEY_RE = re.compile(r"^s\d{1,2}$")  # series that are entities (one line per platform, ...)
STACKED = ("stacked_bar", "stacked_percent")  # parts of a whole
MAX_CATEGORIES = 20   # bars / top-N items in one chat chart
MAX_PIE_SLICES = 10
MAX_POINTS = 120      # monthly points (the dataset spans 25 months)
X_KEYS = {"month": "Month", "campaign": "Campaign", "name": None}  # None: the dimension's own label
UNITS = {m.name: m.unit for m in catalog.METRICS.values()}
LABELS = {m.name: m.label for m in catalog.METRICS.values()}
# Additive, never-negative-by-definition totals: the only metrics a pie can split into parts.
PIE_METRICS = {"spend", "revenue", "conversions", "clicks", "impressions", "campaign_count", "profit"}
CAMPAIGN_METRICS = {"spend", "revenue", "profit", "roas", "cpa", "conversion_rate"}  # filter_campaigns columns

_c = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
VIZ_RE = _c(r"\b(?:charts?|chart(?:ed|ing)|plot(?:s|ted|ting)?|graphs?|graph(?:ed|ing)|visuali[sz]\w*|diagrams?"
            r"|line (?:chart|graph)|bar (?:chart|graph)|column chart|pie(?: chart)?)\b")
SHOW_RE = _c(r"^\s*(?:please\s+)?(?:show|display|draw)\b")
SAFE_TEXT_RE = re.compile(r"^[^<>`\x00-\x1f\\]{1,160}$")
MONTH_KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
SCRIPT_RE = _c(r"javascript:|data:|on\w+\s*=|<script")



class ChartSpecError(ValueError):
    """The chart spec broke the contract; it is never sent to the frontend."""


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


def _label(metric: str) -> str:
    return LABELS[metric] if metric in ("roas", "roi_pct", "cpa", "cpc", "ctr") else LABELS[metric].lower()


# ---------------------------------------------------------------------------
# Validation: the only gate between a spec and the frontend
# ---------------------------------------------------------------------------

def _text(value, what: str) -> str:
    if not isinstance(value, str) or not SAFE_TEXT_RE.match(value) or SCRIPT_RE.search(value):
        raise ChartSpecError(f"unsafe or missing {what}")
    return value


def _series_metric(s: dict) -> str:
    """Metric behind a series entry. Legacy form: {key: metric, label, unit}. Generic form (series
    that are entities, e.g. one line per platform): {key: "s0".."s11" or the metric, label, unit, metric}."""
    fields = set(s) if isinstance(s, dict) else set()
    if fields == {"key", "label", "unit"}:
        if s["key"] not in LABELS or s["label"] != LABELS[s["key"]] or s["unit"] != UNITS[s["key"]]:
            raise ChartSpecError(f"unknown metric or unit {s.get('key')!r}")
        return s["key"]
    if fields == {"key", "label", "unit", "metric"}:
        if s["metric"] not in LABELS or s["unit"] != UNITS[s["metric"]]:
            raise ChartSpecError(f"unknown metric or unit {s.get('metric')!r}")
        if not (GENERIC_KEY_RE.match(str(s["key"])) or s["key"] == s["metric"]):
            raise ChartSpecError(f"bad series key {s.get('key')!r}")
        _text(s["label"], "series label")
        return s["metric"]
    raise ChartSpecError("series entries need exactly key, label, unit (generic series: plus metric)")


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
    if chart_type == "scatter" and x_key == "month":
        raise ChartSpecError("a scatter plot relates two measures, not months")

    series = spec["series"]
    if not isinstance(series, list) or not 1 <= len(series) <= MAX_CHART_SERIES:
        raise ChartSpecError(f"series must be a list of 1-{MAX_CHART_SERIES} entries")
    metrics = [_series_metric(s) for s in series]
    keys = [s["key"] for s in series]
    if len(set(keys)) != len(keys):
        raise ChartSpecError("duplicate series")
    if chart_type == "scatter":
        if len(series) != 2:
            raise ChartSpecError("a scatter plot needs exactly two measures (x, y)")
    elif len({s["unit"] for s in series}) > 1:
        raise ChartSpecError("series with different units")
    if chart_type == "pie" and (len(series) != 1 or metrics[0] not in PIE_METRICS):
        raise ChartSpecError("a pie chart needs one additive metric")
    if chart_type in STACKED and not all(m in PIE_METRICS for m in metrics):
        raise ChartSpecError("stacked bars need additive measures")

    data = spec["data"]
    limit = (MAX_SCATTER_POINTS if chart_type == "scatter" else MAX_POINTS if x_key == "month"
             else MAX_PIE_SLICES if chart_type == "pie" else MAX_CATEGORIES)
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
            if chart_type in ("pie", *STACKED) and v < 0:
                raise ChartSpecError("negative value in a part-to-whole chart")
        if chart_type == "stacked_percent" and not sum(row[k] for k in keys) > 0:
            raise ChartSpecError("each 100% stacked bar needs a positive total")
    if x_key != "month" and len({row[x_key] for row in data}) != len(data):
        raise ChartSpecError("duplicate categories")
    if chart_type == "pie" and not sum(row[keys[0]] for row in data) > 0:
        raise ChartSpecError("pie total must be positive")
    return spec
