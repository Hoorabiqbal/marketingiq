"""
MarketingIQ AI Analyst — data/query layer.

Every function here operates on the REAL dataset (loaded once at startup)
and returns real computed numbers. The AI never sees raw rows unless a tool
explicitly returns them; it never invents a number itself. This is the
"backend executes against real data" step of the architecture.

This module owns the business logic — metric formulas, what each dashboard
filter means, ranking and rounding. Row filtering and aggregation are delegated
to campaign_repository.py (DuckDB by default, Pandas as fallback/reference), so
no function here depends on a particular database engine.
"""
import os
import re

import numpy as np
import pandas as pd

from campaign_repository import PandasCampaignRepository, create_repository

CSV_PATH = None  # set by main.py at startup
_df = None  # only for the Pandas backend (see load_data)
_row_count = 0
_repo = None

# Maps the "dimension" names the AI is allowed to ask for onto real dataset columns.
# Adding a new askable dimension = one new line here. No per-question code.
DIMENSION_COLUMNS = {
    "platform": "platform",
    "objective": "campaign_objective",
    "device": "device_type",
    "age_group": "target_audience_age",
    "gender": "target_audience_gender",
    "vertical": "industry_vertical",
    "budget_tier": "budget_tier",
    "creative_format": "creative_format",
    "creative_emotion": "creative_emotion",
    "placement": "ad_placement",
    "income_bracket": "income_bracket",
    "audience_interest": "audience_interest_category",
    "operating_system": "operating_system",
    "retargeting": "retargeting_flag",
}

# Further categorical columns, for breakdowns in the analytics planner (schema_catalog.py). They are
# kept out of DIMENSION_COLUMNS so the Query Router's entity index ("long", "monday", ...) is unchanged.
EXTRA_DIMENSION_COLUMNS = {
    "day_of_week": "day_of_week",
    "quarter": "quarter",
    "creative_size": "creative_size",
    "ad_copy_length": "ad_copy_length",
    "call_to_action": "has_call_to_action",
    "purchase_intent": "purchase_intent_score",
}
# Per-campaign measures reported as the mean over campaigns (not summable).
AVG_FIELD_COLUMNS = {"bounce_rate": "bounce_rate", "avg_session_duration": "avg_session_duration_seconds",
                     "pages_per_session": "pages_per_session", "quality_score": "quality_score"}
AVG_DECIMALS = {"bounce_rate": 2, "avg_session_duration": 1, "pages_per_session": 2, "quality_score": 2}
# Per-campaign numeric columns for relationships (scatter), by metric name.
CAMPAIGN_FIELD_COLUMNS = {**{"spend": "ad_spend", "revenue": "revenue", "profit": "profit", "roas": "ROAS",
                             "cpa": "CPA", "cpc": "CPC", "ctr": "CTR", "conversion_rate": "conversion_rate",
                             "clicks": "clicks", "conversions": "conversions", "impressions": "impressions"},
                          **AVG_FIELD_COLUMNS}


def all_dimension_columns() -> dict:
    return {**DIMENSION_COLUMNS, **EXTRA_DIMENSION_COLUMNS}


# Maps askable "metric" names onto how to compute them. Base sums are pulled
# straight from the data; derived metrics are computed from those sums using
# the real formulas (never a hardcoded value).
BASE_SUM_FIELDS = ["ad_spend", "revenue", "profit", "clicks", "impressions", "conversions"]

# Askable per-campaign numeric fields -> dataset columns (get_numeric_field_stats, filter_campaigns).
NUMERIC_FIELD_COLUMNS = {"spend": "ad_spend", "revenue": "revenue", "profit": "profit",
                         "roas": "ROAS", "cpa": "CPA", "ctr": "CTR", "conversion_rate": "conversion_rate",
                         "clicks": "clicks", "conversions": "conversions"}
CONDITION_OPERATORS = {">", "<", ">=", "<=", "=="}
MONTH_KEY_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")

# Creative-age buckets for get_creative_fatigue: right-closed intervals (0,15], (15,30], ...
CREATIVE_AGE_EDGES = [0, 15, 30, 45, 60, 90]
CREATIVE_AGE_LABELS = ["0-15", "16-30", "31-45", "46-60", "61-90"]


def _read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["start_date"].dt.strftime("%Y-%m")
    return df


def load_data(csv_path: str, backend: str = None):
    """Parse the CSV once, then build the analytical backend from that same parsed data
    (so DuckDB and Pandas see identical values). backend: 'duckdb' (default) or 'pandas';
    also settable via MARKETINGIQ_DATA_BACKEND.

    DuckDB copies the rows into its own table, so the parsed DataFrame is kept only when the
    repository is Pandas-backed (it then IS the data). No request path reads it."""
    global _df, _repo, _row_count, CSV_PATH
    CSV_PATH = csv_path
    df = _read_csv(csv_path)
    _repo = create_repository(df, backend or os.getenv("MARKETINGIQ_DATA_BACKEND", "duckdb"))
    _row_count = len(df)
    _df = df if _repo.name == "pandas" else None


def row_count() -> int:
    get_repository()
    return _row_count


def get_dataframe() -> pd.DataFrame:
    """The dataset as a DataFrame, for tests and offline tools. With the DuckDB backend this
    parses a fresh copy from the CSV on every call, so never use it on a request path."""
    get_repository()
    return _df if _df is not None else _read_csv(CSV_PATH)


def get_repository():
    if _repo is None:
        raise RuntimeError("Data not loaded — call load_data() at startup.")
    return _repo


def data_backend_name() -> str:
    return _repo.name if _repo is not None else "not_loaded"


def list_available_fields() -> dict:
    """Lets the AI check what dimensions/metrics/values actually exist before answering,
    so it can say 'not available' instead of guessing."""
    profile = get_repository().profile(list(DIMENSION_COLUMNS.values()), "start_date")
    dims = {}
    for name, col in DIMENSION_COLUMNS.items():
        vals = profile["distinct"][col]
        if len(vals) <= 20:
            dims[name] = sorted([str(v) for v in vals])
        else:
            dims[name] = f"{len(vals)} distinct values (too many to list)"
    return {
        "dimensions": dims,
        "metrics": ["spend", "revenue", "profit", "conversions", "clicks", "impressions",
                    "roas", "cpa", "cpc", "ctr", "conversion_rate", "quality_score", "bounce_rate"],
        "date_range": [_date_str(profile["date_min"]), _date_str(profile["date_max"])],
        "total_campaigns": profile["row_count"],
    }


def _date_str(value) -> str:
    return str(value.date()) if hasattr(value, "date") else str(value)


# Frontend chart-click filters use these field names (see filterState in index.html);
# they don't all match DIMENSION_COLUMNS' naming, so map them explicitly here.
CHART_FILTER_COLUMNS = {
    "gender": "target_audience_gender",
    "device": "device_type",
    "age": "target_audience_age",
    "creative": "creative_format",
    "emotion": "creative_emotion",
    "placement": "ad_placement",
    "income": "income_bracket",
}

# Sidebar dropdown filters: (filter key, column, the "no filter" option label).
SIDEBAR_FILTERS = [
    ("platform", "platform", "All Platforms"),
    ("objective", "campaign_objective", "All Objectives"),
    ("vertical", "industry_vertical", "All Industry Verticals"),
    ("budget", "budget_tier", "All Budget Tiers"),
]


def _filter_conditions(filters: dict) -> list:
    """Mirrors the dashboard's own campaignMatches() logic exactly, so the AI is always
    reasoning over the SAME subset of data the user is currently looking at on screen —
    including chart-click cross-filters (gender/device/age/creative/emotion/placement/income),
    not just the five sidebar dropdown filters. Returns engine-neutral (column, op, value)
    conditions for the data layer."""
    if not filters:
        return []
    conditions = []
    for key, col, all_label in SIDEBAR_FILTERS:
        val = filters.get(key)
        if val and val != all_label:
            conditions.append((col, "=", _as_text(val)))
    retarget = filters.get("retargeting")
    if retarget == "Retargeting Only":
        conditions.append(("retargeting_flag", "=", True))
    elif retarget == "Cold Audience Only":
        conditions.append(("retargeting_flag", "=", False))
    for field, col in CHART_FILTER_COLUMNS.items():
        val = filters.get(field)
        if val:
            conditions.append((col, "=", _as_text(val)))
    # Inclusive calendar-month range ("2025-01".."2025-12"), used by the analytics planner.
    for key, op in (("month_from", ">="), ("month_to", "<=")):
        val = filters.get(key)
        if isinstance(val, str) and MONTH_KEY_RE.match(val):
            conditions.append(("month", op, val))
    return conditions


def _as_text(value) -> str:
    # Filter columns hold text; a non-text value can never equal one (same as the Pandas behaviour).
    return value if isinstance(value, str) else str(value)


def apply_global_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Pandas form of the dashboard filters (same meaning as _filter_conditions)."""
    return PandasCampaignRepository(df)._filtered(_filter_conditions(filters))


def _apply_extra_filter(df: pd.DataFrame, dimension: str, value: str) -> pd.DataFrame:
    if not dimension or not value:
        return df
    col = DIMENSION_COLUMNS.get(dimension)
    if not col:
        return df
    return df[df[col].astype(str) == str(value)]


def _metrics_from_sums(s: dict) -> dict:
    """s: campaign_count plus the summed BASE_SUM_FIELDS (as returned by the data layer)."""
    n = int(s["campaign_count"])
    if n == 0:
        return {"campaign_count": 0, "spend": 0, "revenue": 0, "profit": 0, "conversions": 0,
                "clicks": 0, "impressions": 0, "roas": None, "cpa": None, "cpc": None,
                "ctr": None, "conversion_rate": None, "roi_pct": None}
    spend = float(s["ad_spend"])
    revenue = float(s["revenue"])
    profit = float(s["profit"])
    conversions = int(s["conversions"])
    clicks = int(s["clicks"])
    impressions = int(s["impressions"])
    return {
        "campaign_count": n,
        "spend": round(spend, 2),
        "revenue": round(revenue, 2),
        "profit": round(profit, 2),
        "conversions": conversions,
        "clicks": clicks,
        "impressions": impressions,
        "roas": round(revenue / spend, 3) if spend else None,
        "cpa": round(spend / conversions, 2) if conversions else None,
        "cpc": round(spend / clicks, 3) if clicks else None,
        "ctr": round(clicks / impressions * 100, 3) if impressions else None,
        "conversion_rate": round(conversions / clicks * 100, 3) if clicks else None,
        "roi_pct": round((revenue - spend) / spend * 100, 1) if spend else None,
    }


def metric_table(filters: dict = None, group_by: str = None) -> list:
    """Every metric, overall (one row) or per group: group_by is a dimension name or "month".
    Sum-based metrics come from the same formulas as get_totals (_metrics_from_sums); per-campaign
    measures (AVG_FIELD_COLUMNS) are means. Rows: {"group": str or None, **metrics}, in group order."""
    col = "month" if group_by == "month" else all_dimension_columns().get(group_by) if group_by else None
    if group_by and col is None:
        return []
    repo = get_repository()
    where = _filter_conditions(filters or {})
    sums = repo.aggregate(where, group_by=col)
    avgs = {str(r.get("group")): r for r in repo.aggregate_avg(list(AVG_FIELD_COLUMNS.values()), where, group_by=col)}
    rows = []
    for s in sums:
        if not s["campaign_count"]:
            continue
        key = str(s["group"]) if col else None
        a = avgs.get(key if col else "None", {})
        m = _metrics_from_sums(s)
        for name, c in AVG_FIELD_COLUMNS.items():
            v = a.get(c)
            m[name] = round(float(v), AVG_DECIMALS[name]) if v is not None else None
        rows.append({"group": key, **m})
    return rows


def campaign_points(x_metric: str, y_metric: str, filters: dict = None) -> dict:
    """Per-campaign (x, y) values for a relationship: every matching campaign, ordered by id, plus the
    Pearson correlation over all of them (computed here from the data-layer rows)."""
    xc, yc = CAMPAIGN_FIELD_COLUMNS[x_metric], CAMPAIGN_FIELD_COLUMNS[y_metric]
    cols = list(dict.fromkeys(["campaign_id", "platform", xc, yc]))
    n, rows = get_repository().find_campaigns(_filter_conditions(filters or {}), "campaign_id", True,
                                              max(_row_count, 1), cols)
    points = [{"id": str(r["campaign_id"]), "platform": str(r["platform"]), "x": _round2(r[xc]), "y": _round2(r[yc])}
              for r in rows if r[xc] is not None and r[yc] is not None]
    r_value = None
    if len(points) >= 3:
        xs = np.array([float(p["x"]) for p in points]); ys = np.array([float(p["y"]) for p in points])
        if xs.std() > 0 and ys.std() > 0:
            r_value = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
    return {"x": x_metric, "y": y_metric, "count": len(points), "correlation": r_value, "points": points}


def _metrics_from_rows(df: pd.DataFrame) -> dict:
    return _metrics_from_sums({"campaign_count": len(df), **{c: df[c].sum() for c in BASE_SUM_FIELDS}})


def get_totals(filters: dict = None) -> dict:
    return _metrics_from_sums(get_repository().aggregate(_filter_conditions(filters or {}))[0])


def rank_dimension(dimension: str, metric: str = "roas", order: str = "desc",
                    limit: int = 10, filters: dict = None) -> dict:
    if dimension not in DIMENSION_COLUMNS:
        return {"error": f"Unknown dimension '{dimension}'. Valid dimensions: {list(DIMENSION_COLUMNS.keys())}"}
    col = DIMENSION_COLUMNS[dimension]
    rows = []
    for g in get_repository().aggregate(_filter_conditions(filters or {}), group_by=col):
        m = _metrics_from_sums(g)
        m["name"] = str(g["group"])
        rows.append(m)
    if not rows:  # no campaigns match the filters
        return {"dimension": dimension, "metric": metric, "results": []}
    valid_metric = metric if metric in rows[0] else "roas"
    rows = [r for r in rows if r.get(valid_metric) is not None]
    rows.sort(key=lambda r: r[valid_metric], reverse=(order == "desc"))
    return {"dimension": dimension, "metric": valid_metric, "results": rows[:limit]}


def compare_entities(dimension: str, names: list, filters: dict = None) -> dict:
    if dimension not in DIMENSION_COLUMNS:
        return {"error": f"Unknown dimension '{dimension}'. Valid dimensions: {list(DIMENSION_COLUMNS.keys())}"}
    col = DIMENSION_COLUMNS[dimension]
    groups = {str(g["group"]): g for g in get_repository().aggregate(_filter_conditions(filters or {}), group_by=col)}
    available = set(groups)
    results, not_found = [], []
    for name in names:
        if str(name) not in available:
            not_found.append(name)
            continue
        m = _metrics_from_sums(groups[str(name)])
        m["name"] = name
        results.append(m)
    out = {"dimension": dimension, "results": results}
    if not_found:
        out["not_found"] = not_found
        out["available_values"] = sorted(list(available))[:20]
    return out


def get_entity_metrics(dimension: str, name: str, filters: dict = None) -> dict:
    return compare_entities(dimension, [name], filters)


def percentage_share(dimension: str, name: str, metric: str = "spend", filters: dict = None) -> dict:
    totals = get_totals(filters)
    entity = get_entity_metrics(dimension, name, filters)
    if entity.get("not_found"):
        return entity
    entity_val = entity["results"][0].get(metric)
    total_val = totals.get(metric)
    if entity_val is None or not total_val:
        return {"error": f"Metric '{metric}' not available for this comparison."}
    return {
        "dimension": dimension, "name": name, "metric": metric,
        "entity_value": entity_val, "total_value": total_val,
        "share_pct": round(entity_val / total_val * 100, 2),
    }


def get_numeric_field_stats(fields: list, filters: dict = None) -> dict:
    """Lets the AI discover real percentiles/min/max/median before deciding what counts
    as 'high' or 'low' — so thresholds come from the data, not a guess."""
    repo = get_repository()
    columns = list(dict.fromkeys(NUMERIC_FIELD_COLUMNS[f] for f in fields
                                 if f in NUMERIC_FIELD_COLUMNS and repo.has_column(NUMERIC_FIELD_COLUMNS[f])))
    stats = repo.numeric_stats(columns, _filter_conditions(filters or {})) if columns else {}
    out = {}
    for f in fields:
        col = NUMERIC_FIELD_COLUMNS.get(f)
        if col is None or col not in stats:
            out[f] = {"error": "field not available"}
            continue
        s = stats[col]
        if s is None:  # no campaigns match the filters
            out[f] = {k: None for k in ("min", "max", "mean", "median", "p25", "p75")}
            continue
        out[f] = {k: round(float(s[k]), 2) for k in ("min", "max", "mean", "median", "p25", "p75")}
    return out


def filter_campaigns(conditions: list, sort_by: str = "profit", order: str = "asc",
                      limit: int = 10, filters: dict = None, include_aggregates: bool = False) -> dict:
    """conditions: list of {field, operator, value} e.g. [{"field":"spend","operator":">","value":40000},
    {"field":"revenue","operator":"<","value":20000}]. Lets the AI find campaigns matching
    real numeric criteria (e.g. 'high spend, low revenue') using thresholds it derived from
    get_numeric_field_stats, rather than us hardcoding what 'high' or 'low' means."""
    repo = get_repository()
    where = _filter_conditions(filters or {})
    for cond in conditions or []:
        col = NUMERIC_FIELD_COLUMNS.get(cond.get("field"))
        op = cond.get("operator")
        if col is None or op not in CONDITION_OPERATORS or not repo.has_column(col):
            continue
        where.append((col, op, cond["value"]))
    sort_col = NUMERIC_FIELD_COLUMNS.get(sort_by, "profit")
    cols = ["campaign_id", "platform", "campaign_objective", "ad_spend", "revenue",
            "profit", "ROAS", "CPA", "conversion_rate"]
    matched, rows = repo.find_campaigns(where, sort_col, ascending=(order == "asc"), limit=limit, columns=cols)
    rename = {"campaign_id": "id", "campaign_objective": "objective", "ad_spend": "spend",
              "ROAS": "roas", "CPA": "cpa", "conversion_rate": "conversion_rate"}
    campaigns = [{rename.get(k, k): _round2(v) for k, v in r.items()} for r in rows]
    out = {"matched_count": int(matched), "campaigns": campaigns}
    if include_aggregates:
        # Totals for the whole matching group (same formulas as get_totals), so an explanation
        # never has to present the few example rows above as if they described the group.
        out["aggregates"] = _metrics_from_sums(repo.aggregate(where)[0])
    return out


def _round2(value):
    # Same rounding as DataFrame.round(2) (numpy half-to-even on the float).
    return float(np.round(value, 2)) if isinstance(value, (float, np.floating)) else value


def trend_over_time(metric: str = "revenue", filters: dict = None) -> dict:
    field_map = {"revenue": "revenue", "spend": "ad_spend", "profit": "profit", "conversions": "conversions"}
    col = field_map.get(metric, "revenue")
    series = get_repository().monthly_sum(col, _filter_conditions(filters or {}))
    return {"metric": metric, "series": [{"month": m, "value": round(float(v), 2)} for m, v in series]}


def get_creative_fatigue(filters: dict = None) -> dict:
    buckets = get_repository().bucket_totals(_filter_conditions(filters or {}), "creative_age_days",
                                             CREATIVE_AGE_EDGES, ["clicks", "impressions"])
    return {"buckets": [
        {"age_range": CREATIVE_AGE_LABELS[i],
         "ctr": float(np.round(s["clicks"] / s["impressions"] * 100, 2)) if s["impressions"] else None}
        for i, s in buckets
    ]}


TOOL_REGISTRY = {
    "get_totals": get_totals,
    "rank_dimension": rank_dimension,
    "compare_entities": compare_entities,
    "get_entity_metrics": get_entity_metrics,
    "percentage_share": percentage_share,
    "get_numeric_field_stats": get_numeric_field_stats,
    "filter_campaigns": filter_campaigns,
    "trend_over_time": trend_over_time,
    "get_creative_fatigue": get_creative_fatigue,
    "list_available_fields": list_available_fields,
}
