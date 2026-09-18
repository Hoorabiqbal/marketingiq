"""
MarketingIQ AI Analyst — data/query layer.

Every function here operates on the REAL dataset (loaded once at startup)
and returns real computed numbers. The AI never sees raw rows unless a tool
explicitly returns them; it never invents a number itself. This is the
"backend executes against real data" step of the architecture.
"""
import pandas as pd
import numpy as np

CSV_PATH = None  # set by main.py at startup
_df = None

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

# Maps askable "metric" names onto how to compute them. Base sums are pulled
# straight from the data; derived metrics are computed from those sums using
# the real formulas (never a hardcoded value).
BASE_SUM_FIELDS = ["ad_spend", "revenue", "profit", "clicks", "impressions", "conversions"]


def load_data(csv_path: str):
    global _df, CSV_PATH
    CSV_PATH = csv_path
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["start_date"].dt.strftime("%Y-%m")
    _df = df
    return df


def get_dataframe() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Data not loaded — call load_data() at startup.")
    return _df


def list_available_fields() -> dict:
    """Lets the AI check what dimensions/metrics/values actually exist before answering,
    so it can say 'not available' instead of guessing."""
    df = get_dataframe()
    dims = {}
    for name, col in DIMENSION_COLUMNS.items():
        vals = df[col].dropna().unique().tolist()
        if len(vals) <= 20:
            dims[name] = sorted([str(v) for v in vals])
        else:
            dims[name] = f"{len(vals)} distinct values (too many to list)"
    return {
        "dimensions": dims,
        "metrics": ["spend", "revenue", "profit", "conversions", "clicks", "impressions",
                    "roas", "cpa", "cpc", "ctr", "conversion_rate", "quality_score", "bounce_rate"],
        "date_range": [str(df["start_date"].min().date()), str(df["start_date"].max().date())],
        "total_campaigns": len(df),
    }


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


def apply_global_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Mirrors the dashboard's own campaignMatches() logic exactly, so the AI is always
    reasoning over the SAME subset of data the user is currently looking at on screen —
    including chart-click cross-filters (gender/device/age/creative/emotion/placement/income),
    not just the five sidebar dropdown filters."""
    if not filters:
        return df
    if filters.get("platform") and filters["platform"] != "All Platforms":
        df = df[df["platform"] == filters["platform"]]
    if filters.get("objective") and filters["objective"] != "All Objectives":
        df = df[df["campaign_objective"] == filters["objective"]]
    if filters.get("vertical") and filters["vertical"] != "All Industry Verticals":
        df = df[df["industry_vertical"] == filters["vertical"]]
    if filters.get("budget") and filters["budget"] != "All Budget Tiers":
        df = df[df["budget_tier"] == filters["budget"]]
    retarget = filters.get("retargeting")
    if retarget == "Retargeting Only":
        df = df[df["retargeting_flag"] == True]  # noqa: E712
    elif retarget == "Cold Audience Only":
        df = df[df["retargeting_flag"] == False]  # noqa: E712
    for field, col in CHART_FILTER_COLUMNS.items():
        val = filters.get(field)
        if val:
            df = df[df[col] == val]
    return df


def _apply_extra_filter(df: pd.DataFrame, dimension: str, value: str) -> pd.DataFrame:
    if not dimension or not value:
        return df
    col = DIMENSION_COLUMNS.get(dimension)
    if not col:
        return df
    return df[df[col].astype(str) == str(value)]


def _metrics_from_rows(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"campaign_count": 0, "spend": 0, "revenue": 0, "profit": 0, "conversions": 0,
                "clicks": 0, "impressions": 0, "roas": None, "cpa": None, "cpc": None,
                "ctr": None, "conversion_rate": None, "roi_pct": None}
    spend = float(df["ad_spend"].sum())
    revenue = float(df["revenue"].sum())
    profit = float(df["profit"].sum())
    conversions = int(df["conversions"].sum())
    clicks = int(df["clicks"].sum())
    impressions = int(df["impressions"].sum())
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


def get_totals(filters: dict = None) -> dict:
    df = apply_global_filters(get_dataframe(), filters or {})
    return _metrics_from_rows(df)


def rank_dimension(dimension: str, metric: str = "roas", order: str = "desc",
                    limit: int = 10, filters: dict = None) -> dict:
    if dimension not in DIMENSION_COLUMNS:
        return {"error": f"Unknown dimension '{dimension}'. Valid dimensions: {list(DIMENSION_COLUMNS.keys())}"}
    col = DIMENSION_COLUMNS[dimension]
    df = apply_global_filters(get_dataframe(), filters or {})
    rows = []
    for name, g in df.groupby(col):
        m = _metrics_from_rows(g)
        m["name"] = str(name)
        rows.append(m)
    valid_metric = metric if metric in rows[0] else "roas" if rows else metric
    rows = [r for r in rows if r.get(valid_metric) is not None]
    rows.sort(key=lambda r: r[valid_metric], reverse=(order == "desc"))
    return {"dimension": dimension, "metric": valid_metric, "results": rows[:limit]}


def compare_entities(dimension: str, names: list, filters: dict = None) -> dict:
    if dimension not in DIMENSION_COLUMNS:
        return {"error": f"Unknown dimension '{dimension}'. Valid dimensions: {list(DIMENSION_COLUMNS.keys())}"}
    col = DIMENSION_COLUMNS[dimension]
    df = apply_global_filters(get_dataframe(), filters or {})
    available = set(df[col].astype(str).unique())
    results, not_found = [], []
    for name in names:
        if str(name) not in available:
            not_found.append(name)
            continue
        sub = df[df[col].astype(str) == str(name)]
        m = _metrics_from_rows(sub)
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
    df = apply_global_filters(get_dataframe(), filters or {})
    field_map = {"spend": "ad_spend", "revenue": "revenue", "profit": "profit",
                 "roas": "ROAS", "cpa": "CPA", "ctr": "CTR", "conversion_rate": "conversion_rate",
                 "clicks": "clicks", "conversions": "conversions"}
    out = {}
    for f in fields:
        col = field_map.get(f)
        if col is None or col not in df.columns:
            out[f] = {"error": "field not available"}
            continue
        s = df[col].dropna()
        out[f] = {
            "min": round(float(s.min()), 2), "max": round(float(s.max()), 2),
            "mean": round(float(s.mean()), 2), "median": round(float(s.median()), 2),
            "p25": round(float(s.quantile(0.25)), 2), "p75": round(float(s.quantile(0.75)), 2),
        }
    return out


def filter_campaigns(conditions: list, sort_by: str = "profit", order: str = "asc",
                      limit: int = 10, filters: dict = None) -> dict:
    """conditions: list of {field, operator, value} e.g. [{"field":"spend","operator":">","value":40000},
    {"field":"revenue","operator":"<","value":20000}]. Lets the AI find campaigns matching
    real numeric criteria (e.g. 'high spend, low revenue') using thresholds it derived from
    get_numeric_field_stats, rather than us hardcoding what 'high' or 'low' means."""
    df = apply_global_filters(get_dataframe(), filters or {})
    field_map = {"spend": "ad_spend", "revenue": "revenue", "profit": "profit",
                 "roas": "ROAS", "cpa": "CPA", "ctr": "CTR", "conversion_rate": "conversion_rate",
                 "clicks": "clicks", "conversions": "conversions"}
    ops = {">": lambda s, v: s > v, "<": lambda s, v: s < v,
           ">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v, "==": lambda s, v: s == v}
    for cond in conditions or []:
        col = field_map.get(cond.get("field"))
        op = ops.get(cond.get("operator"))
        if col is None or op is None or col not in df.columns:
            continue
        df = df[op(df[col], cond["value"])]
    sort_col = field_map.get(sort_by, "profit")
    df = df.sort_values(sort_col, ascending=(order == "asc"))
    cols = ["campaign_id", "platform", "campaign_objective", "ad_spend", "revenue",
            "profit", "ROAS", "CPA", "conversion_rate"]
    out = df[cols].head(limit).rename(columns={
        "campaign_id": "id", "campaign_objective": "objective", "ad_spend": "spend",
        "ROAS": "roas", "CPA": "cpa", "conversion_rate": "conversion_rate",
    })
    return {"matched_count": len(df), "campaigns": out.round(2).to_dict(orient="records")}


def trend_over_time(metric: str = "revenue", filters: dict = None) -> dict:
    df = apply_global_filters(get_dataframe(), filters or {})
    field_map = {"revenue": "revenue", "spend": "ad_spend", "profit": "profit", "conversions": "conversions"}
    col = field_map.get(metric, "revenue")
    g = df.groupby("month")[col].sum().reset_index()
    return {"metric": metric, "series": [{"month": r["month"], "value": round(float(r[col]), 2)} for _, r in g.iterrows()]}


def get_creative_fatigue(filters: dict = None) -> dict:
    df = apply_global_filters(get_dataframe(), filters or {})
    df = df.copy()
    df["age_bucket"] = pd.cut(df["creative_age_days"], bins=[0, 15, 30, 45, 60, 90],
                               labels=["0-15", "16-30", "31-45", "46-60", "61-90"])
    g = df.groupby("age_bucket", observed=True).agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
    g["ctr"] = (g["clicks"] / g["impressions"] * 100).round(2)
    return {"buckets": [{"age_range": str(idx), "ctr": float(row["ctr"])} for idx, row in g.iterrows()]}


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
