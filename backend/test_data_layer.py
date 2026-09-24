"""
Correctness tests for the DuckDB data layer (campaign_repository.py + data_tools.py).

1. Golden comparison: every case in CASES is run through data_tools on the DuckDB
   backend and on the Pandas backend, and compared with data_layer_golden.json —
   outputs captured from the ORIGINAL pure-Pandas data_tools.py before the migration.
2. Independent checks: key figures recomputed directly from the raw DataFrame.
3. Safety / failure handling of the repository (identifier allowlist, bound values,
   unknown operators, query failures, empty results, DuckDB unavailable).

No LLM is involved. Run:  python test_data_layer.py      (or: pytest test_data_layer.py)
"""
import json
import math
from pathlib import Path
from unittest.mock import patch

import data_tools as dt
import campaign_repository as cr

HERE = Path(__file__).parent
CSV = str(HERE / ".." / "data" / "tech_advertising_campaigns_dataset.csv")
GOLDEN = HERE / "data_layer_golden.json"
# Engines sum/interpolate floats in a different order, so a value sitting exactly on a
# rounding tie can land either side of it: e.g. revenue p75 is exactly 22165.825 — numpy
# computes 22165.824999999997 (-> .82), DuckDB 22165.825 (-> .83). One cent is the most that
# can move a 2-decimal figure; everything else must match exactly.
FLOAT_ABS_TOL = 0.01 + 1e-9

MULTI = {"platform": "Facebook", "budget": "High", "retargeting": "Retargeting Only", "device": "Mobile"}
CHART = {"objective": "Conversions", "vertical": "SaaS", "gender": "Female", "age": "25-34",
         "creative": "Video", "emotion": "Trust", "placement": "Feed", "income": "$50K-$100K"}

# (case id, tool, kwargs). Covers every tool, every filter kind and the edge cases.
CASES = [
    ("totals", "get_totals", {}),
    ("totals_platform", "get_totals", {"filters": {"platform": "TikTok"}}),
    ("totals_budget", "get_totals", {"filters": {"budget": "High"}}),
    ("totals_multi", "get_totals", {"filters": MULTI}),
    ("totals_chart_filters", "get_totals", {"filters": {"gender": "Male", "device": "Desktop", "age": "18-24"}}),
    ("totals_cold", "get_totals", {"filters": {"retargeting": "Cold Audience Only", "vertical": "Gaming"}}),
    ("totals_all_sentinels", "get_totals", {"filters": {"platform": "All Platforms", "objective": "All Objectives",
                                                        "vertical": "All Industry Verticals",
                                                        "budget": "All Budget Tiers", "retargeting": "All"}}),
    ("totals_empty", "get_totals", {"filters": {"platform": "Nonexistent"}}),
    ("totals_bool_value", "get_totals", {"filters": {"gender": True}}),
    *[(f"rank_{d}", "rank_dimension", {"dimension": d, "metric": "roas", "limit": 20}) for d in dt.DIMENSION_COLUMNS],
    ("rank_platform_revenue_asc", "rank_dimension", {"dimension": "platform", "metric": "revenue", "order": "asc"}),
    ("rank_platform_top1", "rank_dimension", {"dimension": "platform", "metric": "roas", "limit": 1}),
    ("rank_device_count", "rank_dimension", {"dimension": "device", "metric": "campaign_count", "limit": 1}),
    ("rank_objective_cpa_filtered", "rank_dimension", {"dimension": "objective", "metric": "cpa", "order": "asc",
                                                       "filters": {"platform": "LinkedIn"}}),
    ("rank_unknown_metric", "rank_dimension", {"dimension": "platform", "metric": "nonsense"}),
    ("rank_unknown_dimension", "rank_dimension", {"dimension": "country"}),
    *[(f"trend_{m}", "trend_over_time", {"metric": m}) for m in ("revenue", "spend", "profit", "conversions")],
    ("trend_filtered", "trend_over_time", {"metric": "revenue", "filters": MULTI}),
    ("trend_unknown_metric", "trend_over_time", {"metric": "ctr"}),
    ("filter_roas_gt_8", "filter_campaigns", {"conditions": [{"field": "roas", "operator": ">", "value": 8}],
                                              "sort_by": "roas", "order": "desc", "limit": 10}),
    ("filter_spend_revenue", "filter_campaigns", {"conditions": [
        {"field": "spend", "operator": ">", "value": 4996.16}, {"field": "revenue", "operator": "<", "value": 665}],
        "sort_by": "profit", "order": "asc", "limit": 5}),
    ("filter_multi_with_filters", "filter_campaigns", {"conditions": [
        {"field": "conversion_rate", "operator": ">=", "value": 5}, {"field": "cpa", "operator": "<=", "value": 80}],
        "sort_by": "revenue", "order": "desc", "limit": 10, "filters": {"platform": "TikTok", "budget": "Medium"}}),
    ("filter_invalid_condition_skipped", "filter_campaigns", {"conditions": [
        {"field": "nonsense", "operator": ">", "value": 1}, {"field": "roas", "operator": "!=", "value": 1},
        {"field": "ctr", "operator": ">", "value": 7.5}], "sort_by": "unknown", "limit": 10}),
    ("filter_no_match", "filter_campaigns", {"conditions": [{"field": "spend", "operator": ">", "value": 10 ** 9}]}),
    ("compare_platforms", "compare_entities", {"dimension": "platform", "names": ["TikTok", "LinkedIn", "Meta Ads"]}),
    ("compare_retargeting", "compare_entities", {"dimension": "retargeting", "names": ["True", "False"]}),
    ("compare_filtered", "compare_entities", {"dimension": "device", "names": ["Mobile", "Tablet"], "filters": CHART}),
    ("entity_tiktok", "get_entity_metrics", {"dimension": "platform", "name": "TikTok"}),
    ("share_tiktok_spend", "percentage_share", {"dimension": "platform", "name": "TikTok", "metric": "spend"}),
    ("share_filtered", "percentage_share", {"dimension": "device", "name": "Mobile", "metric": "revenue",
                                            "filters": {"platform": "Instagram"}}),
    ("stats_all", "get_numeric_field_stats", {"fields": list(dt.NUMERIC_FIELD_COLUMNS) + ["nonsense"]}),
    ("stats_filtered", "get_numeric_field_stats", {"fields": ["spend", "roas"], "filters": MULTI}),
    ("fatigue", "get_creative_fatigue", {}),
    ("fatigue_filtered", "get_creative_fatigue", {"filters": {"platform": "Twitter", "creative": "Image"}}),
    ("fields", "list_available_fields", {}),
]


def run_case(tool, kwargs, module=dt):
    try:
        return module.TOOL_REGISTRY[tool](**kwargs)
    except Exception as e:  # the original raised on some empty-filter edge cases
        return {"__exception__": type(e).__name__}


def compare(a, b, path="", out=None):
    """Structural equality; numbers equal within FLOAT_ABS_TOL. Returns 'path: a != b' strings."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing on {'left' if k not in a else 'right'}")
            else:
                compare(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= FLOAT_ABS_TOL):
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def _load(backend):
    dt.load_data(CSV, backend=backend)
    assert dt.data_backend_name() == backend


def _check_against_golden(backend):
    _load(backend)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    failures = []
    for case_id, tool, kwargs in CASES:
        got = json.loads(json.dumps(run_case(tool, kwargs), default=str))
        diffs = compare(golden[case_id], got)
        if diffs:
            failures.append(f"{case_id}: {diffs[:3]}")
    assert not failures, f"{backend} differs from original Pandas output:\n" + "\n".join(failures)


def test_duckdb_matches_original_pandas():
    _check_against_golden("duckdb")


def test_pandas_backend_matches_original_pandas():
    _check_against_golden("pandas")


def test_independent_raw_dataframe_checks():
    _load("duckdb")
    df = dt.get_dataframe()
    t = dt.get_totals()
    assert t["campaign_count"] == len(df) == 10000
    assert t["revenue"] == round(float(df["revenue"].sum()), 2)
    assert t["spend"] == round(float(df["ad_spend"].sum()), 2)
    assert t["roas"] == round(float(df["revenue"].sum() / df["ad_spend"].sum()), 3)
    sub = df[(df["platform"] == "Facebook") & (df["budget_tier"] == "High") & df["retargeting_flag"]
             & (df["device_type"] == "Mobile")]
    assert dt.get_totals(MULTI)["campaign_count"] == len(sub)
    assert abs(dt.get_totals(MULTI)["revenue"] - sub["revenue"].sum()) < 0.01
    ranked = dt.rank_dimension("platform", "roas", limit=1)["results"][0]["name"]
    g = df.groupby("platform")[["revenue", "ad_spend"]].sum()
    assert ranked == (g["revenue"] / g["ad_spend"]).idxmax()
    jan = next(p for p in dt.trend_over_time("revenue")["series"] if p["month"] == "2025-01")
    assert abs(jan["value"] - df.loc[df["month"] == "2025-01", "revenue"].sum()) < 0.01
    hi = dt.filter_campaigns([{"field": "roas", "operator": ">", "value": 8}], sort_by="roas", order="desc")
    assert hi["matched_count"] == int((df["ROAS"] > 8).sum())
    assert [c["roas"] for c in hi["campaigns"]] == sorted((c["roas"] for c in hi["campaigns"]), reverse=True)


def test_rank_dimension_with_no_matching_campaigns():
    for backend in ("duckdb", "pandas"):
        _load(backend)
        r = dt.rank_dimension("platform", filters={"platform": "Nonexistent"})
        assert r == {"dimension": "platform", "metric": "roas", "results": []}
        stats = dt.get_numeric_field_stats(["spend"], filters={"platform": "Nonexistent"})
        assert stats["spend"]["median"] is None


def test_identifiers_are_allowlisted_and_values_bound():
    _load("duckdb")
    repo = dt.get_repository()
    for bad_column in ('platform" OR 1=1 --', "campaigns; DROP TABLE campaigns", "nonexistent"):
        try:
            repo.aggregate([(bad_column, "=", "x")])
            raise AssertionError("unknown column accepted")
        except cr.DataAccessError as e:
            assert "DROP" not in str(e) or "Unknown field" in str(e)
    try:
        repo.aggregate([("platform", "; DROP", "x")])
        raise AssertionError("unknown operator accepted")
    except cr.DataAccessError:
        pass
    # A hostile VALUE is just a string that matches nothing — it is bound, never executed.
    assert dt.get_totals({"platform": "x' OR '1'='1"})["campaign_count"] == 0
    assert dt.get_totals()["campaign_count"] == 10000  # table intact


def test_query_failure_is_wrapped_without_sql_details():
    _load("duckdb")
    try:
        dt.filter_campaigns([{"field": "spend", "operator": ">", "value": "not-a-number"}])
        raise AssertionError("expected DataAccessError")
    except cr.DataAccessError as e:
        assert str(e) == "The analytics query failed."
        assert "SELECT" not in str(e) and "campaigns" not in str(e)


def test_falls_back_to_pandas_when_duckdb_unavailable():
    with patch.object(cr.DuckDBCampaignRepository, "__init__", side_effect=ImportError("No module named 'duckdb'")):
        dt.load_data(CSV)
    assert dt.data_backend_name() == "pandas"
    assert dt.get_totals()["campaign_count"] == 10000
    _load("duckdb")


def test_router_and_endpoint_use_duckdb():
    import query_router as qr
    from fastapi.testclient import TestClient
    import main
    assert dt.data_backend_name() == "duckdb"
    r = qr.route_query("Which platform has the highest ROAS?")
    assert r["route"] == "DIRECT_DATABASE" and r["result"]["results"][0]["name"] == "TikTok"
    health = TestClient(main.app).get("/api/health").json()
    assert health["campaigns_loaded"] == 10000 and health["data_backend"] == "duckdb"


def test_concurrent_queries_from_threads():
    from concurrent.futures import ThreadPoolExecutor
    _load("duckdb")
    expected = dt.get_totals({"platform": "TikTok"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: dt.get_totals({"platform": "TikTok"}), range(64)))
    assert all(r == expected for r in results)


def test_duckdb_backend_keeps_no_duplicate_dataframe():
    """DuckDB holds its own copy of the rows, so no parsed DataFrame is kept alongside it."""
    _load("duckdb")
    assert dt._df is None and dt.row_count() == 10000
    reference = dt.get_dataframe()  # tests/tools only: a fresh parse, never cached
    assert len(reference) == 10000 and dt._df is None and dt.get_dataframe() is not reference
    _load("pandas")  # the Pandas backend keeps the DataFrame: it is the data
    assert dt._df is not None and dt.get_dataframe() is dt._df and dt.row_count() == 10000
    _load("duckdb")


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} DATA LAYER TESTS PASSED ===")
