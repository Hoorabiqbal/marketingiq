"""
Generic execution of validated analytics plans that the original planner handlers don't cover:
several metrics or series, a second breakdown (platform x device), several entities over time,
relationships between two measures, per-campaign averages and the extra catalog dimensions.

    validated Plan -> data_tools.metric_table / campaign_points (DuckDB) -> result table
                   -> charts chosen from the table's shape and units -> chart_builder.validate_chart

Every number comes from the data layer; directions, rankings and correlations are computed here.
Charts never put different units on one axis: a mixed request becomes one chart per unit.
"""
import math
from dataclasses import dataclass

import chart_builder as cb
import data_tools as dt
import query_planner as qp
import query_router as qr
import schema_catalog as catalog

MAX_CHARTS = 3
MAX_TREND_SERIES = 8


def handles(plan) -> bool:
    if plan.intent in ("period_comparison", "change_ranking", "campaign_list"):
        return False
    averages = any(catalog.METRICS[m].aggregation == "avg" for m in plan.metrics)
    extra = plan.dimension in dt.EXTRA_DIMENSION_COLUMNS or plan.dimension2 in dt.EXTRA_DIMENSION_COLUMNS
    units = {catalog.unit(m) for m in plan.metrics}
    return bool(plan.dimension2 or plan.intent == "relationship" or plan.exclude or averages or extra
                or (plan.intent == "trend" and (plan.entities or plan.dimension))
                or (plan.visualization and len(plan.metrics) > 1 and plan.intent in ("breakdown", "compare_entities"))
                or (plan.intent == "trend" and plan.visualization and len(units) > 1))


def fmt(metric: str, value) -> str:
    if value is None:
        return "n/a"
    u = catalog.unit(metric)
    if u == "$":
        return f"${value:,.2f}"
    if u == "x":
        return f"{value:,.2f}x"
    if u == "%":
        return f"{value:,.2f}%"
    if u == "count":
        return f"{int(value):,}"
    if u == "s":
        return f"{value:,.1f} s"
    return f"{value:,.2f}"


@dataclass
class Table:
    x_key: str             # "month", "name" or "campaign"
    x_label: str
    rows: list             # [{x_key: label, series key: value}]
    series: list           # [{"key", "label", "metric", "unit"}]
    title: str
    whole: bool = False    # one series covering every value of a breakdown (a pie is possible)
    parts: bool = False    # the series add up to each bar's total (stacking is meaningful)


def execute(plan, filters: dict):
    base = {**(filters or {}), **plan.filters}
    if plan.intent == "relationship":
        return _relationship(plan, {**base, **qp._period_filter(plan)})
    if plan.intent == "trend":
        return _trend(plan, {**base, **qp._period_filter(plan)})
    base = {**base, **qp._period_filter(plan)}
    if plan.dimension2:
        return _cross(plan, base)
    return _breakdown(plan, base)


def _answer(lines, tables, plan, extra_notes=()):
    notes = list(extra_notes)
    charts = []
    if plan.visualization:
        for t in tables:
            charts += charts_for(t, plan.visualization, notes)
        charts = charts[:MAX_CHARTS]
    return qp.PlannedAnswer([x for x in (*lines, *notes) if x], chart=charts[0] if charts else None,
                            charts=charts if len(charts) > 1 else None)


def _dim_label(dimension: str) -> str:
    return dimension.replace("_", " ")


def _scope(plan, base) -> str:
    parts = [f"{k} = {v}" for k, v in plan.filters.items()]
    if base.get("month_from"):
        parts.append(f"period {base['month_from']} to {base['month_to']}")
    return f"Scope: {', '.join(parts)}." if parts else ""


def _metric_series(metrics) -> list:
    return [{"key": m, "label": catalog.label(m), "metric": m, "unit": catalog.unit(m)} for m in metrics]


# --- metric x dimension (one or several metrics), entity comparisons, totals ---------------------

def _breakdown(plan, base):
    metrics = plan.metrics
    if not plan.dimension:
        rows = dt.metric_table(base)
        if not rows:
            return qp.PlannedAnswer(["No campaigns match that selection."], status="no_data")
        r = rows[0]
        lines = [f"{catalog.label(m)}: {fmt(m, r[m])}" for m in metrics] + [f"Based on {r['campaign_count']:,} campaigns.",
                                                                            _scope(plan, base)]
        notes = ["That result is a single figure per metric, so there's nothing to chart."] if plan.visualization else []
        return qp.PlannedAnswer([x for x in (*lines, *notes) if x])
    rows = dt.metric_table(base, group_by=plan.dimension)
    if plan.intent == "compare_entities" and plan.entities:
        rows = [r for r in rows if r["group"] in plan.entities]
    rows = [r for r in rows if r["group"] not in plan.exclude and r.get(metrics[0]) is not None]
    if not rows:
        return qp.PlannedAnswer(["No campaigns match that selection."], status="no_data")
    m0 = metrics[0]
    rows.sort(key=lambda r: r[m0], reverse=(plan.order != "asc"))
    full = len(rows)
    rows = rows[:plan.limit or qp.MAX_ROWS]
    word = "Lowest" if plan.order == "asc" else "Highest"
    scope_word = "of those compared" if plan.intent == "compare_entities" else f"by {_dim_label(plan.dimension)}"
    lines = [f"{word} {catalog.lower_label(m0)} {scope_word}: {rows[0]['group']} ({fmt(m0, rows[0][m0])})."]
    lines += [f"{i}. {r['group']}: " + ", ".join(f"{catalog.label(m)} {fmt(m, r[m])}" for m in metrics)
              for i, r in enumerate(rows, 1)]
    if plan.exclude:
        lines.append(f"Excluded: {', '.join(plan.exclude)}.")
    lines.append(_scope(plan, base))
    table = Table("name", catalog.dimensions()[plan.dimension].label,
                  [{"name": r["group"], **{m: r[m] for m in metrics}} for r in rows], _metric_series(metrics),
                  f"{' and '.join(catalog.label(m) for m in metrics)} by {catalog.dimensions()[plan.dimension].label}",
                  whole=(plan.intent == "breakdown" and not plan.exclude and len(rows) == full))
    return _answer(lines, [table], plan)


# --- metric x dimension x dimension2 (grouped / stacked) -------------------------------------------

def _cross(plan, base):
    dims = catalog.dimensions()
    d1, d2 = plan.dimension, plan.dimension2
    key2 = qr.ENTITY_FILTER_KEYS[d2]
    restricted = bool(plan.entities) and all(e in dims[d2].values for e in plan.entities)
    values2 = [v for v in (plan.entities if restricted else dims[d2].values) if v not in plan.exclude]
    per_value = {v: {r["group"]: r for r in dt.metric_table({**base, key2: v}, group_by=d1)} for v in values2}
    xs = sorted({x for rows in per_value.values() for x in rows})
    if not xs:
        return qp.PlannedAnswer(["No campaigns match that selection."], status="no_data")
    series = [{"key": f"s{i}", "label": v, "value": v} for i, v in enumerate(values2)]
    lines, tables, notes = [], [], []
    for m in plan.metrics:
        additive = catalog.METRICS[m].additive
        rows = []
        for x in xs:
            vals = {s["key"]: (per_value[s["value"]].get(x) or {}).get(m) for s in series}
            if any(v is None for v in vals.values()):
                if not additive:
                    notes.append(f"{x} is left out of the {catalog.lower_label(m)} chart: some combinations have no data.")
                    continue
                vals = {k: (v if v is not None else 0) for k, v in vals.items()}
            rows.append({"name": x, **vals})
        lines.append(f"{catalog.label(m)} by {_dim_label(d1)} and {_dim_label(d2)}:")
        for r in rows[:qp.MAX_ROWS]:
            lines.append(f"- {r['name']}: " + " · ".join(f"{s['label']} {fmt(m, r[s['key']])}" for s in series))
        tables.append(Table("name", dims[d1].label, rows,
                            [{"key": s["key"], "label": s["label"], "metric": m, "unit": catalog.unit(m)} for s in series],
                            f"{catalog.label(m)} by {dims[d1].label} and {dims[d2].label}",
                            parts=additive and not restricted))
    lines.append(_scope(plan, base))
    return _answer(lines, tables, plan, dict.fromkeys(notes))


# --- metric(s) over time for several entities ---------------------------------------------------------

def _trend(plan, base):
    dims = catalog.dimensions()
    entities, notes = list(plan.entities), []
    if not entities and plan.dimension:
        values = [v for v in dims[plan.dimension].values if v not in plan.exclude]
        if len(values) <= MAX_TREND_SERIES and plan.dimension in qr.ENTITY_FILTER_KEYS:
            entities = values
        else:
            notes.append(f"Too many {_dim_label(plan.dimension)} values for one chart; showing the overall trend.")
    entities = [e for e in entities if e not in plan.exclude]
    if entities:
        key = qr.ENTITY_FILTER_KEYS[plan.dimension]
        tables_by = {e: {r["group"]: r for r in dt.metric_table({**base, key: e}, group_by="month")} for e in entities}
    else:
        tables_by = {None: {r["group"]: r for r in dt.metric_table(base, group_by="month")}}
    months = sorted({mo for t in tables_by.values() for mo in t})
    complete = [mo for mo in months if qp._complete(int(mo[:4]), int(mo[5:]), int(mo[5:]))]
    if len(complete) < len(months):
        notes.append(f"Not included: {', '.join(mo for mo in months if mo not in complete)} "
                     f"(incomplete; the data ends on {qp._data_end()}).")
    if not complete:
        return qp.PlannedAnswer(["No campaigns match that selection."], status="no_data")
    lines, tables = [], []
    names = entities or [None]
    for m in plan.metrics:
        for name in names:
            series = [(mo, (tables_by[name].get(mo) or {}).get(m)) for mo in complete]
            series = [(mo, v) for mo, v in series if v is not None]
            if len(series) < 2:
                continue
            (m0, v0), (m1, v1) = series[0], series[-1]
            hi = max(series, key=lambda x: x[1])
            who = f"{name} " if name else ""
            lines.append(f"{who}{catalog.lower_label(m)}: {fmt(m, v0)} in {m0} and {fmt(m, v1)} in {m1} "
                         f"({qp._change(v0, v1)}); highest {fmt(m, hi[1])} in {hi[0]}.")
    lines.append(_scope(plan, base))
    if entities:
        # One chart per metric, one line per entity (same metric: same unit).
        for m in plan.metrics:
            additive = catalog.METRICS[m].additive
            rows = []
            for mo in complete:
                vals = {f"s{i}": (tables_by[e].get(mo) or {}).get(m) for i, e in enumerate(entities)}
                if any(v is None for v in vals.values()):
                    if not additive:
                        continue  # a missing ratio isn't zero
                    vals = {k: v or 0 for k, v in vals.items()}  # no campaigns that month: nothing was summed
                rows.append({"month": mo, **vals})
            tables.append(Table("month", "Month", rows,
                                [{"key": f"s{i}", "label": e, "metric": m, "unit": catalog.unit(m)} for i, e in enumerate(entities)],
                                f"Monthly {catalog.label(m)}: " + " vs ".join(entities) if len(entities) <= 3
                                else f"Monthly {catalog.label(m)} by {dims[plan.dimension].label}",
                                parts=catalog.METRICS[m].additive and not plan.entities))
    else:
        rows = [{"month": mo, **{m: tables_by[None][mo].get(m) for m in plan.metrics}} for mo in complete]
        rows = [r for r in rows if all(r[m] is not None for m in plan.metrics)]
        tables.append(Table("month", "Month", rows, _metric_series(plan.metrics),
                            "Monthly " + " and ".join(catalog.label(m) for m in plan.metrics)))
    return _answer(lines, tables, plan, notes)


# --- relationship between two measures -------------------------------------------------------------

def _strength(r: float) -> str:
    a = abs(r)
    word = "no clear" if a < 0.1 else "a weak" if a < 0.3 else "a moderate" if a < 0.5 else "a strong"
    return f"{word}{'' if a < 0.1 else ' positive' if r > 0 else ' negative'} relationship"


def _relationship(plan, base):
    x, y = plan.metrics
    notes = []
    if plan.dimension:
        rows = [r for r in dt.metric_table(base, group_by=plan.dimension) if r[x] is not None and r[y] is not None]
        pts = [{"name": r["group"], x: r[x], y: r[y]} for r in rows]
        n, x_key, unit_word = len(pts), "name", f"{_dim_label(plan.dimension)} values"
        r_value = None
        if n >= 3:
            import numpy as np
            xs, ys = np.array([p[x] for p in pts], float), np.array([p[y] for p in pts], float)
            r_value = round(float(np.corrcoef(xs, ys)[0, 1]), 3) if xs.std() > 0 and ys.std() > 0 else None
    else:
        res = dt.campaign_points(x, y, base)
        n, r_value, x_key, unit_word = res["count"], res["correlation"], "campaign", "campaigns"
        points = res["points"]
        step = max(1, math.ceil(len(points) / cb.MAX_SCATTER_POINTS))
        if step > 1:
            notes.append(f"The chart shows every {step}th campaign by ID ({len(points[::step]):,} of {n:,}); "
                         "the correlation uses all of them.")
        pts = [{"campaign": p["id"], x: p["x"], y: p["y"]} for p in points[::step]]
    if n < 2:
        return qp.PlannedAnswer(["No campaigns match that selection."], status="no_data")
    lines = [f"{catalog.label(x)} vs {catalog.label(y)} across {n:,} {unit_word}: "
             + (f"correlation r = {r_value:.3f}, {_strength(r_value)}." if r_value is not None
                else "too few points to measure a correlation."),
             "Correlation shows how the two move together; it doesn't show that one causes the other.",
             _scope(plan, base)]
    charts = []
    if plan.visualization:
        if plan.visualization != "scatter":
            notes.append("A relationship between two measures is shown as a scatter plot.")
        spec = {"type": "scatter", "title": f"{catalog.label(x)} vs {catalog.label(y)}", "x_key": x_key,
                "x_label": "Campaign" if x_key == "campaign" else catalog.dimensions()[plan.dimension].label,
                "series": [{"key": m, "label": catalog.label(m), "unit": catalog.unit(m)} for m in (x, y)], "data": pts}
        try:
            charts.append(cb.validate_chart(spec))
        except cb.ChartSpecError as e:
            notes.append(f"No chart: {e}.")
    return qp.PlannedAnswer([l for l in (*lines, *notes) if l], chart=charts[0] if charts else None)


# --- visualization planner: result shape + units -> validated chart specs ----------------------------

def _series_entry(s: dict) -> dict:
    if s["key"] == s["metric"] and s["label"] == catalog.label(s["metric"]):
        return {"key": s["key"], "label": s["label"], "unit": s["unit"]}  # the original (metric) form
    return {"key": s["key"], "label": s["label"], "unit": s["unit"], "metric": s["metric"]}


def _choose(table: Table, group: list, requested: str, notes: list) -> str:
    additive = all(catalog.METRICS[s["metric"]].additive for s in group)
    nonneg = all(r[s["key"]] >= 0 for r in table.rows for s in group)
    stackable = len(group) > 1 and table.parts and additive and nonneg
    if table.x_key == "month":
        if requested in cb.STACKED and stackable:
            return requested
        if requested == "bar" and len(group) <= 3:
            return "bar"
        if requested == "pie":
            notes.append("A pie chart can't show change over time; shown as a line chart.")
        elif requested in cb.STACKED:
            notes.append("Stacking needs parts of one total; shown as lines.")
        return "line"
    if len(group) == 1:
        s = group[0]
        if requested == "pie":
            if table.whole and s["metric"] in cb.PIE_METRICS and nonneg and len(table.rows) <= cb.MAX_PIE_SLICES:
                return "pie"
            notes.append(f"{s['label'] if s['key'] == s['metric'] else catalog.label(s['metric'])} can't be shown as "
                         "shares of one whole here, so it's a bar chart.")
        elif requested == "line":
            notes.append("Line charts are for trends over time; shown as a bar chart.")
        return "bar"
    if requested in cb.STACKED:
        if stackable:
            return requested
        notes.append("Stacking needs measures that are parts of one total; shown as grouped bars.")
    elif requested == "pie":
        notes.append("A pie chart shows one measure; shown as grouped bars.")
    return "bar"


def charts_for(table: Table, requested: str, notes: list) -> list:
    if not table.rows:
        return []
    groups = {}
    for s in table.series:
        groups.setdefault(s["unit"], []).append(s)
    if len(groups) > 1:
        names = [" and ".join(catalog.label(s["metric"]) if s["key"] == s["metric"] else s["label"] for s in g)
                 for g in groups.values()]
        notes.append(f"{'; '.join(names)} use different units, so they're shown in separate charts "
                     "rather than on one misleading axis.")
    charts = []
    for group in list(groups.values())[:MAX_CHARTS]:
        kind = _choose(table, group, requested, notes)
        title = table.title
        if len(groups) > 1:
            title = " and ".join(catalog.label(s["metric"]) for s in group[:3]) + f" by {table.x_label}"
        spec = {"type": kind, "title": title, "x_key": table.x_key, "x_label": table.x_label,
                "series": [_series_entry(s) for s in group],
                "data": [{table.x_key: r[table.x_key], **{s["key"]: r[s["key"]] for s in group}} for r in table.rows]}
        try:
            charts.append(cb.validate_chart(spec))
        except cb.ChartSpecError as e:
            notes.append(f"One chart was left out ({e}).")
    return charts
