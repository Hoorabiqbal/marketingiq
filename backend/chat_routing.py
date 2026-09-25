"""
/api/chat answers: every chat question goes through the Query Router. There is no
free-form LLM fallback.

    latest user message
        ├─ DIRECT_DATABASE ... data tools -> formatted here, no LLM call
        ├─ LLM_REQUIRED ...... data tools -> compact analysis -> LLM Adapter (one call,
        │                      grounding-validated); on an LLM error: the data summary + a notice
        ├─ UNSUPPORTED / NEEDS_CLARIFICATION ... the router's own message, no LLM call
        ├─ CHART ............. explicit chart request -> data tools -> validated chart spec
        │                      (chart_builder.py), no LLM call; "Show ..." answers that are already
        │                      a series or ranking also carry the same kind of chart
        └─ follow-up that depends on earlier turns ("what about TikTok?"), empty or rejected
           input ... a request to ask the full question, no LLM call

At most one LLM call per question, never retried here.

Answers are HTML (the dashboard inserts `answer` as HTML): every value is escaped
and line breaks are <br>.
"""
import html
import re
from dataclasses import dataclass

import chart_builder
import data_tools as dt
import grounding
import query_router as qr

# Only checked when there IS earlier conversation: wording that leans on a previous turn.
FOLLOW_UP_RE = re.compile(
    r"^\s*(?:and|also|but|so|then|or|what about|how about|what if|same|ok(?:ay)?)\b"
    r"|\b(?:it|its|that|those|these|them|they|this|there|ones?|instead|previous|above|earlier|same)\b",
    re.IGNORECASE,
)


CLARIFY = qr.Route.NEEDS_CLARIFICATION.value
ASK_A_QUESTION = "Ask a question about your MarketingIQ campaign data. " + qr.SUPPORTED_HINT
ASK_IN_FULL = ("Please ask that as a complete question, for example \"What is TikTok's ROAS?\". "
               "Each question is answered from the data on its own, so answers can't build on "
               "earlier messages.")


@dataclass
class ChatDecision:
    route: str                 # DIRECT_DATABASE / LLM_REQUIRED / UNSUPPORTED / NEEDS_CLARIFICATION / DATA_ERROR
    answer: str                # HTML
    reason: str = None         # the LLM status, or why no data answer was given
    chart: dict = None         # a validated chart spec (chart_builder.validate_chart), or None


def decide(question: str, has_history: bool, filters: dict, explainer) -> ChatDecision:
    if not question or not question.strip():
        return ChatDecision(CLARIFY, html.escape(ASK_A_QUESTION), reason="no_question")
    if has_history and FOLLOW_UP_RE.search(question):
        return ChatDecision(CLARIFY, html.escape(ASK_IN_FULL), reason="follow_up")
    if chart_builder.wants_chart(question):
        return _chart(question, filters)
    try:
        r = qr.route_query(question, filters, explainer=explainer)
    except qr.InvalidQueryError as e:
        return ChatDecision(CLARIFY, html.escape(e.public_message), reason="invalid_query")
    except qr.QueryRouterError as e:
        return ChatDecision("DATA_ERROR", html.escape(e.public_message), reason=type(e).__name__)

    route = r["route"]
    if route == qr.Route.DIRECT_DATABASE.value:
        chart = chart_builder.from_direct_result(r) if chart_builder.SHOW_RE.search(question) else None
        return ChatDecision(route, format_direct(r), chart=chart)
    if route == qr.Route.LLM_REQUIRED.value:
        llm = r["llm"]
        if llm["status"] == "ok":
            return ChatDecision(route, explanation_html(r["explanation"]), reason="ok")
        if llm["status"] == "skipped":  # empty analysis: the LLM was never called
            return ChatDecision(route, html.escape(llm["message"]), reason=llm.get("error"))
        return ChatDecision(route, _data_without_explanation(question, r, llm["message"]), reason=llm.get("error"))
    # UNSUPPORTED (data not in the dataset, off-topic) or NEEDS_CLARIFICATION: the router's message.
    return ChatDecision(route, html.escape(r.get("message") or ASK_A_QUESTION), reason=r.get("reason"))


CHART_FAILED = "I couldn't build a reliable chart for that request. Try asking for the figures as text."


def _chart(question: str, filters: dict) -> ChatDecision:
    try:
        c = chart_builder.build_chart(question, filters)
    except qr.InvalidQueryError as e:
        return ChatDecision(CLARIFY, html.escape(e.public_message), reason="invalid_query")
    except qr.QueryRouterError as e:
        return ChatDecision("DATA_ERROR", html.escape(e.public_message), reason=type(e).__name__)
    except chart_builder.ChartSpecError as e:
        qr.logger.warning(f"chart spec rejected: {e}")
        return ChatDecision("DATA_ERROR", html.escape(CHART_FAILED), reason="chart_invalid")
    if c.chart is None:
        return ChatDecision(c.route, html.escape(c.message), reason="chart_unresolved")
    answer = html.escape(c.message)
    note = _filter_note({"filters_applied": c.filters_applied, "scope": c.scope})
    if note:
        answer += f"<br><i>Filtered view: {_e(note)}.</i>"
    return ChatDecision("CHART", answer, reason="chart", chart=c.chart)


def _data_without_explanation(question: str, r: dict, notice: str) -> str:
    """The explanation failed (rate limit, timeout, busy, ...): show the supplied figures, which
    come straight from DuckDB, followed by the neutral notice."""
    context = grounding.build_llm_context(question, {"filters_applied": r["filters_applied"],
                                                     "focus_metrics": r["focus_metrics"], "results": r["analysis"]})
    return explanation_html(grounding.safe_summary(context, note=notice))


# ---------------------------------------------------------------------------
# Deterministic answers for DIRECT_DATABASE results (no LLM involved)
# ---------------------------------------------------------------------------

MONEY = {"spend", "revenue", "profit", "cpa", "cpc"}
PERCENT = {"ctr", "conversion_rate", "roi_pct"}
COUNTS = {"campaign_count", "conversions", "clicks", "impressions"}
SUM_METRICS = {"spend", "revenue", "profit", "conversions", "clicks", "impressions"}
LABELS = {"spend": "spend", "revenue": "revenue", "profit": "profit", "roas": "ROAS", "roi_pct": "ROI",
          "cpa": "CPA", "cpc": "CPC", "ctr": "CTR", "conversion_rate": "conversion rate",
          "conversions": "conversions", "clicks": "clicks", "impressions": "impressions",
          "campaign_count": "number of campaigns"}
PLURALS = {"age_group": "age groups", "budget_tier": "budget tiers", "creative_format": "creative formats",
           "creative_emotion": "creative emotions", "income_bracket": "income brackets",
           "audience_interest": "audience interests", "operating_system": "operating systems",
           "retargeting": "retargeting groups"}
OPERATOR_WORDS = {">": "above", "<": "below", ">=": "at least", "<=": "at most", "==": "equal to"}
COLUMN_TO_DIMENSION = {col: name for name, col in dt.DIMENSION_COLUMNS.items()}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
KEY_METRICS = ["revenue", "spend", "profit", "roas", "cpa", "ctr"]


def fmt(metric: str, value) -> str:
    if value is None:
        return "n/a"
    if metric in MONEY:
        return f"${value:,.2f}"
    if metric == "roas":
        return f"{value:,.2f}x"
    if metric in PERCENT:
        return f"{value:,.2f}%"
    if metric in COUNTS:
        return f"{int(value):,}"
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


def _e(value) -> str:
    return html.escape(str(value))


def _plural(dimension: str) -> str:
    return PLURALS.get(dimension, dimension.replace("_", " ") + "s")


def _month(ym: str) -> str:
    try:
        year, month = ym.split("-")
        return f"{MONTHS[int(month) - 1]} {year}"
    except (ValueError, IndexError):
        return ym


def _entity_line(m: dict, extra_metrics=()) -> str:
    metrics = list(dict.fromkeys([*extra_metrics, *KEY_METRICS]))
    parts = ", ".join(f"{LABELS[k]} {fmt(k, m.get(k))}" for k in metrics if k in m)
    return f"{_e(m['name'])}: {parts} ({fmt('campaign_count', m['campaign_count'])} campaigns)"


def _not_found_line(res: dict) -> str:
    missing = ", ".join(_e(n) for n in res["not_found"])
    available = ", ".join(_e(v) for v in res.get("available_values", []))
    verb = "is" if len(res["not_found"]) == 1 else "are"
    return (f"{missing} {verb} not in the MarketingIQ dataset. "
            f"Available {_e(_plural(res['dimension']))}: {available}.")


def _totals(r, res):
    if res["campaign_count"] == 0:
        return ["No campaigns match the current filters."]
    n = fmt("campaign_count", res["campaign_count"])
    if qr.COUNT_RE.search(r["query"].lower()) and not r["focus_metrics"]:
        return [f"There are {n} campaigns{' in the current filtered view' if _filter_note(r) else ' in the dataset'}."]
    focus = [m for m in r["focus_metrics"] if m in res]
    if not focus:
        return [f"Across {n} campaigns: " + ", ".join(
            f"{LABELS[k]} {fmt(k, res[k])}" for k in ("revenue", "spend", "profit", "roas", "roi_pct", "conversions", "cpa"))
            + "."]
    lines = []
    for m in focus:
        prefix = "Total" if m in SUM_METRICS else "Overall"
        lines.append(f"{prefix} {LABELS[m]} is {fmt(m, res[m])} across {n} campaigns.")
    return lines


def _rank(r, res):
    rows = res.get("results") or []
    if not rows:
        return ["No campaigns match the current filters."]
    metric, order = res["metric"], r["tool_input"].get("order", "desc")
    dimension = r["tool_input"]["dimension"]
    if r["tool_input"].get("limit") == 1:
        top = rows[0]
        if metric == "campaign_count":
            word = "fewest" if order == "asc" else "most"
            return [f"{_e(top['name'])} has the {word} campaigns: {fmt(metric, top[metric])}."]
        word = "lowest" if order == "asc" else "highest"
        return [f"{_e(top['name'])} has the {word} {LABELS[metric]}: {fmt(metric, top[metric])} "
                f"({fmt('campaign_count', top['campaign_count'])} campaigns)."]
    head = f"{LABELS[metric][0].upper()}{LABELS[metric][1:]} by {_e(dimension.replace('_', ' '))} " \
           f"({'lowest' if order == 'asc' else 'highest'} first):"
    lines = [head]
    for i, row in enumerate(rows, 1):
        tail = "" if metric == "campaign_count" else f" ({fmt('campaign_count', row['campaign_count'])} campaigns)"
        lines.append(f"{i}. {_e(row['name'])}: {fmt(metric, row[metric])}{tail}")
    return lines


def _trend(r, res):
    metric, series, period = res["metric"], res.get("series") or [], res.get("period")
    label = LABELS.get(metric, metric)
    if not series:
        when = _month(period) if period and len(period) == 7 else (period or "the dataset")
        return [f"There is no {_e(label)} data for {_e(when)}."]
    if period and len(period) == 7 and len(series) == 1:
        return [f"{label[0].upper()}{label[1:]} in {_month(series[0]['month'])} was {fmt(metric, series[0]['value'])}."]
    lines = [f"Monthly {_e(label)}{' in ' + _e(period) if period else ''}:"]
    lines += [f"{_month(p['month'])}: {fmt(metric, p['value'])}" for p in series]
    return lines


def _campaigns(r, res):
    conditions = r["tool_input"].get("conditions") or []
    sort_by, order = r["tool_input"].get("sort_by", "profit"), r["tool_input"].get("order", "asc")
    matched = fmt("campaign_count", res["matched_count"])
    if conditions:
        crit = " and ".join(f"{LABELS.get(c['field'], c['field'])} {OPERATOR_WORDS.get(c['operator'], c['operator'])} "
                            f"{fmt(c['field'], c['value'])}" for c in conditions)
        lines = [f"{matched} campaigns have {_e(crit)}." if res["matched_count"] else f"No campaigns have {_e(crit)}."]
    else:
        lines = []
    if res["campaigns"]:
        lines.append(f"Top {len(res['campaigns'])} by {LABELS.get(sort_by, sort_by)} "
                     f"({'lowest' if order == 'asc' else 'highest'} first):")
        for i, c in enumerate(res["campaigns"], 1):
            lines.append(f"{i}. {_e(c['id'])} ({_e(c['platform'])}, {_e(c['objective'])}): ROAS {fmt('roas', c['roas'])}, "
                         f"spend {fmt('spend', c['spend'])}, revenue {fmt('revenue', c['revenue'])}")
    return lines


def _compare(r, res):
    lines = [_entity_line(m, r["focus_metrics"]) for m in res.get("results", [])]
    if res.get("not_found"):
        lines.append(_not_found_line(res))
    return lines


def _share(r, res):
    if res.get("not_found"):
        return [_not_found_line(res)]
    m = res["metric"]
    return [f"{_e(res['name'])} accounts for {res['share_pct']:.2f}% of total {LABELS.get(m, m)} "
            f"({fmt(m, res['entity_value'])} of {fmt(m, res['total_value'])})."]


def _stats(r, res):
    lines = []
    for field, s in res.items():
        label = LABELS.get(field, field.replace("_", " "))
        if "error" in s:
            lines.append(f"{_e(label)} is not available.")
        elif s["median"] is None:
            lines.append(f"No campaigns match the current filters for {_e(label)}.")
        else:
            lines.append(f"{label[0].upper()}{label[1:]} per campaign: median {fmt(field, s['median'])}, "
                         f"middle 50% {fmt(field, s['p25'])} to {fmt(field, s['p75'])}, "
                         f"range {fmt(field, s['min'])} to {fmt(field, s['max'])}, average {fmt(field, s['mean'])}.")
    return lines


def _fatigue(r, res):
    if not res["buckets"]:
        return ["No campaigns match the current filters."]
    return ["CTR by creative age:"] + [f"{_e(b['age_range'])} days: {fmt('ctr', b['ctr'])}" for b in res["buckets"]]


def _fields(r, res):
    lines = [f"The dataset has {fmt('campaign_count', res['total_campaigns'])} campaigns from "
             f"{_e(res['date_range'][0])} to {_e(res['date_range'][1])}."]
    for dim, values in res["dimensions"].items():
        shown = ", ".join(values) if isinstance(values, list) else values
        lines.append(f"{_e(dim.replace('_', ' ').capitalize())}: {_e(shown)}")
    lines.append("Metrics: " + _e(", ".join(res["metrics"])))
    return lines


FORMATTERS = {
    "get_totals": _totals, "rank_dimension": _rank, "trend_over_time": _trend,
    "filter_campaigns": _campaigns, "compare_entities": _compare, "get_entity_metrics": _compare,
    "percentage_share": _share, "get_numeric_field_stats": _stats, "get_creative_fatigue": _fatigue,
    "list_available_fields": _fields,
}


def _filter_note(r) -> str:
    """Active dashboard filters (plus any entity scope the router added), as the data layer applied them."""
    active = {**(r.get("filters_applied") or {}), **(r.get("scope") or {})}
    parts = []
    for col, _, value in dt._filter_conditions(active):
        if isinstance(value, bool):
            parts.append("retargeting only" if value else "cold audiences only")
        else:
            parts.append(f"{COLUMN_TO_DIMENSION.get(col, col).replace('_', ' ')} = {value}")
    return ", ".join(dict.fromkeys(parts))


def format_direct(r: dict) -> str:
    formatter = FORMATTERS.get(r["tool"])
    lines = formatter(r, r["result"]) if formatter else [_e(r["result"])]
    note = _filter_note(r)
    if note:
        lines.append(f"<i>Filtered view: {_e(note)}.</i>")
    return "<br>".join(lines)


def explanation_html(text: str) -> str:
    """LLM text -> safe HTML: escaped, **bold** kept, line breaks preserved."""
    out = html.escape(text.strip())
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return out.replace("\n", "<br>")
