"""
MarketingIQ grounding layer: the boundary between the analytics engine and any LLM provider.

    router analysis (raw data_tools results)
        -> build_llm_context()      typed, unit-aware, compact context  -> provider (Groq)
    provider text
        -> validate_explanation()   deterministic check of every number in the answer
        -> safe_summary()           deterministic answer used when the check fails

Provider-independent: every provider gets the same context and its answer gets the same checks.
Metric definitions below DESCRIBE what data_tools.py computes; nothing here computes a metric.

Validator limits (by design, see README): it checks numbers written with digits, not number words
("three times"); it cannot judge whether a comparison between two supported values is correct
("A is higher than B"); and its causal check is a keyword heuristic.
"""
import re
from dataclasses import dataclass, field

import data_tools as dt

# metric -> (unit, formula for derived metrics). Units: "$" money, "x" ratio, "%" already a
# percentage, "count".
METRICS = {
    "campaign_count": ("count", None),
    "spend": ("$", None),
    "revenue": ("$", None),
    "profit": ("$", None),
    "conversions": ("count", None),
    "clicks": ("count", None),
    "impressions": ("count", None),
    "roas": ("x", "revenue / spend; a ratio (6.54 means 6.54x), never a percentage"),
    "roi_pct": ("%", "(revenue - spend) / spend * 100"),
    "cpa": ("$", "spend / conversions"),
    "cpc": ("$", "spend / clicks"),
    "ctr": ("%", "clicks / impressions * 100; already a percentage (2.16 means 2.16%)"),
    "conversion_rate": ("%", "conversions / clicks * 100; already a percentage"),
}
# The dataset has no currency column; the dashboard formats money with "$" (site/dashboard.html).
CURRENCY_NOTE = "amounts shown with $ as in the MarketingIQ dashboard; the dataset records no currency code"
TREND_UNITS = {"revenue": "$", "spend": "$", "profit": "$", "conversions": "count"}
COLUMN_TO_DIMENSION = {col: name for name, col in dt.DIMENSION_COLUMNS.items()}
CHANGE_RE = re.compile(r"\b(?:decl|drop|decreas|increas|chang|grow|grew|fell|fall|rose|rising|trend|improv|worsen)",
                       re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. Typed context
# ---------------------------------------------------------------------------

def _unit(metric):
    return METRICS.get(metric, (None,))[0]


def _named_row(row: dict) -> dict:
    """Entity rows with the name first, so each value sits next to what it describes."""
    return {"name": row.get("name"), **{k: v for k, v in row.items() if k != "name"}}


def _truncation(res: dict, key: str) -> dict:
    n = res.get(f"{key}_truncated_from")
    return {"truncated_from": n} if n else {}


def _order_words(order) -> str:
    return "lowest first" if order == "asc" else "highest first"


def _section(item: dict) -> dict:
    tool, inp, res = item.get("tool"), item.get("tool_input") or {}, item.get("result")
    scope = item.get("scope")
    if tool == "get_totals":
        return {"type": "aggregate", "of": "all campaigns in scope", "metrics": res}
    if tool == "rank_dimension":
        return {"type": "comparison", "dimension": res["dimension"], "ranked_by": res["metric"],
                "order": _order_words(inp.get("order", "desc")), "entity_count": len(res["results"]),
                **_truncation(res, "results"), "entities": [_named_row(r) for r in res["results"]]}
    if tool in ("compare_entities", "get_entity_metrics"):
        out = {"type": "comparison", "dimension": res["dimension"], "entity_count": len(res["results"]),
               "entities": [_named_row(r) for r in res["results"]]}
        if res.get("not_found"):
            out["not_in_dataset"] = res["not_found"]
        return out
    if tool == "trend_over_time":
        metric = res["metric"]
        out = {"type": "time_series", "metric": metric, "unit": TREND_UNITS.get(metric),
               "granularity": "calendar month (YYYY-MM), monthly total",
               "of": {"only": scope} if scope else "all campaigns in scope"}
        if res.get("period"):
            out["period"] = res["period"]
        return {**out, **_truncation(res, "series"), "points": res["series"]}
    if tool == "get_creative_fatigue":
        return {"type": "creative_age_buckets", "dimension": "creative_age", "unit": "days",
                "meaning": "age of the ad creative in days (not audience age, not a date)",
                "metric": "ctr",
                "buckets": [{"creative_age_days": b["age_range"], "ctr": b["ctr"]} for b in res["buckets"]]}
    if tool == "filter_campaigns":
        conds = [{"field": c["field"], "operator": c["operator"], "value": c["value"], "unit": _unit(c["field"])}
                 for c in inp.get("conditions") or []]
        n, rows = res["matched_count"], res["campaigns"]
        out = {"type": "filtered_subset", "conditions": conds, "matched_campaign_count": n}
        if res.get("aggregates"):
            out["aggregates"] = res["aggregates"]
        if rows:
            order = inp.get("order", "asc")
            out["examples"] = {
                "count": len(rows),
                "selection": f"{len(rows)} of the {n} matching campaigns, sorted by {inp.get('sort_by', 'profit')} "
                             f"{'ascending' if order == 'asc' else 'descending'}: individual campaigns, "
                             "not averages and not necessarily typical",
                "campaigns": rows}
        return out
    if tool == "percentage_share":
        return {"type": "share", **res, "unit": _unit(res["metric"]), "share_unit": "%"}
    if tool == "get_numeric_field_stats":
        return {"type": "per_campaign_distribution", "fields": {f: {"unit": _unit(f), **s} for f, s in res.items()}}
    return {"type": "other", "tool": tool, "data": res}


def _active_filters(filters: dict) -> dict:
    """Only filters that actually narrow the data (the dashboard also sends "All Platforms" etc.)."""
    out = {}
    for col, _, value in dt._filter_conditions(filters or {}):
        name = COLUMN_TO_DIMENSION.get(col, col)
        out[name] = ("retargeting only" if value else "cold audience only") if isinstance(value, bool) else value
    return out


def _analysis_scope(question: str, sections: list, focus: list) -> dict:
    cannot = ["proving causes: the data is observational, so reasons can only be hypotheses"]
    series = {s["metric"] for s in sections if s["type"] == "time_series"}
    if CHANGE_RE.search(question or ""):
        for m in focus:
            if m not in series:
                cannot.append(f"any change over time in {m}: no time series for it is supplied")
    for s in sections:
        if s["type"] == "creative_age_buckets":
            cannot.append("change over time: creative-age buckets compare creatives of different ages")
        if s["type"] == "filtered_subset":
            if "examples" in s:
                cannot.append("treating the example campaigns as typical of the filtered subset")
            if "aggregates" not in s:
                cannot.append("aggregate figures for the filtered subset (only examples are supplied)")
    return {"supports": ["describing and comparing the supplied figures"], "does_not_support": cannot}


def _metrics_present(node, found: set):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in METRICS and not isinstance(v, (dict, list)):
                found.add(k)
            _metrics_present(v, found)
    elif isinstance(node, list):
        for x in node:
            _metrics_present(x, found)


def build_llm_context(question: str, analysis: dict) -> dict:
    """analysis: the router's {"filters_applied", "focus_metrics", "results": [{tool, tool_input, result}]}.
    Returns the typed context every provider receives. Unknown or unexpected shapes are passed
    through as type "other" (the adapter still refuses anything that isn't plain JSON)."""
    sections = []
    for item in analysis.get("results") or []:
        try:
            sections.append(_section(item))
        except (KeyError, TypeError, AttributeError, IndexError):
            sections.append({"type": "other", "tool": item.get("tool") if isinstance(item, dict) else None,
                             "data": item.get("result") if isinstance(item, dict) else item})
    focus = [m for m in analysis.get("focus_metrics") or [] if isinstance(m, str)]
    used = set()
    _metrics_present(sections, used)
    # A series or bucket list carries its metric in a field, not as a key.
    used.update(s["metric"] for s in sections if s["type"] in ("time_series", "creative_age_buckets"))
    units = {m: f"{METRICS[m][0]}: {METRICS[m][1]}" if METRICS[m][1] else METRICS[m][0]
             for m in METRICS if m in used}
    active = _active_filters(analysis.get("filters_applied"))
    ctx = {"population": "dashboard-filtered view" if active else "all campaigns",
           "active_filters": active, "focus_metrics": focus, "metric_units": units}
    if any(METRICS[m][0] == "$" for m in used if m in METRICS):
        ctx["currency"] = CURRENCY_NOTE
    ctx["sections"] = sections
    ctx["analysis_scope"] = _analysis_scope(question, sections, focus)
    return ctx


# ---------------------------------------------------------------------------
# 2. Facts: every number the provider was given, with its unit, metric and scope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    value: float
    unit: str            # "$", "x", "%", "count", "days", "name" (number inside a label), "any"
    metric: str = None
    scope: str = None    # aggregate / entity / series / bucket / subset / example / condition / meta / question
    entity: str = None
    month: str = None    # "YYYY-MM" for time-series points


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _metric_facts(metrics: dict, scope: str, entity=None):
    for k, v in (metrics or {}).items():
        if _is_num(v) and k in METRICS:
            yield Fact(abs(float(v)), METRICS[k][0], k, scope, entity)


def _name_facts(name, dimension):
    """Numbers inside labels such as "25-34", "<$50K" or "61-90"."""
    for c in _extract(str(name)):
        yield Fact(c.value, c.unit or "name", dimension, "meta", str(name))


def _any_facts(node, scope):
    if _is_num(node):
        yield Fact(abs(float(node)), "any", None, scope)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _any_facts(v, scope)
    elif isinstance(node, list):
        for v in node:
            yield from _any_facts(v, scope)
    elif isinstance(node, str):
        for c in _extract(node):
            yield Fact(c.value, "any", None, scope)


def collect_facts(context: dict, question: str = "") -> list:
    facts = []
    add = facts.extend
    for s in context.get("sections", []):
        t = s.get("type")
        if t == "aggregate":
            add(_metric_facts(s.get("metrics"), "aggregate"))
        elif t == "comparison":
            add([Fact(float(s.get("entity_count", 0)), "count", "entity_count", "meta")])
            for e in s.get("entities", []):
                add(_metric_facts(e, "entity", e.get("name")))
                add(_name_facts(e.get("name"), s.get("dimension")))
        elif t == "time_series":
            unit = s.get("unit") or "any"
            for p in s.get("points", []):
                if _is_num(p.get("value")):
                    add([Fact(abs(float(p["value"])), unit, s.get("metric"), "series", None, p.get("month"))])
        elif t == "creative_age_buckets":
            for b in s.get("buckets", []):
                label = b.get("creative_age_days")
                if _is_num(b.get("ctr")):
                    add([Fact(float(b["ctr"]), "%", "ctr", "bucket", label)])
                for c in _extract(str(label)):
                    add([Fact(c.value, "days", "creative_age", "meta", label)])
        elif t == "filtered_subset":
            for c in s.get("conditions", []):
                if _is_num(c.get("value")):
                    add([Fact(abs(float(c["value"])), c.get("unit") or "any", c.get("field"), "condition")])
            add([Fact(float(s.get("matched_campaign_count", 0)), "count", "matched_campaign_count", "subset")])
            add(_metric_facts(s.get("aggregates"), "subset"))
            ex = s.get("examples") or {}
            if ex:
                add([Fact(float(ex.get("count", 0)), "count", "example_count", "meta")])
            for row in ex.get("campaigns", []):
                add(_metric_facts(row, "example", row.get("id")))
        elif t == "share":
            unit = s.get("unit") or "any"
            for k in ("entity_value", "total_value"):
                if _is_num(s.get(k)):
                    add([Fact(abs(float(s[k])), unit, s.get("metric"), "aggregate")])
            if _is_num(s.get("share_pct")):
                add([Fact(float(s["share_pct"]), "%", "share", "aggregate")])
        else:
            add(_any_facts(s, "other"))
    for dim, value in (context.get("active_filters") or {}).items():
        add(_name_facts(value, dim))
    for c in _extract(question or ""):
        add([Fact(c.value, "any", None, "question")])
    return facts


def _years(context: dict, question: str) -> set:
    text = str(context) + " " + (question or "")
    return set(re.findall(r"(?<!\d)(20\d{2})-(?:0[1-9]|1[0-2])\b", text)) | set(YEAR_WORD_RE.findall(question or ""))


# ---------------------------------------------------------------------------
# 3. Numeric claims in generated text
# ---------------------------------------------------------------------------

_SEP = "[,\u202f\u2009\u00a0]"
_SP = "[ \u202f\u00a0]?"
NUM_RE = re.compile(
    r"(?<![\w.])(?P<cur>US\$|\$)?" + _SP +
    r"(?P<num>\d{1,3}(?:" + _SEP + r"\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:" + _SP + r"(?P<scale>thousand|million|billion|mn|bn|[KMB]|k)(?![A-Za-z]))?"
    r"(?:" + _SP + r"(?P<unit>%|percent(?:age points?)?(?![A-Za-z])|per cent(?![A-Za-z])|pp(?![A-Za-z])"
    r"|x(?![A-Za-z0-9])|\u00d7|times(?![A-Za-z])|days?(?![A-Za-z])|yrs?(?![A-Za-z])|years?(?![A-Za-z])"
    r"|-year-olds?))?"
)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mn": 1e6, "b": 1e9, "billion": 1e9, "bn": 1e9}
YEAR_WORD_RE = re.compile(r"\b(20\d{2})\b")
HYPHENS = {c: "-" for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2212)}  # ‐ ‑ ‒ – − (validation only)
ISO_MONTH_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])(?:-\d{2})?\b")
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_RE = re.compile(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
                      r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?(?:,?\s+(20\d{2}))?")
STRUCTURAL_NOUN_RE = re.compile(
    r"^\s+(?:(?:key|main|possible|likely|important|major|clear|notable|other|more|potential|broad)\s+)?"
    r"(?:reasons?|observations?|points?|factors?|ways?|things?|takeaways?|insights?|hypothes[ie]s|steps?"
    r"|recommendations?|options?|possibilit(?:y|ies)|drivers?|areas?|questions?|explanations?|ideas?"
    r"|patterns?|themes?|caveats?|notes?|parts?|sections?|bullets?)\b", re.IGNORECASE)
STRUCTURAL_BEFORE_RE = re.compile(r"(?:\b(?:step|option|point|reason|phase|part|section|item|key|no\.|number"
                                  r"|observation|takeaway|hypothesis|scenario)\s*|#)$", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"(?:^|\n)[ \t>*\-\u2022]*(?:\*\*)?$")
INLINE_MARKER_RE = re.compile(r"(?:^|(?<=\s))(\d)\)(?=\s+[A-Za-z])")
DERIVED_AFTER_RE = re.compile(r"^\s*(?:higher|lower|more|less|greater|larger|smaller|bigger|cheaper|better|worse"
                              r"|increase|decrease|drop|rise|growth|gain|decline|as (?:much|high|many|large))\b",
                              re.IGNORECASE)
PER_DOLLAR_RE = re.compile(r"^\s*(?:(?:in|of)\s+)?(?:revenue\s+)?(?:earned\s+|generated\s+|returned\s+)?"
                           r"(?:for|per|on|from)\s+(?:each|every)?\s*(?:\$\s?1(?![\d.])|dollar)", re.IGNORECASE)
TOP_N_BEFORE_RE = re.compile(r"\b(?:top|bottom|first|last|all|the)\s*$", re.IGNORECASE)


@dataclass
class Claim:
    text: str
    start: int
    end: int
    value: float
    tol: float
    unit: str            # "$", "%", "x", "days", "years", "scaled" (a K/M/B number with no unit), None
    integer: bool


def _unit_of(m) -> str:
    u = (m.group("unit") or "").lower()
    if m.group("cur"):
        return "$"
    if u in ("%", "pp") or u.startswith("percent") or u == "per cent":
        return "%"
    if u in ("x", "\u00d7", "times"):
        return "x"
    if u.startswith("day"):
        return "days"
    if u.startswith(("yr", "year", "-year")):
        return "years"
    return "scaled" if m.group("scale") else None


def _extract(text: str) -> list:
    out = []
    for m in NUM_RE.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if nxt.isalnum() or nxt == "_":
            continue  # "1st", "20b", "320x50": part of a word or code
        raw = m.group("num")
        digits = re.sub(_SEP, "", raw)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        scale = SCALES.get((m.group("scale") or "").lower(), 1)
        value = float(digits) * scale
        tol = 0.5 * 10 ** -decimals * scale + 1e-9 * max(1.0, value)
        out.append(Claim(m.group(0).strip(), m.start(), m.end(), value, tol, _unit_of(m),
                         decimals == 0 and scale == 1))
    return out


# Metric labels that may sit right next to a number ("CTR (2.16%)", "$3.07M spend").
LABELS = [
    ("conversion_rate", r"conversion[- \u2011]rates?|conv\.? rates?|cvr"),
    ("roas", r"roas|return on ad spend"),
    ("roi_pct", r"roi(?: ?%)?|return on investment"),
    ("cpa", r"cpa|cost per (?:acquisition|conversion)"),
    ("cpc", r"cpc|cost per click"),
    ("ctr", r"ctr|click[- \u2011]?through(?:[- \u2011]rates?)?"),
    ("revenue", r"revenues?|sales"),
    ("profit", r"profits?"),
    ("spend", r"ad spend|spend(?:ing)?|spent"),
    ("conversions", r"conversions"),
    ("clicks", r"clicks"),
    ("impressions", r"impressions"),
    ("campaigns", r"campaigns"),
    ("audience_age", r"age groups?|audience age|age brackets?|years? old|year-olds?"),
]
LABEL_METRICS = {"campaigns": {"campaign_count", "matched_campaign_count", "example_count"},
                 "audience_age": {"age_group"}}
# "campaigns" only counts AFTER a number ("1,107 campaigns"): before one it is rarely the label
# ("across all campaigns is 2.16%"). Adjacency never crosses a line break.
_LABEL_ALT = "|".join(f"(?P<{name}>{pattern})" for name, pattern in LABELS)
_LABEL_ALT_BEFORE = "|".join(f"(?P<{name}>{pattern})" for name, pattern in LABELS if name != "campaigns")
LABEL_BEFORE_RE = re.compile(
    r"\b(?:" + _LABEL_ALT_BEFORE + r")(?:'s)?\b(?:[ \t*:=(\[]|\b(?:of|at|is|was|were|are|to|stood at|reached|hit"
    r"|averag(?:e|ed|es|ing)|about|around|approximately|roughly|nearly|just|only)\b|\u2248|~)*$", re.IGNORECASE)
LABEL_AFTER_RE = re.compile(
    r"^[ \t)*\]]*(?:(?:in|of|total|average|overall)[ \t]+)*(?:" + _LABEL_ALT + r")\b", re.IGNORECASE)
PER_DOLLAR_BEFORE_RE = re.compile(r"\b(?:each|every|per)\s+(?:\$\s?1(?![\d.])|dollar)\b[^.;\n]*$", re.IGNORECASE)
AGGREGATE_CUE_RE = re.compile(r"\b(?:average|avg|mean|overall|typical(?:ly)?|median|as a group|in aggregate"
                              r"|aggregate|in total|combined)\b", re.IGNORECASE)
OVERALL_CUE_RE = re.compile(r"\b(?:overall|all campaigns|all [\d,]+ campaigns|entire|portfolio|whole|across all)\b",
                            re.IGNORECASE)
CLAUSE_BREAK_RE = re.compile(r"(?<!\d)[,;](?!\d)|[;\n]|[.!?](?=\s)|\b(?:while|whereas|but|compared (?:with|to)|versus"
                             r"|vs\.?|then|than)\b", re.IGNORECASE)
CAUSAL_RE = re.compile(r"\b(?:because|due to|caused by|causes|the reason (?:is|was|for)|results? in|resulted in"
                       r"|leads? to|led to|drives?|driven by|driving|explains why|is why|thanks to|attributable to"
                       r"|as a result)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(?:may|might|could|possibl\w*|perhaps|likely|suggests?|appears?|seems?|hypothes\w*"
                      r"|potential\w*|unclear|cannot|can't|not establish\w*|correlat\w*|consistent with|if|whether"
                      r"|one reason|would need)\b", re.IGNORECASE)
CAUSAL_CAVEAT = ("Note: this data shows what happened, not why. Any reasons given are possibilities "
                 "the data does not prove.")
WITHHELD_NOTE = ("The AI explanation was withheld because it contained figures that could not be verified "
                 "against this data. The figures above come directly from the MarketingIQ database; "
                 "they describe what happened, not why.")


def _label(text: str, c: Claim, entity_names: set):
    """The metric label written right next to a number (the closer side wins), or None. A word that
    is also an entity name in the data (the objective "Conversions") is not a label."""
    found = []
    window = max(0, c.start - 45)
    m = LABEL_BEFORE_RE.search(text[window:c.start])
    if m:
        name = next(n for n, _ in LABELS if n != "campaigns" and m.group(n))
        found.append((c.start - window - m.end(name), name, m.group(name)))
    m = LABEL_AFTER_RE.search(text[c.end:c.end + 30])
    if m:
        name = next(n for n, _ in LABELS if m.group(n))
        found.append((m.start(name) + 0.5, name, m.group(name)))  # a tie goes to the label before
    found = [f for f in found if f[2].lower() not in entity_names]
    return min(found)[1] if found else None


def _name_re(name: str) -> str:
    alias = f"|{re.escape(name[:-4])}" if name.lower().endswith(" ads") else ""  # "Google Ads" / "Google"
    return rf"(?<!\w)(?:{re.escape(name)}{alias})(?!\w)"


def _entity_patterns(entity_names: set) -> list:
    return [(n, re.compile(r"[ \t)*]*(?:for|on|in|at|from)\s+(?:the\s+)?" + _name_re(n), re.IGNORECASE),
             re.compile(_name_re(n), re.IGNORECASE)) for n in entity_names]


def _entity_for(after: str, before: str, patterns: list):
    """The one entity a number is written about: "$53.20 for Conversions" / "on TikTok" right after
    it, else the only entity named earlier in its clause ("LinkedIn ($6.55 CPC"). None if unclear."""
    for n, after_re, _ in patterns:
        if after_re.match(after):
            return n
    named = {n for n, _, name_re in patterns if name_re.search(before)}
    return next(iter(named)) if len(named) == 1 else None


class _Clauses:
    """Clause boundaries of one text, computed once (the validator asks for many positions)."""

    def __init__(self, text: str):
        self.breaks = [(m.start(), m.end()) for m in CLAUSE_BREAK_RE.finditer(text)]
        self.length = len(text)

    def at(self, pos: int) -> tuple:
        start, end = 0, self.length
        for s, e in self.breaks:
            if e <= pos:
                start = e
            elif s >= pos:
                end = s
                break
        return start, end


def _sentence(text: str, pos: int) -> tuple:
    starts = [0] + [m.end() for m in re.finditer(r"[.!?](?=\s)|\n", text[:pos])]
    m = re.compile(r"[.!?](?=\s|$)|\n").search(text, pos)
    return starts[-1], (m.end() if m else len(text))


def _months_in(s: str) -> set:
    found = set()
    for y, mo in ISO_MONTH_RE.findall(s):
        found.add((int(mo), y))
    for m in MONTH_RE.finditer(s):
        if m.group(1) == "May" and not m.group(2):
            continue  # "May" is usually the verb
        found.add((MONTH_NAMES.index(m.group(1)[:3].lower()) + 1, m.group(2)))
    return found


def _unit_ok(claim_unit, fact: Fact) -> bool:
    if fact.unit == "any":
        return True
    if claim_unit is None:
        return True
    if claim_unit == "scaled":
        return fact.unit in ("$", "count")
    return claim_unit == fact.unit


def _inline_markers(text: str) -> set:
    """Start positions of inline enumerators "1) ... 2) ... 3)": only a run that counts up from 1."""
    found, expected = set(), 1
    for m in INLINE_MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n == 1:
            expected = 1
        if n == expected:
            found.add(m.start(1))
            expected += 1
    return found


def _is_structural(text: str, c: Claim, facts: list, markers: set = frozenset()) -> bool:
    if not c.integer or c.unit is not None:
        return False
    if c.value <= 99 and text[c.end:c.end + 1] in (".", ")") and LIST_MARKER_RE.search(text[:c.start]):
        return True  # "1. " / "2) " list numbering
    if c.end - len(c.text) in markers:  # c.start may include a leading space
        return True  # "... reasons: 1) cheaper clicks 2) younger audiences"
    if c.value <= 10 and STRUCTURAL_NOUN_RE.match(text[c.end:c.end + 40]):
        return True  # "3 observations", "2 key reasons"
    if c.value <= 10 and STRUCTURAL_BEFORE_RE.search(text[max(0, c.start - 12):c.start]):
        return True  # "step 2", "option 3", "#1"
    if TOP_N_BEFORE_RE.search(text[max(0, c.start - 8):c.start]):
        counts = [f.value for f in facts if f.metric == "entity_count"]
        return bool(counts) and c.value <= max(counts)  # "top 3" within the supplied ranking
    if c.value <= 31 and MONTH_RE.search(text[max(0, c.start - 12):c.start].rstrip() + " ") \
            and re.search(r"[A-Za-z]\.?\s*$", text[:c.start]):
        return True  # "December 31"
    return False


# "A (5) is higher than B (10)": both numbers are supported, the relation is not. `than` forms may
# have words between ("lower returns (5.8x) than ..."); above/below stand alone.
COMPARE_RE = re.compile(r"\b(?P<word>higher|greater|lower|less)\b(?P<mid>(?:[^.;!?\n]|\.(?=\d)){0,60}?)\bthan\b"
                        r"|\b(?P<bound>above|below)\b", re.IGNORECASE)
NEGATED_RE = re.compile(r"\b(?:not|n't|no)\s+(?:\w+\s+)?$", re.IGNORECASE)
# Between two objects of one comparison ("above A (4.99%) and B (5.00%)"): a short gap with
# "and"/"or"/a comma and no clause words — "..., while ROAS is 11.2x" starts a new statement.
LIST_GAP_RE = re.compile(r"^[^\d$]{0,30}$")
LIST_JOIN_RE = re.compile(r"\band\b|\bor\b|,", re.IGNORECASE)
CLAUSE_WORD_RE = re.compile(r"\b(?:while|whereas|but|with|which|is|are|was|were|has|have|had|its|at|of)\b",
                            re.IGNORECASE)


def _claim_unit(c: Claim, cands: list):
    if c.unit not in (None, "scaled"):
        return c.unit
    units = {f.unit for f in cands if f.unit not in ("any", "name")}
    return units.pop() if len(units) == 1 else None


BOTH_RE = re.compile(r"\bboth\b", re.IGNORECASE)
ALL_OTHERS_RE = re.compile(r"\b(?:all|every|each)\s+(?:the\s+)?other\b", re.IGNORECASE)


def _entity_comparisons(region: str, subjects: list, word: str, up: bool, entity_values: dict) -> list:
    """The other side is named, not written as a number: "3.44% ... above Google Ads and LinkedIn",
    "profit ... above all other platforms". Each subject is compared, for its own metric, with
    each named entity's value in the supplied data. Unresolvable subjects are left alone."""
    issues = []
    for c, cands in subjects:
        metrics = {f.metric for f in cands if f.scope in ("entity", "aggregate")}
        owners = {f.entity for f in cands if f.scope == "entity"}
        if len(metrics) != 1 or len(owners) > 1:
            continue
        metric, owner = metrics.pop(), next(iter(owners), None)
        values = entity_values.get(metric, {})
        if ALL_OTHERS_RE.search(region):
            targets = [n for n in values if n != owner]
        else:
            targets = [n for n in values if n != owner and re.search(_name_re(n), region, re.IGNORECASE)]
        for name in targets:
            if (up and not c.value > values[name]) or (not up and not c.value < values[name]):
                issues.append({"category": "wrong_comparison", "claim": f"{c.text} {word} {name}"})
    return issues


def _comparison_issues(scan: str, resolved: list, facts: list = ()) -> list:
    """Deterministic check of "X higher/lower/greater/less than Y" and "X above/below Y" when both
    sides are supported numbers in one sentence, compared as written (unit-aware). The subject is
    the number between the word and "than", else the latest number before it that isn't itself the
    object of an earlier comparison in the sentence. A threshold ("spend above $10,000") is not a
    comparison. Unresolvable units in an explicit "than" comparison are flagged, not guessed."""
    issues = []
    entity_values = {}  # metric -> {entity: value}, from the supplied comparison rows
    for f in facts:
        if f.scope == "entity" and f.entity and f.metric:
            entity_values.setdefault(f.metric, {})[f.entity] = f.value
    sentences = {}
    for c, cands, _ in resolved:
        sentences.setdefault(_sentence(scan, c.start), []).append((c, cands))
    for (ss, se), items in sentences.items():
        matches = list(COMPARE_RE.finditer(scan, ss, se))
        consumed = set()
        for i, m in enumerate(matches):
            if NEGATED_RE.search(scan[max(ss, m.start() - 15):m.start()]):
                continue
            word = (m.group("word") or m.group("bound")).lower()
            up = word in ("higher", "greater", "above")
            limit = matches[i + 1].start() if i + 1 < len(matches) else se
            if m.group("word"):
                mid = [x for x in items if m.start("mid") <= x[0].start < m.end("mid")]
                subject = mid[0] if mid else None
            else:
                subject = None
            if subject is None:
                before = [x for x in items if x[0].start < m.start() and id(x[0]) not in consumed]
                subject = before[-1] if before else None
            after = [x for x in items if m.end() <= x[0].start < limit]
            if subject is None:
                continue
            if not after:  # "above Google Ads and LinkedIn", "both above all other platforms"
                subjects = [subject]
                if BOTH_RE.search(scan[max(ss, m.start() - 20):m.start()]):
                    subjects = [x for x in items if x[0].start < m.start()][-2:]
                issues += _entity_comparisons(scan[m.end():limit], subjects, word, up, entity_values)
                continue
            objects = [after[0]]
            for x in after[1:]:  # "above Google Ads (4.99%) and LinkedIn (5.00%)"
                gap = scan[objects[-1][0].end:x[0].start]
                if not (LIST_GAP_RE.match(gap) and LIST_JOIN_RE.search(gap)) or CLAUSE_WORD_RE.search(gap):
                    break
                objects.append(x)
            for obj, ocands in objects:
                consumed.add(id(obj))
                if m.group("bound") and any(f.scope in ("condition", "question") for f in ocands):
                    continue  # a threshold, not a comparison
                su, ou = _claim_unit(*subject), _claim_unit(obj, ocands)
                if m.group("bound") and (not su or su != ou):
                    continue  # "5 of the 10 campaigns ... below 1.0": a bound, not a like-for-like comparison
                if su and ou and su != ou:
                    issues.append({"category": "comparison_unresolved", "claim": f"{subject[0].text} {word} {obj.text}"})
                    continue
                a, b = subject[0].value, obj.value
                if (up and not a > b) or (not up and not a < b):
                    issues.append({"category": "wrong_comparison", "claim": f"{subject[0].text} {word} {obj.text}"})
    return issues


# "TikTok has the highest profit": checked against every supplied row of the entity's comparison
# section. "most"/"least" need "the" ("most campaigns" can mean a majority); best/worst only where
# the metric's good direction is fixed. "second highest" / "one of the highest" are not claims of first.
SUPERLATIVE_RE = re.compile(r"\b(?:(?P<the>the)\s+)?(?P<word>highest|lowest|most|least|best|worst)\s+"
                            r"(?:(?:overall|total|average)\s+)?(?:" + _LABEL_ALT + r")\b", re.IGNORECASE)
NOT_FIRST_RE = re.compile(r"\b(?:not|n't|one of|among|second|third|fourth|fifth|next|\d+(?:st|nd|rd|th))"
                          r"[\s-]*(?:the\s+)?$", re.IGNORECASE)
# "the highest CPC among the top five platforms": a restricted set, not the supplied rows.
SCOPED_RE = re.compile(r"\b(?:among|excluding|except|apart from|aside from|other than|besides|outside|within)\b",
                       re.IGNORECASE)
HIGHER_IS_BETTER = {"revenue", "profit", "roas", "roi_pct", "ctr", "conversion_rate", "conversions", "clicks",
                    "impressions"}
LOWER_IS_BETTER = {"cpa", "cpc"}


def _superlative_issues(scan: str, context: dict, clauses: "_Clauses") -> list:
    """A superlative about one named entity is false when another supplied row of the same
    comparison beats it. Truncated sections only ever show rows, so a shown row that beats the
    entity is proof; an unclear entity, metric or direction is left alone."""
    issues = []
    groups = [{e["name"]: e for e in s.get("entities", []) if e.get("name")}
              for s in context.get("sections", []) if s.get("type") == "comparison"]
    for m in SUPERLATIVE_RE.finditer(scan):
        word = m.group("word").lower()
        if word in ("most", "least") and not m.group("the"):
            continue
        if NOT_FIRST_RE.search(scan[max(0, m.start() - 20):m.start()]):
            continue
        if SCOPED_RE.search(scan[m.end():_sentence(scan, m.start())[1]]):
            continue
        label = next(n for n, _ in LABELS if m.group(n))
        metric = {"campaigns": "campaign_count"}.get(label, label)
        if metric not in METRICS:
            continue
        if word in ("highest", "most"):
            up = True
        elif word in ("lowest", "least"):
            up = False
        elif metric in HIGHER_IS_BETTER | LOWER_IS_BETTER:
            up = (word == "best") == (metric in HIGHER_IS_BETTER)
        else:
            continue  # "best spend": no fixed direction
        for rows in groups:
            values = {n: r[metric] for n, r in rows.items() if _is_num(r.get(metric))}
            if len(values) < 2:
                continue
            named = set()
            for span in (clauses.at(m.start()), _sentence(scan, m.start())):
                named = {n for n in values if re.search(_name_re(n), scan[span[0]:span[1]], re.IGNORECASE)}
                if named:
                    break
            if len(named) != 1:
                continue
            name = named.pop()
            best = (max if up else min)(values.values())
            if values[name] != best:
                issues.append({"category": "wrong_comparison", "claim": f"{name} {word} {m.group(0).split()[-1]}"})
    return issues


@dataclass
class ValidationReport:
    passed: bool
    issues: list = field(default_factory=list)      # [{"category", "claim", ...}]
    causal_flags: int = 0                           # unhedged causal sentences
    numbers_checked: int = 0

    def categories(self) -> list:
        return sorted({i["category"] for i in self.issues})


def validate_explanation(text: str, context: dict, question: str = "", facts: list = None) -> ValidationReport:
    """Every number written in `text` must be supported by `context` with a compatible unit, and,
    where a metric label sits right next to it, by a value of that metric."""
    facts = facts if facts is not None else collect_facts(context, question)
    years = _years(context, question)
    issues, checked = [], 0
    # Validation-only copy: Unicode hyphens (as in "2024‑01") become "-" (same length, so positions
    # still match `text`). ISO dates/months are checked as dates, then blanked so "2025-01" isn't
    # read as 2025 and 1.
    scan = text.translate(HYPHENS)
    for m in ISO_MONTH_RE.finditer(scan):
        if m.group(1) not in years:
            issues.append({"category": "unsupported_date", "claim": m.group(0)})
        scan = scan[:m.start()] + " " * (m.end() - m.start()) + scan[m.end():]
    subset_aggs = {f.metric for f in facts if f.scope == "subset"}
    entity_names = {f.entity for f in facts if f.scope == "entity" and f.entity}
    lower_names = {n.lower() for n in entity_names}
    patterns = _entity_patterns(entity_names)
    clauses = _Clauses(scan)
    claims = _extract(scan)
    markers = _inline_markers(scan)
    resolved = []  # (claim, candidates) for supported claims, for sentence-level checks
    for c in claims:
        if c.integer and c.unit is None and 1900 <= c.value <= 2100 and "," not in c.text:
            checked += 1
            if str(int(c.value)) not in years:
                issues.append({"category": "unsupported_date", "claim": c.text})
            continue
        if _is_structural(scan, c, facts, markers):
            continue
        if c.unit == "$" and c.value == 1 and re.search(r"\b(?:each|every|per)\s*$", scan[max(0, c.start - 8):c.start]):
            continue  # "for each $1 spent"
        checked += 1
        unit = c.unit
        if unit == "$" and (PER_DOLLAR_RE.match(scan[c.end:c.end + 45])
                            or PER_DOLLAR_BEFORE_RE.search(scan[clauses.at(c.start)[0]:c.start])):
            unit = "x"  # "$6.54 for every $1 spent" / "each dollar spent generates $6.54" restate a ratio
        if unit in ("x", "%") and DERIVED_AFTER_RE.match(scan[c.end:c.end + 20]):
            issues.append({"category": "derived_value", "claim": c.text})
            continue
        label = _label(scan, c, lower_names) if unit == c.unit else "roas"
        allowed = LABEL_METRICS.get(label, {label}) if label else None
        # Rounded or truncated at the precision written: "$10,001" or "$10,002" for 10,001.88.
        by_value = [f for f in facts if c.value - c.tol <= f.value < c.value + 2 * c.tol]
        if c.integer and c.unit is None:
            # A bare whole number supports only counts/labels exactly, or a labelled metric at rounding.
            by_value = [f for f in by_value if f.unit in ("count", "days", "name", "any")
                        or (allowed and f.metric in allowed)]
        by_unit = [f for f in by_value if _unit_ok(unit, f)]
        cands = [f for f in by_unit if not allowed or f.metric in allowed or f.unit == "any" or f.scope == "question"]
        if not by_value:
            issues.append({"category": "unsupported_value", "claim": c.text})
        elif not by_unit:
            issues.append({"category": "unit_mismatch", "claim": c.text, "claimed_unit": unit,
                           "data_units": sorted({f.unit for f in by_value})})
        elif not cands:
            issues.append({"category": "metric_label_mismatch", "claim": c.text, "label": label,
                           "data_metric": sorted({str(f.metric) for f in by_unit})[0]})
        else:
            resolved.append((c, cands, label))

    for c, cands, label in resolved:
        cs, ce = clauses.at(c.start)
        before = scan[cs:c.start]
        clause = scan[cs:ce]
        scopes = {f.scope for f in cands}
        # An individual example campaign's value presented as a group figure.
        if scopes == {"example"} and AGGREGATE_CUE_RE.search(before):
            issues.append({"category": "example_as_aggregate", "claim": c.text})
            continue
        # An overall figure attributed to the filtered subset, when the subset has its own figure.
        if label and label in subset_aggs and "subset" not in scopes and "question" not in scopes:
            ss, se = _sentence(scan, c.start)
            sentence_claims = [(x, xc) for x, xc, _ in resolved if ss <= x.start < se]
            subset_cue = any(f.scope == "condition" or f.metric == "matched_campaign_count"
                             for x, xc in sentence_claims for f in xc)
            has_subset_value = any("subset" in {f.scope for f in xc} and f.metric == label
                                   for x, xc in sentence_claims for f in xc)
            if subset_cue and not has_subset_value and not OVERALL_CUE_RE.search(before):
                issues.append({"category": "scope_mismatch", "claim": c.text, "label": label})
                continue
        # A value attached to the wrong entity or month.
        if scopes == {"entity"}:
            named = _entity_for(scan[c.end:ce], before, patterns)
            if named and not any(f.entity == named for f in cands):
                issues.append({"category": "entity_mismatch", "claim": c.text, "entity": named})
                continue
        if scopes == {"series"}:
            months = _months_in(clause)
            if len(months) == 1:
                mo, yr = next(iter(months))
                if not any(f.month and int(f.month[5:7]) == mo and (not yr or f.month[:4] == yr) for f in cands):
                    issues.append({"category": "month_mismatch", "claim": c.text})

    issues += _comparison_issues(scan, resolved, facts)
    issues += _superlative_issues(scan, context, clauses)

    causal = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if CAUSAL_RE.search(sentence) and not HEDGE_RE.search(sentence):
            causal += 1
    return ValidationReport(not issues, issues, causal, checked)


# ---------------------------------------------------------------------------
# 4. Deterministic answer when an explanation fails validation
# ---------------------------------------------------------------------------

SHORT = {"campaign_count": "campaigns", "roas": "ROAS", "roi_pct": "ROI", "cpa": "CPA", "cpc": "CPC", "ctr": "CTR",
         "conversion_rate": "conversion rate"}
DEFAULT_SUMMARY_METRICS = ["revenue", "spend", "roas", "cpa", "ctr", "conversion_rate"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]


def fmt(metric: str, value, unit: str = None) -> str:
    unit = unit or _unit(metric)
    if value is None:
        return "n/a"
    if unit == "$":
        return f"${value:,.2f}"
    if unit == "x":
        return f"{value:,.2f}x"
    if unit == "%":
        return f"{value:,.2f}%"
    if unit == "count":
        return f"{int(value):,}"
    return f"{value:,}" if _is_num(value) else str(value)


def _name(metric: str) -> str:
    return SHORT.get(metric, metric.replace("_", " "))


def _month_name(ym: str) -> str:
    try:
        return f"{MONTH_FULL[int(ym[5:7]) - 1]} {ym[:4]}"
    except (ValueError, IndexError, TypeError):
        return str(ym)


def _metric_list(metrics: dict, focus: list) -> str:
    keys = [m for m in dict.fromkeys([*focus, *DEFAULT_SUMMARY_METRICS]) if m in metrics and m in METRICS
            and m != "campaign_count"][:6]
    return "; ".join(f"{_name(k)} {fmt(k, metrics[k])}" for k in keys)


def _summary_lines(s: dict, focus: list) -> list:
    t = s.get("type")
    if t == "aggregate":
        m = s.get("metrics") or {}
        return [f"All campaigns in scope ({fmt('campaign_count', m.get('campaign_count'))} campaigns): "
                f"{_metric_list(m, focus)}."]
    if t == "comparison" and s.get("entities"):
        metric = s.get("ranked_by") or next((f for f in focus if f in s["entities"][0]), "roas")
        items = "; ".join(f"{e['name']} {fmt(metric, e.get(metric))}" for e in s["entities"][:10])
        order = f", {s['order']}" if s.get("order") else ""
        return [f"{_name(metric)} by {s['dimension'].replace('_', ' ')}{order}: {items}."]
    if t == "time_series" and s.get("points"):
        pts, metric = s["points"], s["metric"]
        hi, lo = max(pts, key=lambda p: p["value"]), min(pts, key=lambda p: p["value"])
        unit = s.get("unit")
        return [f"Monthly {_name(metric)} from {_month_name(pts[0]['month'])} to {_month_name(pts[-1]['month'])}: "
                f"highest {fmt(metric, hi['value'], unit)} in {_month_name(hi['month'])}; "
                f"lowest {fmt(metric, lo['value'], unit)} in {_month_name(lo['month'])}."]
    if t == "creative_age_buckets" and s.get("buckets"):
        items = "; ".join(f"{b['creative_age_days']} days {fmt('ctr', b['ctr'])}" for b in s["buckets"])
        return [f"CTR by creative age (age of the ad creative): {items}."]
    if t == "filtered_subset":
        crit = " and ".join(f"{_name(c['field'])} {c['operator']} {fmt(c['field'], c['value'])}"
                            for c in s.get("conditions", []))
        line = f"{fmt('campaign_count', s.get('matched_campaign_count'))} campaigns match {crit or 'the filter'}"
        if s.get("aggregates"):
            line += f"; as a group: {_metric_list(s['aggregates'], focus)}"
        return [line + "."]
    if t == "share":
        return [f"{s.get('name')} accounts for {fmt('share', s.get('share_pct'), '%')} of total {_name(s.get('metric', ''))}."]
    return []


def safe_summary(context: dict, note: str = WITHHELD_NOTE) -> str:
    """The supplied figures as plain text, then `note` (why no AI explanation is shown)."""
    focus = context.get("focus_metrics") or []
    lines = ["What the data shows:"]
    for s in context.get("sections", []):
        lines += [f"- {line}" for line in _summary_lines(s, focus)]
    if context.get("active_filters"):
        lines.append("- These figures reflect the current dashboard filters: "
                     + ", ".join(f"{k.replace('_', ' ')} = {v}" for k, v in context["active_filters"].items()) + ".")
    lines += ["", note]
    return "\n".join(lines)
