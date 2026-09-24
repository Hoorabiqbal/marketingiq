"""
MarketingIQ Query Router — deterministic routing layer (no LLM call).

Sits between the FastAPI API layer and the existing analytics tools in
data_tools.py. It decides WHERE a question goes; it never computes a metric
itself. Every number in a routed result comes from an existing data_tools
function run against the real dataset.

Routes:
  DIRECT_DATABASE      the question is answered exactly by one existing tool
                       (e.g. "What is total revenue?" -> get_totals)
  LLM_REQUIRED         the question needs an explanation ("why", "explain", ...).
                       The router runs the relevant tools and builds a SMALL
                       structured `analysis` block. That block is the only thing
                       the LLM Adapter (llm_adapter.py, via `explainer`) receives —
                       never the CSV, never all 10,000 campaigns, never a DataFrame.
  NEEDS_CLARIFICATION  about the campaign data, but no deterministic plan fits
  UNSUPPORTED          not about the data, or asks for something the dataset
                       doesn't have (e.g. geography, products)

Classification is keyword/regex based on purpose: no API call is made just to
decide whether an LLM is needed. It is intentionally simple and extensible —
add a phrase to a pattern below rather than a new per-question branch.

Usage:
    route_query("Which platform has the highest ROAS?", filters={...})
    classify_query("...")  # the plan only, nothing executed
"""
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import data_tools as dt

logger = logging.getLogger("marketingiq.query_router")
if not logger.handlers:
    # uvicorn doesn't configure non-uvicorn loggers, so INFO lines would be dropped.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Route(str, Enum):
    DIRECT_DATABASE = "DIRECT_DATABASE"
    LLM_REQUIRED = "LLM_REQUIRED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNSUPPORTED = "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Errors — each carries an HTTP status and a user-safe message (no internals).
# ---------------------------------------------------------------------------

class QueryRouterError(Exception):
    status_code = 500
    public_message = "The query router hit an unexpected error. Please try again."

    def __init__(self, public_message: str = None):
        if public_message:
            self.public_message = public_message
        super().__init__(self.public_message)


class InvalidQueryError(QueryRouterError):
    status_code = 400
    public_message = "The query is invalid."


class DataUnavailableError(QueryRouterError):
    status_code = 503
    public_message = "The MarketingIQ dataset is not available right now. Please try again shortly."


class ToolUnavailableError(QueryRouterError):
    status_code = 503
    public_message = "The analytics tool needed for this question is not available right now."


class ToolExecutionError(QueryRouterError):
    status_code = 500
    public_message = "The analytics query failed. Please try again or rephrase the question."


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 1000
MAX_DIRECT_ROWS = 25          # cap on list results (top N, campaign lists)
MAX_LLM_TOOL_CALLS = 3        # tools run to build LLM context
MAX_LLM_LIST_ITEMS = 30       # per-list cap inside LLM context (fits a 25-month series)
MAX_LLM_CAMPAIGN_ROWS = 10    # individual campaigns ever passed toward an LLM
MAX_LLM_CONTEXT_BYTES = 16_000

# Tool parameter vocabularies (mirrors the enums in main.py's TOOL_DEFS).
RANK_METRICS = {"spend", "revenue", "profit", "conversions", "clicks", "impressions",
                "roas", "cpa", "cpc", "ctr", "conversion_rate", "roi_pct"}
FILTER_FIELDS = {"spend", "revenue", "profit", "roas", "cpa", "ctr", "conversion_rate",
                 "clicks", "conversions"}
TREND_METRICS = {"revenue", "spend", "profit", "conversions"}
SHARE_METRICS = {"spend", "revenue", "profit", "conversions", "clicks", "impressions"}
COST_METRICS = {"cpa", "cpc"}  # lower is better, so "best" means ascending
# Ratio metrics have no monthly tool; their monthly inputs explain a change instead.
TREND_INPUTS = {"roas": ["revenue", "spend"], "roi_pct": ["revenue", "spend"], "cpa": ["spend", "conversions"]}

# Tools that accept the dashboard's active filters — derived from the real signatures.
FILTERABLE_TOOLS = {name for name, fn in dt.TOOL_REGISTRY.items()
                    if "filters" in inspect.signature(fn).parameters}

# Dimension -> key understood by dt.apply_global_filters (for scoping a trend to an entity).
ENTITY_FILTER_KEYS = {
    "platform": "platform", "objective": "objective", "vertical": "vertical",
    "budget_tier": "budget", "device": "device", "age_group": "age", "gender": "gender",
    "creative_format": "creative", "creative_emotion": "emotion", "placement": "placement",
    "income_bracket": "income",
}

# ---------------------------------------------------------------------------
# Vocabulary (lower-cased input). Order of METRIC_PATTERNS matters: compound
# names are matched and blanked out first, so "return on ad spend" isn't also
# read as "spend" and "conversion rate" isn't also read as "conversions".
# ---------------------------------------------------------------------------

METRIC_PATTERNS = [
    ("conversion_rate", r"\bconversion rates?\b|\bconv\.? rates?\b|\bcvr\b"),
    ("roas", r"\broas\b|\breturn on ad spend\b"),
    ("roi_pct", r"\broi\b|\breturn on investment\b"),
    ("cpa", r"\bcpa\b|\bcost per (?:acquisition|conversion)\b"),
    ("cpc", r"\bcpc\b|\bcost per click\b"),
    ("ctr", r"\bctr\b|\bclick[- ]through(?: rates?)?\b"),
    ("spend", r"\bad spend\b|\bspend(?:ing)?\b|\bspent\b|\bcosts?\b"),
    ("revenue", r"\brevenues?\b|\bsales\b"),
    ("profit", r"\bprofits?\b|\bprofitability\b"),
    ("conversions", r"\bconversions?\b"),
    ("clicks", r"\bclicks?\b"),
    ("impressions", r"\bimpressions?\b"),
]
METRIC_PATTERNS = [(m, re.compile(p)) for m, p in METRIC_PATTERNS]

DIMENSION_PATTERNS = {
    "platform": r"\bplatforms?\b|\bchannels?\b|\bnetworks?\b",
    "objective": r"\bobjectives?\b|\bgoals?\b",
    "device": r"\bdevices?\b",
    "age_group": r"\bage(?: groups?| brackets?)?\b|\bages\b",
    "gender": r"\bgenders?\b",
    "vertical": r"\bverticals?\b|\bindustr(?:y|ies)\b",
    "budget_tier": r"\bbudget (?:tiers?|levels?)\b",
    "creative_format": r"\bcreative formats?\b|\bformats?\b",
    "creative_emotion": r"\bemotions?\b|\bemotional\b",
    "placement": r"\bplacements?\b",
    "income_bracket": r"\bincome(?: brackets?| levels?)?\b",
    "audience_interest": r"\binterests?\b",
    "operating_system": r"\boperating systems?\b|\bos\b",
    "retargeting": r"\bretargeting\b",
}
DIMENSION_PATTERNS = {d: re.compile(p) for d, p in DIMENSION_PATTERNS.items()}

_c = re.compile
EXPLANATION_RE = _c(
    r"\bwhy\b|\bexplain|\bexplanation|\breasons?\b|\bcaus(?:e|es|ed|ing)\b|\binterpret"
    r"|\bwhat (?:does|do|did) (?:this|these|that|those|it) mean\b|\bmean for\b|\binsights?\b"
    r"|\brecommend|\bsuggest|\badvi[cs]e\b|\bshould (?:we|i)\b|\bhow (?:can|could|do|should) (?:we|i)\b"
    r"|\bwhat(?:'s| is) driving\b|\bdrivers?\b"
)
CHANGE_RE = _c(r"\bdecl|\bdrop|\bdecreas|\bincreas|\bchang|\bgrow|\bgrew|\bfell\b|\bfall|\brose\b|\brising\b|\btrend|\bimprov|\bworsen")
COMPARATIVE_RE = _c(r"\bbetter\b|\bworse\b|\bdifferen|\bgap\b|\bvs\.?\b|\bversus\b|\boutperform|\bunderperform|\bcompare|\bcomparison\b|\bperform")
COUNT_RE = _c(r"\bhow many (?:campaigns|ads)\b|\bnumber of campaigns\b|\bcampaign count\b|\bcount (?:of )?campaigns\b|\b(?:most|fewest|least) campaigns\b")
TREND_RE = _c(r"\bmonthly\b|\bby month\b|\bper month\b|\beach month\b|\bover time\b|\btrends?\b|\btime series\b|\bmonth[- ](?:over|by)[- ]month\b")
MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
               "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_RE = _c(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(20\d{2})\b")
ISO_MONTH_RE = _c(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
YEAR_RE = _c(r"\b(?:in|during|for|of)\s+(20\d{2})\b")
SUPERLATIVE_RE = _c(r"\b(highest|lowest|best|worst|top|bottom|most|least|largest|smallest|biggest|leading|strongest|weakest|maximum|minimum)\b")
ASCENDING_WORDS = {"lowest", "worst", "bottom", "least", "smallest", "weakest", "minimum"}
QUALITY_WORDS = {"best", "worst", "strongest", "weakest", "leading"}
TOP_N_RE = _c(r"\b(?:top|bottom|first)\s+(\d{1,3})\b")
RANK_CUE_RE = _c(r"\brank|\bcompar|\bbreak ?down\b|\bby\b|\bper\b|\beach\b|\bacross\b|\bsplit\b")
SHARE_RE = _c(r"\bshare\b|\bpercent(?:age)?\b|%\s*of\b|\bproportion\b")
STATS_RE = _c(r"\bdistribution\b|\bpercentiles?\b|\bmedian\b|\bstatistics\b|\bstats\b|\bspread\b|\brange of\b")
FATIGUE_RE = _c(r"\bcreative fatigue\b|\bfatigue\b|\bcreative age\b")
FIELDS_RE = _c(r"\bwhat (?:data|fields|dimensions|metrics)\b|\bavailable (?:fields|dimensions|metrics|data)\b|\blist (?:the )?(?:fields|dimensions|metrics)\b|\bdate range\b")
TOTALS_RE = _c(r"\btotal\b|\boverall\b|\bsummary\b|\bsummari[sz]e\b|\bkpis?\b|\bentire\b|\ball campaigns\b")
CAMPAIGN_RE = _c(r"\bcampaigns?\b")
THRESHOLD_RE = _c(
    r"(greater than or equal to|less than or equal to|greater than|more than|higher than|less than|lower than"
    r"|at least|at most|above|over|exceeding|below|under|>=|<=|>|<)\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:%|x\b|k\b)?"
)
COMPARATORS = {
    "greater than or equal to": ">=", "at least": ">=", ">=": ">=",
    "less than or equal to": "<=", "at most": "<=", "<=": "<=",
    "greater than": ">", "more than": ">", "higher than": ">", "above": ">", "over": ">", "exceeding": ">", ">": ">",
    "less than": "<", "lower than": "<", "below": "<", "under": "<", "<": "<",
}
# Concepts the dataset does not contain — answered honestly as UNSUPPORTED rather
# than silently routed to an unrelated tool (e.g. "revenue by country").
UNAVAILABLE_RE = _c(r"\bcountr(?:y|ies)\b|\bregions?\b|\bgeograph|\bcit(?:y|ies)\b|\blocations?\b|\bproducts?\b|\bsku\b"
                    r"|\blifetime value\b|\bltv\b|\bchurn\b")
# Words a direct plan can't honour: qualitative judgements ("high spend", "underperforming"),
# change over time ("is revenue growing") and relative dates ("last month" — the router has no
# notion of "now"). Without this, e.g. "campaigns with high spend but low revenue" would be
# answered with the all-time totals. Numeric thresholds ("ROAS above 8") are removed first.
UNCERTAIN_RE = _c(
    r"\b(?:high|low|underperform\w*|overperform\w*|poor\w*|weak|strong|good|bad|efficient|inefficient"
    r"|wast\w*|expensive|cheap|profitable|unprofitable|better|worse|grow\w*|grew|declin\w*|drop\w*"
    r"|decreas\w*|increas\w*|fell|fall\w*|rose|rising|improv\w*|worsen\w*|chang\w*)\b"
    r"|\b(?:last|this|previous|past|next|current)\s+(?:month|year|quarter|week)s?\b"
    r"|\byesterday\b|\btoday\b|\brecent\w*|\blatest\b|\bytd\b|\byear[- ]to[- ]date\b|\bso far\b"
)
DATA_VOCAB_RE = _c(r"\bcampaigns?\b|\bads?\b|\badvertis|\bmarketing\b|\bperform|\btrends?\b|\bdata(?:set)?\b"
                   r"|\bresults?\b|\bnumbers\b|\bkpis?\b|\bbudget|\baudiences?\b|\bcreatives?\b")

# Dimension values that collide with ordinary words or metric names; not matched as entities.
AMBIGUOUS_ENTITY_VALUES = {"all", "high", "low", "medium", "other", "conversions", "text",
                           "search", "true", "false"}

SUPPORTED_HINT = ("Try asking about revenue, spend, profit, ROAS, ROI, CPA, CPC, CTR, conversions, "
                  "clicks or impressions — overall, by platform/objective/device/audience/creative, "
                  "by month, or for campaigns above/below a threshold.")


# ---------------------------------------------------------------------------
# Entity index — built once from the existing list_available_fields tool, so the router
# never touches the underlying DataFrame (keeps it independent of the data engine).
# ---------------------------------------------------------------------------

_entity_index = None


def _get_entity_index() -> list:
    global _entity_index
    if _entity_index is not None:
        return _entity_index
    try:
        fields = dt.list_available_fields()
    except RuntimeError:
        raise DataUnavailableError()
    entries = []
    for dim, values in fields["dimensions"].items():
        if dim == "retargeting" or not isinstance(values, list):  # >20 values come back as a summary string
            continue
        for v in values:
            value = str(v)
            if value.lower() in AMBIGUOUS_ENTITY_VALUES:
                continue
            aliases = [value.lower()]
            if value.lower().endswith(" ads"):
                aliases.append(value.lower()[:-4])  # "google ads" -> also "google"
            pattern = "|".join(re.escape(a) for a in aliases)
            entries.append((dim, value, re.compile(rf"(?<![\w$<>]){pattern}(?![\w])")))
    _entity_index = entries
    return entries


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool: str
    tool_input: dict = field(default_factory=dict)
    period: str = None           # "2025-01" or "2025": narrows a trend_over_time series
    extra_filters: dict = None   # entity scoping merged into the dashboard filters

    def describe(self) -> dict:
        out = {"tool": self.tool, "tool_input": self.tool_input}
        if self.period:
            out["period"] = self.period
        if self.extra_filters:
            out["scope"] = self.extra_filters
        return out


@dataclass
class Plan:
    route: Route
    calls: list = field(default_factory=list)
    message: str = None
    focus_metrics: list = field(default_factory=list)
    reason: str = None  # why a question was not planned: unavailable_data / off_topic / uncertain / unresolved


@dataclass
class _Features:
    text: str
    metrics: list          # [(metric, start)] in order of appearance
    dimensions: list       # explicitly named dimensions, in order of appearance
    entities: list         # [(dimension, value)]
    thresholds: list       # [(field, operator, value)]
    period: str


def _parse(query: str, entity_index: list) -> _Features:
    text = " ".join(query.lower().split())

    metrics, scratch = [], text
    for metric, pattern in METRIC_PATTERNS:
        for m in pattern.finditer(scratch):
            metrics.append((metric, m.start()))
        scratch = pattern.sub(lambda m: " " * len(m.group(0)), scratch)
    metrics.sort(key=lambda x: x[1])

    dim_hits = sorted((m.start(), d) for d, p in DIMENSION_PATTERNS.items() for m in [p.search(text)] if m)
    dimensions = [d for _, d in dim_hits]

    entities = []
    for dim, value, pattern in entity_index:
        m = pattern.search(text)
        if m:
            entities.append((m.start(), dim, value))
    entities = [(d, v) for _, d, v in sorted(entities)]

    thresholds = []
    for m in THRESHOLD_RE.finditer(text):
        # Pair each comparator with the nearest filterable metric named before it.
        preceding = [mt for mt, pos in metrics if pos < m.start() and mt in FILTER_FIELDS]
        if preceding:
            value = float(m.group(2).replace(",", ""))
            if m.group(0).rstrip().endswith("k"):
                value *= 1000
            thresholds.append((preceding[-1], COMPARATORS[m.group(1)], value))

    period = None
    if m := MONTH_RE.search(text):
        period = f"{m.group(2)}-{MONTH_NAMES[m.group(1)]:02d}"
    elif m := ISO_MONTH_RE.search(text):
        period = f"{m.group(1)}-{m.group(2)}"
    elif m := YEAR_RE.search(text):
        period = m.group(1)

    return _Features(text, metrics, dimensions, entities, thresholds, period)


def _metric_names(f: _Features) -> list:
    seen = []
    for m, _ in f.metrics:
        if m not in seen:
            seen.append(m)
    return seen


def _entity_scope(f: _Features) -> dict:
    """Named entities (e.g. "TikTok") as dashboard filters, to scope a trend or campaign list."""
    return {ENTITY_FILTER_KEYS[d]: v for d, v in f.entities if d in ENTITY_FILTER_KEYS}


def _rank_order(text: str, metric: str) -> str:
    words = set(SUPERLATIVE_RE.findall(text))
    ascending = bool(words & ASCENDING_WORDS)
    if metric in COST_METRICS and words & QUALITY_WORDS:
        ascending = not ascending  # "best CPA" = lowest CPA
    return "asc" if ascending else "desc"


def _top_n(text: str, default: int) -> int:
    m = TOP_N_RE.search(text)
    return min(int(m.group(1)), MAX_DIRECT_ROWS) if m else default


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _plan_direct(f: _Features):
    """Returns (calls, clarification_message). Exactly one of them is set."""
    text, metrics = f.text, _metric_names(f)
    metric = metrics[0] if metrics else None
    superlative = bool(SUPERLATIVE_RE.search(text))
    dimension = f.dimensions[0] if f.dimensions else None

    if FIELDS_RE.search(text):
        return [ToolCall("list_available_fields")], None

    if COUNT_RE.search(text):
        if dimension:
            order = "asc" if re.search(r"\b(?:fewest|least)\b", text) else "desc"
            return [ToolCall("rank_dimension", {"dimension": dimension, "metric": "campaign_count",
                                                "order": order, "limit": _top_n(text, 1 if superlative else 10)})], None
        return [ToolCall("get_totals")], None

    if FATIGUE_RE.search(text):
        return [ToolCall("get_creative_fatigue")], None

    if STATS_RE.search(text):
        fields = [m for m in metrics if m in FILTER_FIELDS]
        if fields:
            return [ToolCall("get_numeric_field_stats", {"fields": fields})], None
        return None, "Which metric's distribution do you want? " + SUPPORTED_HINT

    if f.thresholds:
        field_, _, _ = f.thresholds[0]
        conditions = [{"field": fl, "operator": op, "value": v} for fl, op, v in f.thresholds]
        order = "asc" if f.thresholds[0][1].startswith("<") else "desc"
        scope = _entity_scope(f)
        return [ToolCall("filter_campaigns", {"conditions": conditions, "sort_by": field_,
                                              "order": order, "limit": _top_n(text, 10)},
                         extra_filters=scope or None)], None

    if f.period or TREND_RE.search(text):
        trend_metrics = [m for m in metrics if m in TREND_METRICS]
        if metrics and not trend_metrics:
            return None, ("Monthly figures are available for revenue, spend, profit and conversions, "
                          f"not {metric.replace('_pct', '').replace('_', ' ')}.")
        scope = _entity_scope(f)
        return [ToolCall("trend_over_time", {"metric": trend_metrics[0] if trend_metrics else "revenue"},
                         period=f.period, extra_filters=scope or None)], None

    if SHARE_RE.search(text) and f.entities:
        share_metric = next((m for m in metrics if m in SHARE_METRICS), "spend")
        dim, name = f.entities[0]
        return [ToolCall("percentage_share", {"dimension": dim, "name": name, "metric": share_metric})], None

    same_dim = [e for e in f.entities if f.entities and e[0] == f.entities[0][0]]
    if len(same_dim) >= 2:
        return [ToolCall("compare_entities", {"dimension": same_dim[0][0], "names": [v for _, v in same_dim]})], None

    if dimension and (metric or superlative or RANK_CUE_RE.search(text)):
        rank_metric = metric if metric in RANK_METRICS else "roas"
        return [ToolCall("rank_dimension", {"dimension": dimension, "metric": rank_metric,
                                            "order": _rank_order(text, rank_metric),
                                            "limit": _top_n(text, 1 if superlative else 10)})], None

    if f.entities:
        dim, name = f.entities[0]
        return [ToolCall("get_entity_metrics", {"dimension": dim, "name": name})], None

    if CAMPAIGN_RE.search(text) and (superlative or TOP_N_RE.search(text)) and metric in FILTER_FIELDS:
        return [ToolCall("filter_campaigns", {"conditions": [], "sort_by": metric,
                                              "order": _rank_order(text, metric), "limit": _top_n(text, 10)})], None

    if metric or TOTALS_RE.search(text):
        return [ToolCall("get_totals")], None

    return None, None


def _plan_llm_context(f: _Features) -> list:
    """Chooses a few small, relevant tool results to ground a future LLM explanation."""
    text, metrics = f.text, _metric_names(f)
    metric = metrics[0] if metrics else "roas"
    calls = []

    dimension = (f.dimensions[0] if f.dimensions else
                 f.entities[0][0] if f.entities else
                 "platform" if COMPARATIVE_RE.search(text) else None)
    if dimension:
        # All values of one dimension (a handful of rows) gives peers for comparison.
        calls.append(ToolCall("rank_dimension", {"dimension": dimension,
                                                 "metric": metric if metric in RANK_METRICS else "roas",
                                                 "order": "desc", "limit": 10}))

    if CHANGE_RE.search(text) or TREND_RE.search(text) or f.period:
        trend_metric = next((m for m in metrics if m in TREND_METRICS), None)
        if trend_metric:
            trend_metrics = [trend_metric]
        else:
            trend_metrics = TREND_INPUTS.get(metric, []) if metrics else ["revenue"]
        for m in trend_metrics:
            calls.append(ToolCall("trend_over_time", {"metric": m}, extra_filters=_entity_scope(f) or None))

    if metric == "ctr" or FATIGUE_RE.search(text):
        calls.append(ToolCall("get_creative_fatigue"))

    if f.thresholds:
        conditions = [{"field": fl, "operator": op, "value": v} for fl, op, v in f.thresholds]
        calls.append(ToolCall("filter_campaigns", {"conditions": conditions, "sort_by": f.thresholds[0][0],
                                                   "limit": MAX_LLM_CAMPAIGN_ROWS}))

    if len(calls) < MAX_LLM_TOOL_CALLS:
        calls.append(ToolCall("get_totals"))  # overall baseline, 13 numbers
    return calls[:MAX_LLM_TOOL_CALLS]


def _has_data_vocab(f: _Features) -> bool:
    return bool(f.metrics or f.dimensions or f.entities or DATA_VOCAB_RE.search(f.text))


def _validate(query, filters):
    if not isinstance(query, str):
        raise InvalidQueryError("The query must be a text string.")
    if not query.strip():
        raise InvalidQueryError("The query is empty. Ask a question about your campaign data.")
    if len(query) > MAX_QUERY_LENGTH:
        raise InvalidQueryError(f"The query is too long (max {MAX_QUERY_LENGTH} characters).")
    if filters is None:
        return {}
    if not isinstance(filters, dict) or not all(
            isinstance(k, str) and (v is None or isinstance(v, (str, bool))) for k, v in filters.items()):
        raise InvalidQueryError("Filters must be an object of text or true/false values.")
    return filters


def classify_query(query: str) -> Plan:
    """Deterministic classification + tool plan. Executes nothing (beyond the cached entity index)."""
    _validate(query, None)
    f = _parse(query, _get_entity_index())
    focus = _metric_names(f)

    if UNAVAILABLE_RE.search(f.text):
        return Plan(Route.UNSUPPORTED, reason="unavailable_data",
                    message="That information is not available in the current MarketingIQ dataset. " + SUPPORTED_HINT)
    if not _has_data_vocab(f):
        return Plan(Route.UNSUPPORTED, reason="off_topic",
                    message="That question isn't about the MarketingIQ campaign data. " + SUPPORTED_HINT)
    if EXPLANATION_RE.search(f.text):
        return Plan(Route.LLM_REQUIRED, calls=_plan_llm_context(f), focus_metrics=focus)

    if UNCERTAIN_RE.search(THRESHOLD_RE.sub(" ", f.text)):
        return Plan(Route.NEEDS_CLARIFICATION, reason="uncertain",
                    message="That question needs a judgement or time reference I can't map to an exact "
                            "figure. " + SUPPORTED_HINT)

    calls, clarification = _plan_direct(f)
    if calls:
        return Plan(Route.DIRECT_DATABASE, calls=calls, focus_metrics=focus)
    return Plan(Route.NEEDS_CLARIFICATION, reason="unresolved",
                message=clarification or "I couldn't match that to a specific metric or breakdown. " + SUPPORTED_HINT)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _execute(call: ToolCall, filters: dict) -> dict:
    fn = dt.TOOL_REGISTRY.get(call.tool)
    if fn is None:
        raise ToolUnavailableError()
    kwargs = dict(call.tool_input)
    if call.tool in FILTERABLE_TOOLS:
        kwargs["filters"] = {**filters, **(call.extra_filters or {})}
    try:
        result = fn(**kwargs)
    except RuntimeError as e:
        if "not loaded" in str(e).lower():
            raise DataUnavailableError()
        raise ToolExecutionError() from e
    except Exception as e:
        raise ToolExecutionError() from e
    if isinstance(result, dict) and "error" in result:
        raise ToolExecutionError() from ValueError(f"{call.tool}: {result['error']}")
    if call.period and call.tool == "trend_over_time":
        result = {**result, "period": call.period,
                  "series": [p for p in result["series"] if p["month"].startswith(call.period)]}
    return result


def _compact(value):
    """Caps every list so an LLM context can never grow with dataset size."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            limit = MAX_LLM_CAMPAIGN_ROWS if k == "campaigns" else MAX_LLM_LIST_ITEMS
            if isinstance(v, list) and len(v) > limit:
                out[k] = [_compact(x) for x in v[:limit]]
                out[f"{k}_truncated_from"] = len(v)
            else:
                out[k] = _compact(v)
        return out
    if isinstance(value, list):
        return [_compact(x) for x in value]
    return value


def route_query(query: str, filters: dict = None, explainer=None) -> dict:
    """Classify a question and run the matching existing analytics tool(s).

    explainer: optional object with .explain(question, analysis) -> dict (llm_adapter.LLMAdapter).
    Only LLM_REQUIRED questions use it; DIRECT_DATABASE never does. Without one, LLM_REQUIRED
    returns the analysis with explanation=None.

    Raises a QueryRouterError subclass (with .status_code / .public_message) on failure.
    """
    started = time.perf_counter()
    route, tools, llm_status = None, [], None
    try:
        filters = _validate(query, filters)
        plan = classify_query(query)
        route, tools = plan.route, [c.tool for c in plan.calls]
        base = {"route": plan.route.value, "query": query.strip(), "filters_applied": filters}

        if plan.route == Route.DIRECT_DATABASE:
            call = plan.calls[0]
            response = {**base, **call.describe(), "result": _execute(call, filters),
                        "focus_metrics": plan.focus_metrics, "llm_required": False}
        elif plan.route == Route.LLM_REQUIRED:
            analysis = [{**c.describe(), "result": _compact(_execute(c, filters))} for c in plan.calls]
            while len(analysis) > 1 and len(json.dumps(analysis, default=str)) > MAX_LLM_CONTEXT_BYTES:
                analysis.pop()
            response = {**base, "analysis": analysis, "focus_metrics": plan.focus_metrics,
                        "llm_required": True, "explanation": None,
                        "llm_context_bytes": len(json.dumps(analysis, default=str))}
            if explainer is None:
                response["llm"] = {"status": "not_configured", "error": "no_explainer",
                                   "message": "No LLM explainer is attached; `analysis` holds the data."}
            else:
                llm = _explain(explainer, query.strip(), {"filters_applied": filters,
                                                          "focus_metrics": plan.focus_metrics,
                                                          "results": analysis})
                response["explanation"] = llm.pop("text", None)
                response["llm"] = llm
            llm_status = response["llm"]["status"]
        else:
            response = {**base, "message": plan.message, "reason": plan.reason, "llm_required": False}

        elapsed = round((time.perf_counter() - started) * 1000, 2)
        response["elapsed_ms"] = elapsed
        _log("ok", query, route, tools, elapsed, llm_status=llm_status)
        return response
    except QueryRouterError as e:
        _log("error", query, route, tools, round((time.perf_counter() - started) * 1000, 2),
             error=type(e).__name__, cause=repr(e.__cause__) if e.__cause__ else None)
        raise
    except Exception:
        logger.exception("query_router unexpected error")
        _log("error", query, route, tools, round((time.perf_counter() - started) * 1000, 2), error="Unexpected")
        raise QueryRouterError()


def _explain(explainer, question: str, llm_input: dict) -> dict:
    """The adapter already turns provider failures into results; this also guards against an
    adapter bug, so an LLM problem can never fail a request whose analysis succeeded."""
    try:
        return dict(explainer.explain(question, llm_input))
    except Exception:
        logger.exception("query_router explainer error")
        return {"status": "error", "error": "adapter_error",
                "message": "The AI explanation could not be generated. The analysis above is still valid."}


def _log(status, query, route, tools, elapsed_ms, **extra):
    record = {"event": "query_routed", "status": status,
              "query": query[:200] if isinstance(query, str) else f"<{type(query).__name__}>",
              "route": route.value if route else None, "tools": tools, "elapsed_ms": elapsed_ms}
    record.update({k: v for k, v in extra.items() if v is not None})
    (logger.info if status == "ok" else logger.warning)(json.dumps(record, default=str))
