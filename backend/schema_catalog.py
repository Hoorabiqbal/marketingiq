"""
Schema catalog: what the AI Analyst may ask of the campaign dataset, built from the real columns
(data/tech_advertising_campaigns_dataset.csv, as loaded by data_tools) and the data layer.

It is the single allowlist for the analytics planner and the deterministic parser: every metric,
dimension, filter and chart role an analytical plan may name comes from here, with units,
aggregation rules and formulas. Its compact text form is what the Groq planner sees; rows never are.
"""
import re
from dataclasses import dataclass, field

import data_tools as dt
import query_router as qr


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str                    # "$", "x", "%", "count", "s" (seconds), "num" (plain number)
    aggregation: str             # "sum", "ratio" (of sums; formula), "avg" (mean over campaigns), "count"
    additive: bool               # totals of parts add up to the whole (pie / stacked allowed)
    aliases: tuple = ()
    formula: str = None
    campaign_column: str = None  # per-campaign column, for relationships (scatter)
    lower_is_better: bool = False


@dataclass(frozen=True)
class Dimension:
    name: str
    label: str
    column: str
    aliases: tuple = ()
    filterable: bool = False     # can narrow data by one value (dashboard filter key exists)
    values: tuple = field(default=())


M = Metric
METRICS = {m.name: m for m in [
    M("revenue", "Revenue", "$", "sum", True, ("revenues", "sales", "income", "turnover", "total revenue"),
      campaign_column="revenue"),
    M("spend", "Spend", "$", "sum", True, ("ad spend", "spending", "spent", "cost", "costs", "budget spent",
                                           "total spend"), campaign_column="ad_spend"),
    M("profit", "Profit", "$", "sum", True, ("profits", "net profit", "profitability"), campaign_column="profit"),
    M("conversions", "Conversions", "count", "sum", True, ("conversion count", "conversions"),
      campaign_column="conversions"),
    M("clicks", "Clicks", "count", "sum", True, ("click count",), campaign_column="clicks"),
    M("impressions", "Impressions", "count", "sum", True, ("impression count", "views"),
      campaign_column="impressions"),
    M("campaign_count", "Campaigns", "count", "count", True, ("number of campaigns", "campaign count")),
    M("roas", "ROAS", "x", "ratio", False, ("return on ad spend",), "revenue / spend", "ROAS"),
    M("roi_pct", "ROI", "%", "ratio", False, ("roi", "return on investment"), "(revenue - spend) / spend * 100"),
    M("cpa", "CPA", "$", "ratio", False, ("cost per acquisition", "cost per conversion"), "spend / conversions",
      "CPA", True),
    M("cpc", "CPC", "$", "ratio", False, ("cost per click",), "spend / clicks", "CPC", True),
    M("ctr", "CTR", "%", "ratio", False, ("click-through rate", "click through rate"), "clicks / impressions * 100",
      "CTR"),
    M("conversion_rate", "Conversion Rate", "%", "ratio", False, ("conversion rate", "conv rate", "cvr"),
      "conversions / clicks * 100", "conversion_rate"),
    M("bounce_rate", "Bounce Rate", "%", "avg", False, ("bounce rate", "bounces"), campaign_column="bounce_rate",
      lower_is_better=True),
    M("avg_session_duration", "Avg Session Duration", "s", "avg", False,
      ("session duration", "session length", "time on site", "time on page"),
      campaign_column="avg_session_duration_seconds"),
    M("pages_per_session", "Pages per Session", "num", "avg", False, ("pages per session", "page depth"),
      campaign_column="pages_per_session"),
    M("quality_score", "Quality Score", "num", "avg", False, ("quality score", "ad quality"),
      campaign_column="quality_score"),
]}

_DIM_ALIASES = {
    "platform": ("platforms", "channel", "channels", "network", "networks"),
    "objective": ("objectives", "campaign objective", "goal", "goals"),
    "device": ("devices", "device type"),
    "age_group": ("age", "ages", "age group", "age groups", "age bracket"),
    "gender": ("genders",),
    "vertical": ("verticals", "industry", "industries", "industry vertical", "industry verticals"),
    "budget_tier": ("budget tier", "budget tiers", "budget level", "budget levels"),
    "creative_format": ("creative format", "creative formats", "format", "formats", "ad format"),
    "creative_emotion": ("emotion", "emotions", "creative emotion", "tone"),
    "placement": ("placements", "ad placement"),
    "income_bracket": ("income", "income bracket", "income brackets", "income level"),
    "audience_interest": ("interest", "interests", "audience interest", "audience interests", "audience"),
    "operating_system": ("operating system", "operating systems", "os"),
    "retargeting": ("retargeting", "retargeted"),
    "day_of_week": ("day of week", "day of the week", "weekday", "weekdays"),
    "quarter": ("quarter", "quarters"),
    "creative_size": ("creative size", "creative sizes", "ad size", "banner size"),
    "ad_copy_length": ("ad copy length", "copy length", "ad copy"),
    "call_to_action": ("call to action", "call-to-action", "cta"),
    "purchase_intent": ("purchase intent", "intent score", "buying intent"),
}

TEMPORAL = {"month": "calendar month of start_date (YYYY-MM)", "year": "calendar year of start_date",
            "quarter": "quarter number of start_date (1-4, a dimension)",
            "day_of_week": "weekday of start_date (a dimension)"}
UNSUPPORTED = ("country", "region", "city", "location", "product", "sku", "customer lifetime value", "churn",
               "customer name", "email")

_catalog = None


def dimensions() -> dict:
    """Dimensions with their real values, read once from the data layer."""
    global _catalog
    if _catalog is None:
        repo = dt.get_repository()
        columns = dt.all_dimension_columns()
        profile = repo.profile(list(columns.values()), "start_date")
        out = {}
        for name, col in columns.items():
            values = sorted(str(v) for v in profile["distinct"][col])
            label = name.replace("_", " ").title()
            out[name] = Dimension(name, label, col, _DIM_ALIASES.get(name, ()), name in qr.ENTITY_FILTER_KEYS,
                                  tuple(values))
        _catalog = out
    return _catalog


def unit(metric: str) -> str:
    return METRICS[metric].unit


def label(metric: str) -> str:
    return METRICS[metric].label


def lower_label(metric: str) -> str:
    m = METRICS[metric]
    return m.label if m.label.isupper() or m.name in ("roi_pct",) else m.label.lower()


def _alias_re(names: dict) -> list:
    """[(canonical, compiled regex)] longest aliases first, so "conversion rate" beats "conversions"."""
    pairs = []
    for canonical, aliases in names.items():
        for a in {canonical.replace("_", " "), *aliases}:
            pairs.append((len(a), canonical, a))
    pairs.sort(reverse=True)
    return [(c, re.compile(r"(?<![\w-])" + re.escape(a).replace(r"\ ", r"[\s-]+") + r"(?![\w-])", re.IGNORECASE))
            for _, c, a in pairs]


METRIC_ALIAS_RES = _alias_re({n: m.aliases for n, m in METRICS.items()})
DIMENSION_ALIAS_RES = _alias_re(_DIM_ALIASES)


def find_metrics(text: str) -> list:
    """Catalog metrics named in `text`, in order of appearance (compound names win)."""
    found, scratch = [], text
    for name, pattern in METRIC_ALIAS_RES:
        for m in pattern.finditer(scratch):
            found.append((m.start(), name))
        scratch = pattern.sub(lambda m: " " * len(m.group(0)), scratch)
    return list(dict.fromkeys(n for _, n in sorted(found)))


def find_dimensions(text: str) -> list:
    found, scratch = [], text
    for name, pattern in DIMENSION_ALIAS_RES:
        for m in pattern.finditer(scratch):
            found.append((m.start(), name))
        scratch = pattern.sub(lambda m: " " * len(m.group(0)), scratch)
    return list(dict.fromkeys(n for _, n in sorted(found)))


def prompt_text() -> str:
    """The compact catalog the Groq planner receives."""
    dims = dimensions()
    metric_lines = []
    for m in METRICS.values():
        how = {"sum": "summed", "count": "counted", "ratio": f"= {m.formula}", "avg": "average per campaign"}[m.aggregation]
        metric_lines.append(f"- {m.name} ({m.unit}, {how}; aliases: {', '.join(m.aliases[:4]) or '-'})")
    dim_lines = [f"- {d.name}{'' if d.filterable else ' (breakdown only)'}: {', '.join(d.values)}" for d in dims.values()]
    return ("Metrics:\n" + "\n".join(metric_lines) + "\nDimensions and their exact values:\n" + "\n".join(dim_lines)
            + "\nTime: month (YYYY-MM) and year of the campaign start date."
            + "\nNot in the dataset: " + ", ".join(UNSUPPORTED) + ".")
