"""
Exact campaign-ID lookup ("CAMP_00012", "camp 12"): one campaign's own row from DuckDB, answered
deterministically. No model call: the local planner was not trained on campaign IDs, and an exact
lookup needs no interpretation.

    campaign ID -> data tool get_campaign (+ its platform and all-campaign benchmarks for
                   explanations/advice) -> profile, outcome (dashboard rule), findings, recommendations
"""
import re

import analytics_executor as ax
import data_tools as dt
import query_planner as qp
import schema_catalog as catalog

CAMPAIGN_ID_RE = re.compile(r"\bcamp[_ -]?(\d{1,5})\b", re.IGNORECASE)
PROFILE = ("platform", "objective", "vertical", "device")
METRICS = ["spend", "revenue", "profit", "roas", "cpa", "cpc", "ctr", "conversion_rate", "conversions", "clicks",
           "impressions"]
COMPARED = ("roas", "cpa", "conversion_rate")       # efficiency measures compared with the benchmarks
ATTRIBUTE_LABELS = {"platform": "platform", "objective": "objective", "vertical": "industry vertical",
                    "budget_tier": "budget tier", "device": "device", "age_group": "target age group",
                    "gender": "target gender", "placement": "ad placement", "creative_format": "creative format",
                    "creative_emotion": "creative emotion", "income_bracket": "income bracket",
                    "audience_interest": "audience interest", "operating_system": "operating system",
                    "retargeting": "audience"}


def campaign_id(text: str):
    """The campaign ID a message names ("camp_2555", "CAMP_02555" -> "CAMP_02555"), or None."""
    m = CAMPAIGN_ID_RE.search(text or "")
    return f"CAMP_{int(m.group(1)):05d}" if m else None


def _fmt(metric, value):
    return qp._fmt(metric, value)


def _attribute(dim: str, value) -> str:
    if dim == "retargeting":
        return "retargeting" if value else "cold audience"
    return str(value)


def _month(ym: str) -> str:
    try:
        year, month = ym.split("-")
        return f"{qp.calendar.month_name[int(month)]} {year}"
    except (ValueError, IndexError):
        return ym


def _better(metric: str, a: float, b: float) -> bool:
    return a < b if metric in ("cpa", "cpc") else a > b


def answer(question: str, kind: str = "analysis"):
    """(lines, campaign id or None). kind: analysis / explanation / advice."""
    cid = campaign_id(question)
    res = ax.run("get_campaign", {}, campaign_id=cid, benchmarks=kind != "analysis")
    if not res.get("found"):
        return [f"{cid} isn't in the MarketingIQ dataset. Campaigns are identified by IDs in the form CAMP_00001."], None
    c = res["campaign"]
    text = question.lower()
    lines = []
    for dim in catalog.find_dimensions(text):
        if dim in c and dim in ATTRIBUTE_LABELS:
            lines.append(f"{cid} is on {c[dim]}." if dim == "platform" else
                         f"{cid}'s {ATTRIBUTE_LABELS[dim]} is {_attribute(dim, c[dim])}.")
    lines += [f"{cid}'s {catalog.lower_label(m)} is {_fmt(m, c[m])}." for m in catalog.find_metrics(text) if m in c]
    profile = ", ".join(_attribute(d, c[d]) for d in PROFILE if d in c)
    budget = f", {c['budget_tier']} budget" if "budget_tier" in c else ""
    lines.append(f"{cid} ({profile}{budget}, {_attribute('retargeting', c.get('retargeting'))}, started "
                 f"{_month(c['month'])}): " + ", ".join(f"{catalog.lower_label(m)} {_fmt(m, c[m])}" for m in METRICS
                                                         if m in c) + ".")
    if res["outcome"] == "unsuccessful":
        lines.append(f"It is unsuccessful: {dt.UNSUCCESSFUL_RULE}, the rule behind the dashboard's Unsuccessful "
                     "Campaigns table.")
    else:
        lines.append("It is profitable (profit above $0), so it is not an unsuccessful campaign.")
    if kind == "analysis":
        return lines, cid
    return _explained(cid, c, res, kind), cid


def _explained(cid, c, res, kind) -> list:
    """Findings (the campaign vs its platform and vs all campaigns) and, for advice, rule-based actions."""
    platform, overall = res["platform_benchmark"], res["all_campaigns"]
    findings = [f"{cid} is {'unsuccessful (' + dt.UNSUCCESSFUL_RULE + ')' if res['outcome'] == 'unsuccessful' else 'profitable'}"
                f", with profit of {_fmt('profit', c['profit'])} on spend of {_fmt('spend', c['spend'])}."]
    weaker = []
    for m in COMPARED:
        if c.get(m) is None or platform.get(m) is None:
            continue
        rel = "better than" if _better(m, c[m], platform[m]) else "weaker than" if _better(m, platform[m], c[m]) else "equal to"
        findings.append(f"Its {catalog.lower_label(m)} of {_fmt(m, c[m])} is {rel} the {platform['name']} average of "
                        f"{_fmt(m, platform[m])} (all campaigns: {_fmt(m, overall[m])}).")
        if rel == "weaker than":
            weaker.append(m)
    findings.append("This compares one campaign with averages; it shows how it performed, not why.")
    if kind != "advice":
        return ["The data shows:", *findings]
    actions = []
    if res["outcome"] == "unsuccessful":
        actions.append(f"Review or pause {cid} if its losses persist, and investigate its spend, revenue and "
                       "conversion efficiency.")
    if weaker:
        actions.append(f"Review its targeting and creative against stronger {platform['name']} campaigns, starting with "
                       + " and ".join(catalog.lower_label(m) for m in weaker) + ".")
    if not actions:
        actions.append(f"Consider testing a modest budget increase on {cid} while monitoring whether its efficiency "
                       "holds.")
    return ["Finding:", *findings, "Recommended action:", *actions]
