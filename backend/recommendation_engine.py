"""
Deterministic recommendations: findings_engine.Findings -> bounded, rule-based actions. No LLM.

Each rule is tied to a verified finding and its direction comes from the metric itself (higher is
better for ROAS / ROI / profit / rates; lower is better for CPA / CPC; spend is neutral and judged
by its ROAS). Recommendations name the entities and metrics from the findings and add no numbers:
they say what to review, test, investigate or monitor, never what will happen.
"""
from findings_engine import Findings, label, poss

# Budget actions follow efficiency (return per unit of spend), never totals: a segment with the most revenue or
# profit may simply have the most spend.
EFFICIENCY = {"roas", "roi_pct", "conversion_rate", "ctr"}                 # higher is better, budget-relevant
VOLUME = {"revenue", "profit", "conversions", "clicks", "impressions"}     # higher is better, but scale-driven
COST = {"cpa", "cpc"}                                                      # lower is better
NO_GUARANTEE = "These are suggestions to test against your own context; the data can't predict the result."


def _segment_rules(f: Findings) -> list:
    out = []
    m = f.metric
    if m in COST and f.worst and f.best:
        out.append(f"Prioritize {label(m)} reduction on {f.worst} (the highest {label(m)}) before scaling its spend: "
                   "review its bidding, targeting and conversion path.")
        out.append(f"Use {f.best} (the lowest {label(m)}) as the benchmark when testing those changes.")
    elif m in EFFICIENCY and f.worst and f.best:
        out.append(f"Review targeting, creative performance and spend allocation for {f.worst} (the lowest "
                   f"{label(m)}) before increasing its budget.")
        out.append(f"Consider testing incremental budget on {f.best} (the highest {label(m)}) while monitoring "
                   "whether its efficiency remains stable.")
    elif m in VOLUME and f.worst and f.best:
        out.append(f"Investigate why {f.worst} brings the least {label(m)} compared with {f.best}: check its spend, "
                   "conversions and ROAS before changing its budget.")
    if f.roas_best and f.roas_worst and m not in EFFICIENCY:
        out.append(f"Consider testing incremental budget on {f.roas_best} (the highest ROAS) while monitoring whether "
                   f"its efficiency holds, and review {f.roas_worst} (the lowest ROAS) before adding budget.")
        if f.top_spender and f.top_spender_roas_rank and f.entity_count \
                and f.top_spender_roas_rank > (f.entity_count + 1) / 2 and f.top_spender != f.roas_best:
            out.append(f"Review whether {poss(f.top_spender)} large share of spend is justified: its ROAS is in the lower "
                       "half.")
    return out


def recommend(f: Findings) -> list:
    """Recommended actions for the findings (empty when the evidence doesn't support any)."""
    if f.kind in ("ranking", "comparison", "context"):
        out = _segment_rules(f)
        if f.kind == "context" and f.focus and f.focus == f.worst:
            out.insert(0, f"{f.focus} is the weakest on {label(f.metric)} here, so start the review with it.")
        if not out:
            out.append(f"Compare the strongest and weakest {label(f.metric)} results above on ROAS and CPA before "
                       "moving budget between them.")
    elif f.kind == "campaigns" and f.loss_campaigns is not None:
        out = ["Review or pause persistently loss-making campaigns, starting with the largest losses, and investigate "
               "their spend, revenue and conversion efficiency."] if f.loss_campaigns else \
              ["No campaigns in this selection are loss-making; monitor profit as spend changes."]
    elif f.kind == "campaigns":
        out = ["Review the campaigns listed above against your targets before changing their budgets."]
    elif f.kind == "trend" and f.direction == "decrease":
        lowest = f" (the lowest point was {f.lowest_period})" if f.lowest_period else ""
        out = [f"Investigate the periods contributing to the decline{lowest} and compare spend, conversions and "
               "efficiency before changing budget."]
    elif f.kind == "trend" and f.direction == "increase":
        out = [f"Monitor whether the increase in {label(f.metric)} holds, and check that efficiency (ROAS) keeps pace "
               "with spend."]
    elif f.kind == "trend":
        out = [f"Monitor {label(f.metric)} over the coming months before drawing conclusions."]
    elif f.kind == "relationship" and f.correlation is not None and abs(f.correlation) >= 0.3:
        out = [f"Treat the link between {label(f.x)} and {label(f.y)} as an association, not proof: test changes in "
               f"{label(f.x)} on a small share of budget and monitor {label(f.y)}."]
    elif f.kind == "relationship":
        out = [f"Don't rely on {label(f.x)} to predict {label(f.y)}: they move together only weakly, if at all."]
    elif f.kind == "total":
        out = [f"A single total doesn't show where to act: break {label(f.metric)} down by platform, objective or "
               "month to find the weakest segments."]
    else:
        out = []
    return out
