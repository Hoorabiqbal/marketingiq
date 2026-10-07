"""Surface vocabulary for generated questions. Every metric/dimension word maps to one catalog name
(checked by check_vocab() against schema_catalog's own alias matcher); values are the dataset's."""

METRIC_WORDS = {
    "revenue": ["revenue", "revenue", "revenue", "sales", "sales", "revenues", "turnover"],
    "spend": ["spend", "spend", "ad spend", "ad spend", "spending", "cost", "budget spent"],
    "profit": ["profit", "profit", "net profit", "profits"],
    "conversions": ["conversions", "conversions", "conversions", "conversion count"],
    "clicks": ["clicks", "clicks", "click count"],
    "impressions": ["impressions", "impressions", "views", "impression count"],
    "campaign_count": ["number of campaigns", "campaign count"],
    "roas": ["ROAS", "ROAS", "roas", "return on ad spend"],
    "roi_pct": ["ROI", "ROI", "return on investment"],
    "cpa": ["CPA", "CPA", "cost per acquisition", "cost per conversion"],
    "cpc": ["CPC", "CPC", "cost per click"],
    "ctr": ["CTR", "CTR", "ctr", "click-through rate", "click through rate"],
    "conversion_rate": ["conversion rate", "conversion rate", "CVR", "conv rate"],
    "bounce_rate": ["bounce rate"],
    "avg_session_duration": ["average session duration", "session length", "time on site"],
    "pages_per_session": ["pages per session", "page depth"],
    "quality_score": ["quality score", "ad quality"],
}
MONEY = ["revenue", "spend", "profit"]
COUNTS = ["conversions", "clicks", "impressions"]
RATIOS = ["roas", "roi_pct", "cpa", "cpc", "ctr", "conversion_rate"]
AVERAGES = ["bounce_rate", "avg_session_duration", "pages_per_session", "quality_score"]
ADDITIVE = MONEY + COUNTS
LOWER_BETTER = {"cpa", "cpc", "bounce_rate"}
CAMPAIGN_SORT = ["spend", "revenue", "profit", "roas", "cpa", "conversion_rate"]    # chart_builder.CAMPAIGN_METRICS
CAMPAIGN_FIELDS = ["spend", "revenue", "profit", "roas", "cpa", "cpc", "ctr", "conversion_rate", "clicks",
                   "conversions", "impressions", "bounce_rate", "avg_session_duration", "pages_per_session",
                   "quality_score"]                                                  # data_tools.CAMPAIGN_FIELD_COLUMNS
METRIC_WEIGHTS = {"revenue": 5, "spend": 4, "profit": 4, "roas": 4, "conversions": 3, "ctr": 3, "cpa": 3,
                  "clicks": 2, "conversion_rate": 2, "impressions": 2, "roi_pct": 2, "cpc": 2, "bounce_rate": 1,
                  "avg_session_duration": 1, "pages_per_session": 1, "quality_score": 1}

DIM_WORDS = {  # (singular forms, plural forms)
    "platform": (["platform", "platform", "channel", "network"], ["platforms", "platforms", "channels", "networks"]),
    "objective": (["objective", "campaign objective", "goal"], ["objectives", "campaign objectives", "goals"]),
    "device": (["device", "device type"], ["devices", "device types"]),
    "age_group": (["age group", "age bracket"], ["age groups", "age brackets"]),
    "gender": (["gender"], ["genders"]),
    "vertical": (["vertical", "industry"], ["verticals", "industries"]),
    "budget_tier": (["budget tier", "budget level"], ["budget tiers", "budget levels"]),
    "creative_format": (["creative format", "ad format"], ["creative formats", "ad formats"]),
    "creative_emotion": (["creative emotion", "emotion"], ["creative emotions", "emotions"]),
    "placement": (["placement", "ad placement"], ["placements"]),
    "income_bracket": (["income bracket", "income level"], ["income brackets"]),
    "audience_interest": (["audience interest", "interest"], ["audience interests", "interests"]),
    "operating_system": (["operating system", "OS"], ["operating systems"]),
    "retargeting": (["retargeting status"], ["retargeting statuses"]),
    "day_of_week": (["day of the week", "weekday"], ["weekdays", "days of the week"]),
    "quarter": (["quarter"], ["quarters"]),
    "creative_size": (["creative size", "ad size"], ["creative sizes", "ad sizes"]),
    "ad_copy_length": (["ad copy length", "copy length"], ["ad copy lengths", "copy lengths"]),
    "call_to_action": (["call to action", "CTA"], ["call-to-action options", "CTA options"]),
    "purchase_intent": (["purchase intent level", "purchase intent"], ["purchase intent levels"]),
}
FILTERABLE = ["platform", "objective", "vertical", "budget_tier", "device", "age_group", "gender",
              "creative_format", "creative_emotion", "placement"]   # income_bracket values contain "$": never valid
DIM_WEIGHTS = {"platform": 6, "device": 4, "vertical": 3, "objective": 3, "age_group": 2, "gender": 2,
               "placement": 2, "creative_format": 2, "budget_tier": 1, "creative_emotion": 1}
EXTRA_DIMS = ["income_bracket", "audience_interest", "operating_system", "retargeting", "day_of_week", "quarter",
              "creative_size", "ad_copy_length", "call_to_action", "purchase_intent"]

VALUES = {
    "platform": ["Facebook", "Google Ads", "Instagram", "LinkedIn", "TikTok", "Twitter"],
    "objective": ["App Installs", "Brand Awareness", "Engagement", "Lead Generation"],  # "Conversions": curated only
    "device": ["Desktop", "Mobile", "Tablet"],
    "age_group": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "gender": ["Female", "Male"],
    "vertical": ["E-commerce", "Education", "Finance", "Gaming", "Healthcare", "SaaS"],
    "budget_tier": ["High", "Low", "Medium"],
    "creative_format": ["Carousel", "Image", "Interactive", "Story", "Text", "Video"],
    "creative_emotion": ["Curiosity", "Fear", "Joy", "Neutral", "Trust", "Urgency"],
    "placement": ["Display Network", "Feed", "In-Stream Video", "Search", "Sidebar", "Stories"],
}


def value_forms(d: str, v: str) -> list:
    """Noun phrases for one value, usable after "for" / "on" / "from"."""
    low = v.lower()
    return {
        "platform": [v, v, low] + (["Google"] if v == "Google Ads" else []),
        "device": {"Mobile": ["mobile", "Mobile", "phones", "mobile devices"],
                   "Desktop": ["desktop", "Desktop", "desktop computers"],
                   "Tablet": ["tablet", "tablets", "Tablet"]}[v] if d == "device" else None,
        "gender": {"Male": ["men", "male users", "Male"], "Female": ["women", "female users", "Female"]}.get(v),
        "age_group": [f"{v} year olds", f"the {v} age group", f"ages {v}", v] if v != "65+" else
                     ["the 65+ age group", "65+", "people 65+"],
        "vertical": {"E-commerce": ["e-commerce", "E-commerce", "ecommerce"], "SaaS": ["SaaS", "saas"]}.get(
            v, [v, low, f"the {v} vertical"]),
        "budget_tier": [f"{low}-budget campaigns", f"the {v} budget tier", f"{low} budget campaigns"],
        "creative_format": [f"{low} ads", f"{v} creatives", f"the {v} format"],
        "creative_emotion": [f"{low}-themed creatives", f"{v} ads", f"creatives with a {low} tone"],
        "placement": [f"the {v} placement", f"{v} placements", v],
        "objective": [f"{low} campaigns", f"the {v} objective", v],
    }[d]


def short_value(d: str, v: str) -> list:
    """Bare forms for lists and "A vs B"."""
    low = v.lower()
    return {"platform": [v, v, low], "device": [low, v], "gender": {"Male": ["men", "male"], "Female": ["women", "female"]}.get(v),
            "age_group": [v], "vertical": [v, low] if v != "SaaS" else ["SaaS"], "budget_tier": [f"{low} budget"],
            "creative_format": [v, low], "creative_emotion": [v, low], "placement": [v],
            "objective": [v, low]}[d]


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
QUARTERS = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}
NUM_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 10: "ten"}

UNKNOWN_METRICS = ["customer lifetime value", "CLV", "churn rate", "churn", "inventory level", "stock levels",
                   "average order value", "net promoter score", "NPS", "refund rate", "refunds", "email open rate",
                   "cart abandonment rate", "follower growth", "video completion rate", "market share",
                   "engagement rate", "likes", "shares", "comment count", "app store rating",
                   "customer satisfaction score", "retention rate", "gross margin", "shipping fees",
                   "watch time", "subscriber count", "return rate", "basket size", "lead score"]
UNKNOWN_DIMS = [("country", "countries"), ("region", "regions"), ("city", "cities"), ("language", "languages"),
                ("browser", "browsers"), ("hour of the day", "hours of the day"), ("time of day", "times of day"),
                ("product", "products"), ("product category", "product categories"), ("SKU", "SKUs"),
                ("store", "stores"), ("salesperson", "salespeople"), ("keyword", "keywords"),
                ("landing page", "landing pages"), ("zip code", "zip codes"), ("currency", "currencies"),
                ("weather condition", "weather conditions"),
                ("ad group", "ad groups"), ("account manager", "account managers")]
OUT_OF_RANGE_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2027, 2028, 2029, 2030]


def check_vocab():
    """Every metric word must read as its metric, every singular dimension word as its dimension."""
    import schema_catalog as catalog
    problems = []
    for m, forms in METRIC_WORDS.items():
        for w in set(forms):
            if catalog.find_metrics(w) != [m]:
                problems.append(f"metric word {w!r} -> {catalog.find_metrics(w)} (want {m})")
    for d, (singular, _) in DIM_WORDS.items():
        for w in set(singular):
            if catalog.find_dimensions(w) != [d]:
                problems.append(f"dimension word {w!r} -> {catalog.find_dimensions(w)} (want {d})")
    for u in UNKNOWN_METRICS:
        if catalog.find_metrics(u) or catalog.find_dimensions(u):
            problems.append(f"unknown metric {u!r} overlaps the catalog")
    for s, p in UNKNOWN_DIMS:
        if catalog.find_dimensions(s) or catalog.find_metrics(s) or catalog.find_dimensions(p):
            problems.append(f"unknown dimension {s!r} overlaps the catalog")
    return problems
