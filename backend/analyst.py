"""
MarketingIQ AI Analyst: the /api/chat answer path. Local only: no cloud LLM, no API key.

    message
      ├─ greeting / thanks / "what can you do" / personal topic ... fixed reply, no model call
      ├─ exact campaign ID ("CAMP_00012") ................................ campaign_lookup, no model call
      └─ everything else
           -> Qwen (local_planner): ONE call turns the question (+ the remembered validated plan for a
              follow-up) into a compact plan -> canonicalize -> validate_plan()
              ("why?" / "what should we improve?" reuse the remembered plan: no call at all)
           -> DuckDB (data tools) -> what the user asked for:
                analysis        deterministic answer text + validated chart (query_planner / chart_builder)
                explanation     findings_engine: factual statements computed from the results
                advice          findings_engine + recommendation_engine: rule-based actions

Qwen only plans: it never writes an answer, an explanation, a recommendation or a number. A plan
that fails parsing, canonicalization or validate_plan() never reaches DuckDB. Answers are HTML
(the dashboard inserts `answer` as HTML): every value is escaped and line breaks are <br>.
"""
import html
import re
from dataclasses import dataclass

import analytics_executor as ax
import campaign_lookup
import conversation
import data_tools as dt
import findings_engine
import local_planner as lp
import recommendation_engine

planner = None   # local_planner.LocalPlanner, set by main.py at startup

MAX_QUESTION_LENGTH = 1000
CLARIFY = "NEEDS_CLARIFICATION"
ASK_A_QUESTION = ("Ask a question about your MarketingIQ campaign data. Try asking about revenue, spend, profit, "
                  "ROAS, ROI, CPA, CPC, CTR, conversions, clicks or impressions — overall, by platform, objective, "
                  "device, audience or creative, by month, or for campaigns above or below a threshold.")
TOO_LONG = f"The question is too long (max {MAX_QUESTION_LENGTH} characters)."
ASK_IN_FULL = ("Please ask that as a complete question, for example \"What is TikTok's ROAS?\". "
               "Each question is answered from the data on its own, so answers can't build on earlier messages.")
NOTHING_TO_EXPLAIN = "What would you like me to explain? Ask a question about the data first, then ask why."
NOTHING_TO_ADVISE = ("What would you like recommendations on? Ask a question about the data first (for example "
                     "\"ROAS by platform\"), then ask what to improve.")
EXPLAIN_NOTE = "Based on the available metrics, this describes what happened; the data doesn't show why."
NO_ACTION = "The evidence here doesn't support a specific recommendation; try a breakdown such as ROAS by platform."
DATA_ROUTES = {"explanation": "EXPLANATION", "advice": "RECOMMENDATION"}

# "What platform is it on?", "How can this campaign improve?" after an answer about one campaign.
CAMPAIGN_REF_RE = re.compile(
    r"\b(?:it|its|it's|this|that|same)\s+(?:[\w-]+\s+)?campaign\b"
    r"|\bthe (?:unsuccessful |loss-making |selected |same |above |previous )?campaign\b(?!s)"
    r"|\b(?:this|that) one\b|\bits?\b", re.IGNORECASE)


@dataclass
class Reply:
    route: str                 # ANALYSIS / EXPLANATION / RECOMMENDATION / CAMPAIGN / CONVERSATION / UNSUPPORTED /
                               # NEEDS_CLARIFICATION / MODEL_UNAVAILABLE / DATA_ERROR
    answer: str                # HTML
    reason: str = None
    chart: dict = None         # a validated chart spec (chart_builder.validate_chart), or None
    charts: list = None        # 2+ validated charts (different units); `chart` is always the first
    plan: object = None        # the analysis as a validated query_planner.Plan (conversation memory; not sent)
    campaign: str = None       # the one campaign this answer is about (conversation memory; not sent)


def _html(lines) -> str:
    return "<br>".join(html.escape(str(line)) for line in lines if line)


COLUMN_TO_DIMENSION = {col: name for name, col in dt.DIMENSION_COLUMNS.items()}


def filter_note(filters: dict) -> str:
    """The active dashboard filters, as the data layer applies them ("" when none; "All ..."
    placeholders are not filters)."""
    parts = []
    for col, _, value in dt._filter_conditions(filters or {}):
        if isinstance(value, bool):
            parts.append("retargeting only" if value else "cold audiences only")
        else:
            parts.append(f"{COLUMN_TO_DIMENSION.get(col, col).replace('_', ' ')} = {value}")
    note = ", ".join(dict.fromkeys(parts))
    return f"Filtered view: {note}." if note else ""


def _with_campaign(question: str, session) -> str:
    """The remembered campaign's ID added when the message refers to that campaign ("what platform is
    it on?"), so the exact lookup answers it. Chart requests and messages naming an ID are left alone."""
    if not session.campaign or campaign_lookup.campaign_id(question) \
            or conversation.CHART_WORD_RE.search(question):
        return question
    if CAMPAIGN_REF_RE.search(question) or (conversation.request_kind(question) != "analysis"
                                            and conversation.about_previous(question)):
        return f"{question.strip()} ({session.campaign})"
    return question


def answer(question: str, has_history: bool, filters: dict, session=None) -> Reply:
    """session: the conversation's conversation.SessionState, or None for a stateless request."""
    if not question or not question.strip():
        return Reply(CLARIFY, _html([ASK_A_QUESTION]), reason="no_question")
    if len(question) > MAX_QUESTION_LENGTH:
        return Reply(CLARIFY, _html([TOO_LONG]), reason="too_long")
    talk = conversation.small_talk(question)
    if talk:
        kind, text = talk
        if session is not None:
            session.turns.append({"route": "CONVERSATION", "kind": kind})
        return Reply("UNSUPPORTED" if kind == "personal" else "CONVERSATION", _html([text]), reason=kind)
    if session is not None:
        question = _with_campaign(question, session)
    kind = conversation.request_kind(question)
    if campaign_lookup.campaign_id(question):
        r = _campaign(question, kind)
    elif kind in DATA_ROUTES:
        r = _explain_or_advise(question, kind, filters, session, has_history)
    else:
        r = _analysis(question, filters, session, has_history)
    if session is not None:
        if r.plan is not None:
            if r.chart and not r.plan.visualization:
                r.plan.visualization = r.chart["type"]
            session.remember(r.plan, question, r.route, r.chart)   # a new analysis replaces the old one
            session.campaign = None
        if r.campaign:
            session.campaign = r.campaign
        session.turns.append({"route": r.route, "kind": kind})
    return r


def _campaign(question: str, kind: str) -> Reply:
    try:
        lines, cid = campaign_lookup.answer(question, kind)
    except ax.AnalyticsError as e:
        return Reply("DATA_ERROR", _html([e.public_message]), reason=type(e).__name__)
    return Reply("CAMPAIGN", _html(lines), reason=f"campaign_{kind}", campaign=cid)


def _failure(a) -> Reply:
    """A planning or execution problem: reported, never executed, never sent to another model."""
    route = {"model_unavailable": "MODEL_UNAVAILABLE", "not_answerable": "UNSUPPORTED",
             "execution_error": "DATA_ERROR"}.get(a.status, CLARIFY)
    if a.status == "invalid_plan" and "isn't available in the current MarketingIQ dataset" in " ".join(a.lines):
        route = "UNSUPPORTED"
    return Reply(route, _html(a.lines), reason=a.status)


def _previous(question: str, session, has_history: bool):
    """(previous plan or None, early reply or None): the remembered plan for a follow-up only."""
    if session is None:
        if has_history and (conversation.REFERENCE_RE.search(question) or conversation.LEAD_RE.search(question)):
            return None, Reply(CLARIFY, _html([ASK_IN_FULL]), reason="follow_up")
        return None, None
    return (session.plan if conversation.is_follow_up(question, session.plan) else None), None


def _analysis(question: str, filters: dict, session, has_history: bool) -> Reply:
    previous, early = _previous(question, session, has_history)
    if early:
        return early
    if previous is None and session is not None and session.plan is None \
            and conversation.REFERENCE_RE.search(question) and not conversation.mentions(question).analytical:
        return Reply(CLARIFY, _html([conversation.NO_CONTEXT]), reason="follow_up_unclear")
    a = planner.answer(question, filters, previous=previous)
    if a.status not in ("ok", "no_data"):
        return _failure(a)
    return Reply("ANALYSIS", _html([*a.lines, filter_note(filters)]), reason=f"planned_{a.status}", chart=a.chart,
                 charts=a.charts, plan=a.plan if a.status == "ok" else None)


def _explain_or_advise(question: str, kind: str, filters: dict, session, has_history: bool) -> Reply:
    """Findings (and for advice, recommendations) from DuckDB results of a validated plan: the
    remembered one for "why?" / "what should we improve?", else ONE planning call."""
    remembered = session.plan if session is not None else None
    plan, new_plan = None, False
    if conversation.about_previous(question):
        if remembered is None:
            if session is None and has_history:
                return Reply(CLARIFY, _html([ASK_IN_FULL]), reason="follow_up")
            return Reply(CLARIFY, _html([NOTHING_TO_ADVISE if kind == "advice" else NOTHING_TO_EXPLAIN]),
                         reason="nothing_to_explain")
        plan = remembered
    else:
        previous, early = _previous(question, session, has_history)
        if early:
            return early
        p = planner.plan(conversation.analysis_question(question), previous, chart_question=question)
        if p.status != "ok":
            return _failure(p)
        plan, new_plan = p.plan, True
    try:
        f = findings_engine.find(plan, filters)
    except ax.AnalyticsError as e:
        return Reply("DATA_ERROR", _html([e.public_message]), reason=type(e).__name__)
    except Exception:  # noqa: BLE001 - never a stack trace to the user
        lp.logger.exception("findings failed")
        return Reply("DATA_ERROR", _html([lp.EXECUTION_FAILED]), reason="findings_error")
    if kind == "explanation":
        lines = ["The data shows:", *(f"- {s}" for s in f.statements), EXPLAIN_NOTE]
    else:
        actions = recommendation_engine.recommend(f)
        lines = ["Finding:", *(f"- {s}" for s in f.statements), "Recommended action:",
                 *((f"- {a}" for a in actions) if actions else [NO_ACTION]),
                 recommendation_engine.NO_GUARANTEE]
    return Reply(DATA_ROUTES[kind], _html([*lines, filter_note(filters)]), reason=f"{kind}_{f.kind}",
                 plan=plan if new_plan else None)
