"""
MarketingIQ AI Analyst backend — Gemini edition.

Why Gemini instead of Claude: Anthropic's API has no ongoing free tier (only a
small one-time trial credit for new accounts) — real usage requires billing.
Google's Gemini API has a genuine, ongoing, no-credit-card-required free tier
(Flash-class models, ~1,500 requests/day) that supports the same tool-use /
function-calling pattern this project needs. Get a free key at
https://aistudio.google.com/apikey — no payment method, no credit card.

Architecture (unchanged from the Claude version):
  frontend question + history + active dashboard filters
    -> Gemini (with tools defined below)
    -> Gemini picks a tool + params based on the actual question (real intent
       interpretation, not a hardcoded question list)
    -> we execute that tool against the real pandas dataset (data_tools.py —
       completely unchanged by this swap)
    -> tool result (real numbers) goes back to Gemini
    -> Gemini writes the final answer, grounded ONLY in the tool results
    -> answer returned to frontend

Run locally:
    uvicorn main:app --reload --port 8000
"""
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google.genai import types

import data_tools as dt
from gemini_rotator import GeminiKeyRotator, load_keys

APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env")

CSV_PATH = os.getenv("MARKETINGIQ_CSV_PATH", str(APP_DIR / ".." / "data" / "tech_advertising_campaigns_dataset.csv"))
dt.load_data(CSV_PATH)

app = FastAPI(title="MarketingIQ AI Analyst")

# Local dev: allow the static frontend (opened via file:// or a local server) to call this API.
# Tighten this to your real deployed frontend origin before shipping publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rotator = GeminiKeyRotator(load_keys())
MODEL = "gemini-3.6-flash"  # free-tier model; update here if Google renames/replaces it again

SYSTEM_PROMPT = """You are the MarketingIQ AI Analyst, a data-grounded marketing analytics assistant.

You have tools that query the REAL MarketingIQ campaign dataset. You must use these tools to
answer every data question — never answer a data question from memory or assumption.

Rules, no exceptions:
1. Every number in your answer must come from a tool result. Never invent, estimate, or round
   a number you did not receive from a tool.
2. If a tool returns "not_found" for something the user named (e.g. a platform that doesn't
   exist in the dataset), tell them plainly and mention what IS available instead — do not
   substitute a similar-sounding real value without saying so.
3. If a question asks about something no tool can answer (a field or dimension the dataset
   doesn't have), respond exactly: "That information is not available in the current
   MarketingIQ dataset." — optionally naming what IS available instead. Use list_available_fields
   if you're unsure whether something exists.
4. When the user's question could depend on "high"/"low"/"underperforming" etc., call
   get_numeric_field_stats first to get real percentiles, then use those as thresholds in
   filter_campaigns — never guess a threshold.
5. All dashboard filters (platform/objective/vertical/budget tier/retargeting) the user has
   currently active are automatically applied inside every tool call on the backend. If any
   filter is active, mention in your answer that the figure reflects the current filtered view
   (state which filters, briefly).
6. Write like a business analyst: state the metric, the real value, relevant comparison/context,
   and a short interpretation. Be concise — a few sentences, not a report.
7. For recommendations, only state a recommendation that follows from a real, tool-verified
   pattern (e.g. "high spend + low ROAS" verified via filter_campaigns), and frame it as a
   data-supported observation, not a guarantee.
8. Maintain conversation context — if the user says "it" or "that platform", resolve it from
   the conversation history.
9. If a tool result's campaign_count / matched_count is 0, say so plainly rather than
   describing zero as if it were a normal answer.
"""

# Same tool definitions as the Claude version, just using Gemini's `parameters` key
# instead of Anthropic's `input_schema` (both accept the same JSON-schema shape).
TOOL_DEFS = [
    {
        "name": "get_totals",
        "description": "Get aggregate totals (revenue, spend, profit, conversions, ROAS, ROI, CPA, CTR, conversion rate) across all campaigns matching the active dashboard filters. Use for 'what is total X' or 'overall performance' questions.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "rank_dimension",
        "description": "Rank all values of a dimension (e.g. all platforms, all age groups) by a metric. Use for 'which X has the highest/lowest Y' or 'top N' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": list(dt.DIMENSION_COLUMNS.keys())},
                "metric": {"type": "string", "enum": ["spend", "revenue", "profit", "conversions", "clicks",
                                                        "impressions", "roas", "cpa", "cpc", "ctr", "conversion_rate", "roi_pct"]},
                "order": {"type": "string", "enum": ["desc", "asc"]},
                "limit": {"type": "integer"},
            },
            "required": ["dimension"],
        },
    },
    {
        "name": "compare_entities",
        "description": "Compare specific named values within one dimension (e.g. compare 'Google Ads', 'TikTok', 'LinkedIn' within platform). Returns 'not_found' for any name that doesn't exist in the real data.",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": list(dt.DIMENSION_COLUMNS.keys())},
                "names": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dimension", "names"],
        },
    },
    {
        "name": "get_entity_metrics",
        "description": "Get full metrics for one specific named value within a dimension (e.g. just 'TikTok' within platform).",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": list(dt.DIMENSION_COLUMNS.keys())},
                "name": {"type": "string"},
            },
            "required": ["dimension", "name"],
        },
    },
    {
        "name": "percentage_share",
        "description": "Get what percentage of the total (across all values of a dimension) one specific entity represents for a given metric. Use for 'what % of total spend does X represent' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": list(dt.DIMENSION_COLUMNS.keys())},
                "name": {"type": "string"},
                "metric": {"type": "string", "enum": ["spend", "revenue", "profit", "conversions", "clicks", "impressions"]},
            },
            "required": ["dimension", "name", "metric"],
        },
    },
    {
        "name": "get_numeric_field_stats",
        "description": "Get real min/max/mean/median/25th/75th percentile for numeric fields. ALWAYS call this before deciding what counts as 'high' or 'low' for filter_campaigns — never guess a threshold.",
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {"type": "array", "items": {"type": "string", "enum": ["spend", "revenue", "profit", "roas", "cpa", "ctr", "conversion_rate", "clicks", "conversions"]}},
            },
            "required": ["fields"],
        },
    },
    {
        "name": "filter_campaigns",
        "description": "Find individual campaigns matching numeric conditions (e.g. spend > X and revenue < Y). Use real thresholds from get_numeric_field_stats, not guesses. Use for 'underperforming campaigns', 'high spend low revenue', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "enum": ["spend", "revenue", "profit", "roas", "cpa", "ctr", "conversion_rate", "clicks", "conversions"]},
                            "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "=="]},
                            "value": {"type": "number"},
                        },
                        "required": ["field", "operator", "value"],
                    },
                },
                "sort_by": {"type": "string"},
                "order": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": "integer"},
            },
            "required": ["conditions"],
        },
    },
    {
        "name": "trend_over_time",
        "description": "Get a monthly time series for a metric (revenue, spend, profit, or conversions). Use for trend/change-over-time questions.",
        "parameters": {
            "type": "object",
            "properties": {"metric": {"type": "string", "enum": ["revenue", "spend", "profit", "conversions"]}},
        },
    },
    {
        "name": "get_creative_fatigue",
        "description": "Get CTR by creative age bucket (0-15, 16-30, 31-45, 46-60, 61-90 days). Use for creative fatigue / creative age questions.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "list_available_fields",
        "description": "List every dimension, its real values, every available metric, and the dataset's date range. Call this if you're unsure whether something the user asked about actually exists in the dataset.",
        "parameters": {"type": "object", "properties": {}},
    },
]

GEMINI_TOOL = types.Tool(function_declarations=[
    types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["parameters"])
    for t in TOOL_DEFS
])


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    filters: dict = {}


def run_tool(name: str, tool_input: dict, filters: dict):
    fn = dt.TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    if name in ("get_totals", "rank_dimension", "compare_entities", "get_entity_metrics",
                "percentage_share", "trend_over_time", "get_creative_fatigue",
                "get_numeric_field_stats", "filter_campaigns"):
        tool_input = {**tool_input, "filters": filters}
    try:
        return fn(**tool_input)
    except Exception as e:
        return {"error": str(e)}


def _message_for_error(err: str) -> str:
    if err in (None,):
        return ""
    if err == "no_key":
        return ("No Gemini API key configured. Set GEMINI_API_KEY_1 (and optionally _2, _3, ...) "
                "or GEMINI_API_KEY in backend/.env (or your host's environment variables). "
                "Get a free key at https://aistudio.google.com/apikey, then restart/redeploy.")
    if err.startswith("auth:"):
        return ("The AI Analyst can't authenticate with Gemini — one of your configured "
                "GEMINI_API_KEY_* values is missing or invalid. This is a configuration issue, "
                "not a quota issue, so it won't resolve by retrying. Check backend/.env "
                "(or your host's environment variables) and restart/redeploy.")
    if err.startswith("quota_exhausted:"):
        return ("All configured Gemini API keys have hit their free-tier quota right now. "
                "Wait a bit and try again, or add another GEMINI_API_KEY_N as a fallback "
                f"(currently configured: {len(rotator.clients)} key(s)).")
    if err.startswith("server:"):
        return f"Gemini's servers are temporarily overloaded after a few retries — please try again in a moment. ({err[len('server:'):]})"
    if err.startswith("client:"):
        return f"The AI Analyst hit a request error: {err[len('client:'):]}"
    return f"The AI Analyst hit an unexpected error: {err}"


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not rotator.configured:
        return {"answer": _message_for_error("no_key")}

    filters = req.filters or {}
    contents = [
        types.Content(role=("model" if m.role == "assistant" else "user"), parts=[types.Part.from_text(text=m.content)])
        for m in req.messages
    ]

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[GEMINI_TOOL])

    # Agentic loop: keep calling Gemini, executing any tool calls it makes,
    # until it returns a final plain-text answer (or we hit a safety cap).
    for _ in range(6):
        response = None
        err = None
        # Server-error retries (transient infra overload) stay on whichever key the
        # rotator is currently using — that's a Gemini-wide issue, not a per-key one.
        for attempt in range(3):
            response, err = rotator.call(MODEL, contents, config)
            if response is not None or not (err and err.startswith("server:")):
                break
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))  # brief backoff: 1.5s, then 3s

        if response is None:
            return {"answer": _message_for_error(err)}

        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call is not None]

        if not function_calls:
            final_text = "".join(p.text for p in parts if p.text)
            return {"answer": final_text or "The AI Analyst didn't return a response for that question — try rephrasing it."}

        contents.append(candidate.content)
        response_parts = []
        for fc in function_calls:
            result = run_tool(fc.name, dict(fc.args or {}), filters)
            response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=response_parts))

    return {"answer": "I wasn't able to reach a final answer for that question — try rephrasing it or breaking it into a simpler question."}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "campaigns_loaded": len(dt.get_dataframe()),
        "gemini_key_configured": rotator.configured,
        "gemini_keys_configured": len(rotator.clients),
    }
