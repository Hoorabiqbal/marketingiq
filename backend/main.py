"""
MarketingIQ AI Analyst backend: 100% local, no cloud LLM, no API key.

    question (+ active dashboard filters, + the session's remembered validated plan for follow-ups)
      -> analyst.py
           small talk / exact campaign ID .............. deterministic, no model call
           everything else ............................. fine-tuned Qwen3-0.6B on a local llama-server
                                                          (local_planner.py): ONE call -> compact plan
                                                          -> validate_plan() -> DuckDB
              analysis      deterministic answer + validated chart (query_planner, chart_builder)
              explanation   deterministic findings (findings_engine)
              advice        findings + rule-based recommendations (recommendation_engine)

The model only plans; DuckDB computes every number. The llama-server runs separately and stays up
(experiments/tiny_model/PHASE7_LOCAL_QWEN.md); this app never starts it or loads the model.

Run locally:
    uvicorn main:app --reload --port 8000
"""
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import analyst
import analytics_executor as ax
import conversation
import data_tools as dt
import local_planner

APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env")

CSV_PATH = os.getenv("MARKETINGIQ_CSV_PATH", str(APP_DIR / ".." / "data" / "tech_advertising_campaigns_dataset.csv"))
dt.load_data(CSV_PATH)

app = FastAPI(title="MarketingIQ AI Analyst")

# Browsers may call this API only from the deployed frontend (CORS_ALLOW_ORIGINS, comma-separated;
# "*" allows any origin) or from a page served on this machine (localhost / 127.0.0.1, any port),
# which is how the dashboard is run locally. No cookies or credentials are involved.
DEFAULT_CORS_ORIGINS = "https://marketingiqp.netlify.app"
CORS_ALLOW_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
                      if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# The local planner: one persistent llama-server (MARKETINGIQ_LLAMA_URL), never started per request.
analyst.planner = local_planner.LocalPlanner.from_env()
if not analyst.planner.healthy():
    ax.logger.warning(json.dumps({"event": "local_model_unreachable", "url": analyst.planner.url}))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    filters: dict = {}
    # Opaque id the browser generates for its AI Analyst conversation; enables follow-ups.
    session_id: str | None = Field(default=None, max_length=64)


# Conversation memory: compact structured state per session, in this process only (bounded; a
# restart starts every conversation fresh).
sessions = conversation.SessionStore()


@app.post("/api/chat")
def chat(req: ChatRequest):
    """The dashboard's AI Analyst. `messages` is the conversation; the latest user message is answered
    (earlier turns only mark follow-ups). `chart` is a validated chart spec (data + metadata, never
    code) or null."""
    started = time.perf_counter()
    last = req.messages[-1] if req.messages else None
    question = last.content if last is not None and last.role == "user" else ""
    reply = analyst.answer(question, len(req.messages) > 1, req.filters or {}, session=sessions.get(req.session_id))
    ax.logger.info(json.dumps({"event": "chat_answered", "route": reply.route, "reason": reply.reason,
                               "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}))
    body = {"answer": reply.answer, "route": reply.route, "chart": reply.chart}
    if reply.charts:  # several charts (e.g. metrics in different units); `chart` stays the first
        body["charts"] = reply.charts
    return body


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "campaigns_loaded": dt.row_count(),
        "data_backend": dt.data_backend_name(),
        "planner": "local_qwen",
        "planner_model": analyst.planner.model,
        "local_model_reachable": analyst.planner.healthy(),
    }
