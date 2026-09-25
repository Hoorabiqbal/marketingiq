"""
MarketingIQ AI Analyst backend.

    question (+ active dashboard filters)
      -> Query Router (query_router.py)
           DIRECT_DATABASE -> DuckDB -> exact answer (no LLM)
           LLM_REQUIRED    -> DuckDB -> compact typed analysis -> Groq (llm_adapter.py,
                              groq_provider.py) -> grounding validator (grounding.py) -> safe answer
           UNSUPPORTED / NEEDS_CLARIFICATION -> the router's message (no LLM)
    explicit chart request -> DuckDB -> validated chart spec (chart_builder.py, no LLM)

Groq is the only LLM provider; there is no provider selection and no failover. The LLM never
chooses tools or sees the dataset: it only explains the analysis it is handed.

Run locally:
    uvicorn main:app --reload --port 8000
"""
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import chat_routing
import conversation
import data_tools as dt
import llm_providers
import query_router as qr
from llm_adapter import LLMAdapter, ProviderSlots

APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env")

CSV_PATH = os.getenv("MARKETINGIQ_CSV_PATH", str(APP_DIR / ".." / "data" / "tech_advertising_campaigns_dataset.csv"))
dt.load_data(CSV_PATH)
qr.warm_up()  # router entity index, built once before the first request

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

# Explanation layer for LLM_REQUIRED questions (/api/query and /api/chat): Groq only.
# At most LLM_MAX_CONCURRENT requests wait on the LLM at once, so slow provider calls can never
# occupy all of the server's worker threads (40 by default); extra ones get an immediate "busy".
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "20"))
provider_slots = ProviderSlots(LLM_MAX_CONCURRENT)
llm_adapter = LLMAdapter(llm_providers.build_provider(), timeout_s=llm_providers.timeout_seconds(),
                         slots=provider_slots)


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
    """The dashboard's AI Analyst. `messages` is the conversation; the latest user message is
    answered through the Query Router (earlier turns only mark follow-ups). `chart` is a validated
    chart spec (data + metadata, never code) or null."""
    started = time.perf_counter()
    last = req.messages[-1] if req.messages else None
    question = last.content if last is not None and last.role == "user" else ""
    decision = chat_routing.decide(question, len(req.messages) > 1, req.filters or {}, llm_adapter,
                                   session=sessions.get(req.session_id))
    qr.logger.info(json.dumps({"event": "chat_answered", "route": decision.route, "reason": decision.reason,
                               "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}))
    body = {"answer": decision.answer, "route": decision.route, "chart": decision.chart}
    if decision.charts:  # several charts (e.g. metrics in different units); `chart` stays the first
        body["charts"] = decision.charts
    return body


class QueryRequest(BaseModel):
    query: str = Field(max_length=qr.MAX_QUERY_LENGTH)
    filters: dict[str, str | bool | None] = {}


@app.post("/api/query")
def query(req: QueryRequest):
    """Query Router (query_router.py) as JSON: DIRECT_DATABASE questions are answered by a data
    tool with no LLM call; LLM_REQUIRED questions return the compact analysis plus the
    grounding-validated explanation (or a structured `llm` error)."""
    try:
        return qr.route_query(req.query, req.filters, explainer=llm_adapter)
    except qr.QueryRouterError as e:
        raise HTTPException(status_code=e.status_code, detail=e.public_message)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "campaigns_loaded": dt.row_count(),
        "data_backend": dt.data_backend_name(),
        "llm_provider": llm_adapter.provider.name,
        "llm_model": llm_adapter.provider.model,
        # Whether the Groq key is set (never the key itself).
        "llm_configured": llm_adapter.provider.configured,
    }
