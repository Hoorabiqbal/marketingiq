"""
App-level tests for the Groq-only MarketingIQ backend (main.py): the health contract, the
endpoints the dashboard uses, and the guarantee that no Gemini / google-genai code is part of
the running app. Groq runs on a fake transport (fake_groq.py): no network, no quota.

Run:  python test_app.py      (or: pytest test_app.py)
"""
import json
import sys
from pathlib import Path

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)
from fake_groq import FAKE_KEY

from fastapi.testclient import TestClient

import main
from groq_provider import DEFAULT_GROQ_MODEL

GUARD = fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)
HERE = Path(__file__).parent

# The dashboard's request shape (site/dashboard.html): full history plus its filter state,
# where unset filters are null and "All ..." placeholders mean no filter.
DASHBOARD_FILTERS = {"platform": "All Platforms", "objective": "All Objectives", "vertical": "All Industry Verticals",
                     "budget": "All Budget Tiers", "retargeting": "Retargeting & Cold Combined",
                     "gender": None, "device": None, "age": None, "creative": None, "emotion": None,
                     "placement": None, "income": None}


def test_health_contract():
    r = client.get("/api/health")
    body = r.json()
    assert r.status_code == 200
    assert body == {"status": "ok", "campaigns_loaded": 10000, "data_backend": "duckdb", "llm_provider": "groq",
                    "llm_model": DEFAULT_GROQ_MODEL, "llm_configured": True}
    assert FAKE_KEY not in r.text and "gemini" not in r.text.lower()


def test_no_gemini_in_the_running_app():
    assert not [m for m in sys.modules if m == "google" or m.startswith("google.")]
    assert not [m for m in sys.modules if "gemini" in m.lower()]
    requirements = (HERE / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "genai" not in requirements and "gemini" not in requirements
    for name in ("rotator", "_gemini_tool_loop", "CHAT_ROUTER_ENABLED", "CHAT_FALLBACK_TIMEOUT_S", "TOOL_DEFS"):
        assert not hasattr(main, name), name
    assert not (HERE / "gemini_provider.py").exists() and not (HERE / "gemini_rotator.py").exists()


def test_only_the_expected_endpoints():
    paths = {r.path for r in main.app.routes} - {"/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}
    assert paths == {"/api/chat", "/api/query", "/api/health"}


def test_dashboard_chat_contract():
    """The dashboard sends its whole history and filter state and renders `answer` as HTML
    (plus `chart`, a validated chart spec, when there is one)."""
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    r = client.post("/api/chat", json={"messages": [*history, {"role": "user", "content": "What is total revenue?"}],
                                       "filters": DASHBOARD_FILTERS})
    assert r.status_code == 200 and set(r.json()) == {"answer", "route", "chart"}
    assert r.json() == {"answer": "Total revenue is $284,157,706.47 across 10,000 campaigns.",
                        "route": "DIRECT_DATABASE", "chart": None}
    # A dashboard filter changes the figures server-side.
    filtered = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is total revenue?"}],
                                              "filters": {**DASHBOARD_FILTERS, "budget": "High"}}).json()
    assert filtered["route"] == "DIRECT_DATABASE" and "$284,157,706.47" not in filtered["answer"]
    assert "Filtered view: budget tier = High." in filtered["answer"]
    assert GUARD.requests == []


def test_explanation_through_the_dashboard_contract():
    fake = fake_groq.install(main.llm_adapter, fake_groq.reply("What the data shows: TikTok leads on ROAS.\n"
                                                                "Interpretation: this may reflect cheaper clicks."))
    try:
        body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Why is TikTok performing better?"}],
                                              "filters": {**DASHBOARD_FILTERS, "budget": "High"}}).json()
    finally:
        fake_groq.block_real_calls(main.llm_adapter)
    assert body["route"] == "LLM_REQUIRED" and len(fake.requests) == 1
    assert body["answer"] == ("What the data shows: TikTok leads on ROAS.<br>"
                              "Interpretation: this may reflect cheaper clicks.")
    sent = json.loads(fake.user_prompt().split("ANALYSIS (JSON):\n", 1)[1])
    assert sent["active_filters"] == {"budget_tier": "High"}  # the dashboard filter reached the analysis


def test_query_endpoint_validates_input():
    assert client.post("/api/query", json={"query": ""}).status_code == 400
    assert client.post("/api/query", json={"query": "x" * 5000}).status_code == 422
    assert client.post("/api/query", json={"query": "What is total revenue?", "filters": {"budget": ["x"]}}).status_code == 422
    assert GUARD.requests == []


if __name__ == "__main__":
    import logging
    for name in ("marketingiq.query_router", "marketingiq.llm_adapter", "marketingiq.data"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} APP TESTS PASSED ===")
