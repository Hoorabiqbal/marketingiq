"""
App-level tests for the local MarketingIQ backend (main.py): the health contract, the endpoints the
dashboard uses, the dashboard's request shape, and the guarantee that no cloud-LLM code (Groq, Gemini)
is part of the running app. The model is the fake llama-server (fake_llama.py): no GGUF, no network.

Run:  python test_app.py      (or: pytest test_app.py)
"""
import html
import sys
from pathlib import Path

import fake_llama  # noqa: F401  (first: points the app's model URL at a closed port)

from fastapi.testclient import TestClient

import data_tools as dt
import grounding
import main
from fake_llama import Model

client = TestClient(main.app)
HERE = Path(__file__).parent
TOTAL = {"What is total revenue?": {"intent": "total", "metrics": ["revenue"]}}

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
    assert body == {"status": "ok", "campaigns_loaded": 10000, "data_backend": "duckdb", "planner": "local_qwen",
                    "planner_model": main.analyst.planner.model, "local_model_reachable": False}  # tests: no model
    assert "key" not in r.text.lower()


def test_no_cloud_llm_in_the_running_app():
    loaded = [m for m in sys.modules if m.split(".")[0] in ("groq", "openai", "google", "anthropic")
              or "gemini" in m.lower() or "groq" in m.lower()]
    assert not loaded, loaded
    requirements = (HERE / "requirements.txt").read_text(encoding="utf-8").lower()
    for word in ("groq", "openai", "genai", "gemini", "anthropic"):
        assert word not in requirements, word
    for name in ("llm_adapter", "provider_slots", "PLANNER"):
        assert not hasattr(main, name), name


def test_only_the_expected_endpoints():
    paths = {r.path for r in main.app.routes} - {"/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"}
    assert paths == {"/api/chat", "/api/health"}
    assert client.post("/api/query", json={"query": "What is total revenue?"}).status_code in (404, 405)


def test_dashboard_chat_contract():
    """The dashboard sends its whole history and filter state and renders `answer` as HTML
    (plus `chart`, a validated chart spec, when there is one)."""
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    with Model(TOTAL) as m:
        r = client.post("/api/chat", json={"messages": [*history, {"role": "user", "content": "What is total revenue?"}],
                                           "filters": DASHBOARD_FILTERS})
        assert r.status_code == 200 and set(r.json()) == {"answer", "route", "chart"}
        body = r.json()
        assert body["route"] == "ANALYSIS" and body["chart"] is None and "$284,157,706.47" in body["answer"]
        assert "Filtered view" not in body["answer"]                 # placeholders are not filters
        # A dashboard filter changes the figures server-side and is shown.
        filtered = client.post("/api/chat", json={"messages": [{"role": "user", "content": "What is total revenue?"}],
                                                  "filters": {**DASHBOARD_FILTERS, "budget": "High"}}).json()
        assert m.calls == 2
    assert "$284,157,706.47" not in filtered["answer"]
    assert f"${dt.get_totals({'budget': 'High'})['revenue']:,.2f}" in filtered["answer"]
    assert "Filtered view: budget tier = High." in filtered["answer"]


def test_explanation_through_the_dashboard_contract():
    plans = {"TikTok performing better": {"intent": "breakdown", "metrics": ["roas"], "dimension": "platform"}}
    with Model(plans) as m:
        body = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Why is TikTok performing better?"}],
                                              "filters": {**DASHBOARD_FILTERS, "budget": "High"}}).json()
        assert m.calls == 1                                            # planning only; the text is deterministic
    text = html.unescape(body["answer"])
    rows = dt.rank_dimension("platform", metric="roas", limit=20, filters={"budget": "High"})["results"]
    assert body["route"] == "EXPLANATION" and text.startswith("The data shows:")
    assert (f"{rows[0]['name']} has the highest ROAS across {len(rows)} platforms at "
            f"{grounding.fmt('roas', rows[0]['roas'])}") in text
    assert "Filtered view: budget tier = High." in text


def test_invalid_requests():
    assert client.post("/api/chat", json={"messages": []}).json()["route"] == "NEEDS_CLARIFICATION"
    long = client.post("/api/chat", json={"messages": [{"role": "user", "content": "x" * 5000}]}).json()
    assert long["route"] == "NEEDS_CLARIFICATION" and "too long" in long["answer"]
    assert client.post("/api/chat", json={"messages": "nope"}).status_code == 422


if __name__ == "__main__":
    import logging
    logging.getLogger("marketingiq.analytics").setLevel(logging.CRITICAL)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} APP TESTS PASSED ===")
