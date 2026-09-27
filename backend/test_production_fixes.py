"""
Regression tests for four production issues: a complete new request overridden by conversation
memory, scatter/correlation requests, recommendation questions, and the public fetch-error text.

Groq runs on a fake transport that fails the test if reached, unless a test supplies its reply.

Run:  python test_production_fixes.py      (or: pytest test_production_fixes.py)
"""
import json
import logging
import uuid
from pathlib import Path

import fake_groq  # noqa: F401  (first: fake GROQ_API_KEY before main.py reads .env)
import numpy as np

from fastapi.testclient import TestClient

import data_tools as dt
import main
import query_router as qr

fake_groq.block_real_calls(main.llm_adapter)
client = TestClient(main.app)
ADVICE_REPLY = ("What the data shows:\n- Google Ads has the highest revenue of the platforms shown.\n"
                "Interpretation: the gap may reflect scale.\n"
                "Recommended actions:\n- Consider testing budget shifts toward the higher-ROAS platforms.")


def refuse(request):
    raise AssertionError("Groq must not be called")


class Chat:
    def __init__(self):
        self.session_id = uuid.uuid4().hex
        self.requests = []

    def ask(self, question, reply=None, session=True):
        fake = fake_groq.install(main.llm_adapter, fake_groq.reply(reply) if reply is not None else refuse)
        try:
            body = {"messages": [{"role": "user", "content": question}], "filters": {}}
            if session:
                body["session_id"] = self.session_id
            r = client.post("/api/chat", json=body)
        finally:
            fake_groq.block_real_calls(main.llm_adapter)
        assert r.status_code == 200, r.text
        self.requests = fake.requests
        return r.json(), len(fake.requests)


def correlation(x_col, y_col):
    df = dt.get_dataframe()
    return round(float(np.corrcoef(df[x_col], df[y_col])[0, 1]), 3)


def assert_scatter(body, x, y, x_col, y_col):
    chart = body["chart"]
    assert chart and chart["type"] == "scatter" and [s["key"] for s in chart["series"]] == [x, y], body
    rows = dt.get_dataframe().set_index("campaign_id")
    for p in chart["data"][:20]:  # plotted values are the campaigns' own values
        assert p[x] == round(float(rows.loc[p["campaign"], x_col]), 2)
        assert p[y] == round(float(rows.loc[p["campaign"], y_col]), 2)
    r = correlation(x_col, y_col)
    assert f"r = {r:.3f}" in body["answer"]
    assert ("positive" in body["answer"]) == (r >= 0.1) and ("negative" in body["answer"]) == (r <= -0.1)


def test_a_new_request_overrides_total_revenue_memory():
    c = Chat()
    c.ask("What is total revenue?")
    body, calls = c.ask("Build a scatter plot of revenue and conversions and tell me whether they are positively "
                        "or negatively correlated.")
    assert calls == 0 and "single figure" not in body["answer"]
    assert_scatter(body, "revenue", "conversions", "revenue", "conversions")
    # The real production wording (with "it" in it) too:
    c = Chat()
    c.ask("Total revenue")
    body, calls = c.ask("Build Scatter Plot on revenue and conversions show is it positive correlated or negative correlated")
    assert calls == 0 and "single figure" not in body["answer"]
    assert_scatter(body, "revenue", "conversions", "revenue", "conversions")


def test_b_second_relationship_after_a_breakdown():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("Build a scatter plot of revenue and ROAS.")
    assert calls == 0
    assert_scatter(body, "revenue", "roas", "revenue", "ROAS")


def test_c_memory_follow_ups_unchanged():
    c = Chat()
    c.ask("Show revenue by platform.")
    body, calls = c.ask("Make this a chart.")
    assert calls == 0 and body["chart"]["series"][0]["key"] == "revenue"
    body, calls = c.ask("Now spend.")
    spend = {r["name"]: r["spend"] for r in dt.rank_dimension("platform", "spend", limit=100)["results"]}
    assert calls == 0 and {row["name"]: row["spend"] for row in body["chart"]["data"]} == spend


def test_d_e_f_scatter_and_correlation_variants():
    body, calls = Chat().ask("Build a scatter plot of revenue and conversions.")
    assert calls == 0
    assert_scatter(body, "revenue", "conversions", "revenue", "conversions")
    body, calls = Chat().ask("Build a scatter plot of revenue and ROAS.")
    assert calls == 0
    assert_scatter(body, "revenue", "roas", "revenue", "ROAS")
    body, calls = Chat().ask("Plot conversions vs revenue.")
    assert calls == 0
    assert_scatter(body, "conversions", "revenue", "conversions", "revenue")
    body, calls = Chat().ask("Is revenue correlated with conversions?")  # no chart asked: the figure only
    r = correlation("revenue", "conversions")
    assert calls == 0 and body["chart"] is None and f"r = {r:.3f}" in body["answer"]


def test_g_recommendation_uses_the_conversation_and_data():
    c = Chat()
    c.ask("Compare revenue and profit across platforms.")
    c.ask("Why is Google Ads revenue higher?", reply=ADVICE_REPLY)
    body, calls = c.ask("How can I improve revenue or profit on the other platforms?", reply=ADVICE_REPLY)
    assert calls == 1 and body["route"] == "LLM_REQUIRED", body
    assert "couldn&#x27;t map" not in body["answer"] and "What the data shows" in body["answer"]
    prompt = json.loads(c.requests[0].content)["messages"][1]["content"]
    assert "platform" in prompt.lower() and "profit" in prompt.lower()  # context resolved from memory
    gads = next(r for r in dt.rank_dimension("platform", "revenue", limit=100)["results"] if r["name"] == "Google Ads")
    assert str(gads["revenue"]) in prompt  # the evidence is the data layer's figures
    for q in ("Give me recommendations based on this comparison.", "How can we improve these results?",
              "Based on the previous results, what actions should we take?"):
        body, calls = c.ask(q, reply=ADVICE_REPLY)
        assert calls == 1 and body["route"] == "LLM_REQUIRED" and "couldn&#x27;t map" not in body["answer"], q
    body, calls = c.ask("Make this a chart.")  # the analysis is still what "this" means
    assert calls == 0 and body["chart"] is not None


def test_h_direct_recommendation():
    for q in ("How can we improve profit across platforms?", "How can TikTok performance be improved?",
              "What should we optimize?"):
        for session in (True, False):
            c = Chat()
            body, calls = c.ask(q, reply=ADVICE_REPLY, session=session)
            assert calls == 1 and body["route"] == "LLM_REQUIRED" and body["chart"] is None, (q, session, body)


def test_i_fresh_chart_this_still_asks():
    body, calls = Chat().ask("Chart this.")
    assert calls == 0 and body["route"] == "NEEDS_CLARIFICATION" and body["chart"] is None


def test_j_public_fetch_error_message():
    page = (Path(__file__).parent / ".." / "site" / "dashboard.html").read_text(encoding="utf-8")
    catch = page[page.index("}catch(err){", page.index("async function sendChat")):][:600]
    assert "The analytics service is temporarily unavailable. Please try again in a moment." in catch
    for leak in ("FastAPI", "backend/README", "server is running", "err.message"):
        assert leak not in catch, leak


if __name__ == "__main__":
    for lg in (qr.logger, logging.getLogger("marketingiq.llm_adapter")):
        lg.setLevel(logging.ERROR)
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n=== ALL {len(tests)} PRODUCTION FIX TESTS PASSED ===")
