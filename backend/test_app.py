"""
Tests the full FastAPI app + Gemini tool-use loop WITHOUT calling the real
Gemini API — mocks client.models.generate_content using REAL google-genai
SDK objects (types.Candidate, types.Content, types.Part, types.FunctionCall)
so the test faithfully exercises the actual response-parsing code, not a
guessed mock shape. Verifies the plumbing (routing, filter injection, JSON
serialization, multi-turn tool loop, hallucination guard) independent of
whether a real GEMINI_API_KEY is configured.
"""
import os
os.environ["GEMINI_API_KEY"] = "test-placeholder-not-real"

from unittest.mock import patch
from fastapi.testclient import TestClient
from google.genai import types

import main

client_app = TestClient(main.app)


def make_function_call_response(calls):
    """calls: list of (name, args_dict)"""
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    candidate = types.Candidate(content=types.Content(role="model", parts=parts))
    resp = types.GenerateContentResponse(candidates=[candidate])
    return resp


def make_text_response(text):
    candidate = types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
    return types.GenerateContentResponse(candidates=[candidate])


print("=== TEST 1: health check ===")
r = client_app.get("/api/health")
print(r.status_code, r.json())
assert r.status_code == 200 and r.json()["campaigns_loaded"] == 10000 and r.json()["gemini_key_configured"] is True

print("\n=== TEST 2: single tool call -> final answer (no filters) ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [
        make_function_call_response([("rank_dimension", {"dimension": "platform", "metric": "roas", "order": "desc", "limit": 1})]),
        make_text_response("TikTok has the highest ROAS at 11.18x."),
    ]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "Which platform has the highest ROAS?"}], "filters": {}})
    print(r.status_code, r.json())
    assert "TikTok" in r.json()["answer"]
    # Verify the SECOND call actually received the REAL tool result
    second_call_contents = mock_gen.call_args_list[1].kwargs["contents"]
    tool_result_part = second_call_contents[-1].parts[0]
    result_str = str(tool_result_part.function_response.response)
    print("Tool result sent back to Gemini:", result_str[:150])
    assert "TikTok" in result_str and "11.18" in result_str

print("\n=== TEST 3: tool call WITH active dashboard filter (budget=High) ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [
        make_function_call_response([("rank_dimension", {"dimension": "platform", "metric": "roas", "order": "desc", "limit": 1})]),
        make_text_response("Under the current High-budget filter, TikTok leads at 13.49x ROAS."),
    ]
    r = client_app.post("/api/chat", json={
        "messages": [{"role": "user", "content": "Which platform has the highest ROAS?"}],
        "filters": {"budget": "High"}
    })
    second_call_contents = mock_gen.call_args_list[1].kwargs["contents"]
    result_str = str(second_call_contents[-1].parts[0].function_response.response)
    print("Tool result with filter applied:", result_str[:150])
    assert "13.49" in result_str  # confirms the filter was actually applied server-side

print("\n=== TEST 4: compare_entities with a NON-EXISTENT platform (hallucination guard) ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [
        make_function_call_response([("compare_entities", {"dimension": "platform", "names": ["Google Ads", "Meta Ads", "TikTok", "LinkedIn"]})]),
        make_text_response("Meta Ads isn't in the MarketingIQ dataset — the available platforms are Facebook, Google Ads, Instagram, LinkedIn, TikTok, and Twitter."),
    ]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "Compare Google, Meta, TikTok, and LinkedIn"}], "filters": {}})
    second_call_contents = mock_gen.call_args_list[1].kwargs["contents"]
    result_str = str(second_call_contents[-1].parts[0].function_response.response)
    print("Tool result (should show not_found for Meta Ads):", result_str)
    assert "not_found" in result_str and "Meta Ads" in result_str

print("\n=== TEST 5: multi-tool-call loop (2 sequential tool calls before final answer) ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [
        make_function_call_response([("get_numeric_field_stats", {"fields": ["spend", "revenue"]})]),
        make_function_call_response([("filter_campaigns", {"conditions": [{"field": "spend", "operator": ">", "value": 4996.16}, {"field": "revenue", "operator": "<", "value": 665}], "sort_by": "profit", "order": "asc", "limit": 5})]),
        make_text_response("Found several high-spend, low-revenue campaigns, mostly on LinkedIn..."),
    ]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "Which campaigns have high spend but low revenue?"}], "filters": {}})
    print(r.status_code, r.json())
    print("Number of Gemini calls made:", mock_gen.call_count)
    assert mock_gen.call_count == 3

print("\n=== TEST 6: safety cap (Gemini never stops calling tools) ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [make_function_call_response([("get_totals", {})]) for _ in range(10)]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "loop forever"}], "filters": {}})
    print(r.status_code, r.json())
    assert "wasn't able to reach" in r.json()["answer"]
    assert mock_gen.call_count == 6  # capped, didn't loop forever

print("\n=== TEST 7: multiple simultaneous tool calls in one turn ===")
with patch.object(main.client.models, "generate_content") as mock_gen:
    mock_gen.side_effect = [
        make_function_call_response([
            ("get_entity_metrics", {"dimension": "platform", "name": "TikTok"}),
            ("get_entity_metrics", {"dimension": "platform", "name": "LinkedIn"}),
        ]),
        make_text_response("TikTok significantly outperforms LinkedIn on ROAS."),
    ]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "Compare TikTok and LinkedIn"}], "filters": {}})
    second_call_contents = mock_gen.call_args_list[1].kwargs["contents"]
    response_parts = second_call_contents[-1].parts
    print("Number of function responses sent back:", len(response_parts))
    assert len(response_parts) == 2

print("\n=== TEST 8: no API key configured -> graceful message, no crash ===")
with patch.object(main, "client", None):
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "test"}], "filters": {}})
    print(r.status_code, r.json())
    assert "GEMINI_API_KEY" in r.json()["answer"]

print("\n=== TEST 9: automatic retry on transient server overload (503), succeeds on 3rd try ===")
with patch.object(main.client.models, "generate_content") as mock_gen, patch.object(main.time, "sleep"):
    from google.genai import errors as ge
    server_err = ge.ServerError(code=503, response_json={"error": {"message": "high demand"}})
    mock_gen.side_effect = [server_err, server_err, make_text_response("Recovered after retries.")]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "test retry"}], "filters": {}})
    print(r.status_code, r.json())
    assert r.json()["answer"] == "Recovered after retries."
    assert mock_gen.call_count == 3

print("\n=== TEST 10: gives up gracefully after 3 straight server overloads ===")
with patch.object(main.client.models, "generate_content") as mock_gen, patch.object(main.time, "sleep"):
    from google.genai import errors as ge
    server_err = ge.ServerError(code=503, response_json={"error": {"message": "high demand"}})
    mock_gen.side_effect = [server_err, server_err, server_err]
    r = client_app.post("/api/chat", json={"messages": [{"role": "user", "content": "test retry exhausted"}], "filters": {}})
    print(r.status_code, r.json())
    assert "overloaded" in r.json()["answer"]
    assert mock_gen.call_count == 3

print("\n=== ALL TESTS PASSED ===")
