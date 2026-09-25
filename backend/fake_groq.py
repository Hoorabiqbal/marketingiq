"""
Test double for the Groq API, shared by the test modules.

The real GroqProvider runs unchanged (request building, retries, error mapping); only its
HTTP transport is replaced by httpx.MockTransport. Tests therefore need no GROQ_API_KEY,
make no network calls and use no quota.

    import fake_groq                 # first: sets a fake GROQ_API_KEY before main.py loads .env
    import main
    guard = fake_groq.block_real_calls(main.llm_adapter)   # nothing can reach api.groq.com
"""
import json
import os

import httpx

FAKE_KEY = "gsk_test_FAKE_not_a_real_key"
# load_dotenv never overrides an existing variable, so this keeps the real key in .env unused.
os.environ["GROQ_API_KEY"] = FAKE_KEY
os.environ.pop("GROQ_MODEL", None)
os.environ.pop("LLM_PROVIDER", None)

from groq_provider import DEFAULT_GROQ_MODEL, GroqProvider  # noqa: E402


def completion(text, model=DEFAULT_GROQ_MODEL, finish_reason="stop"):
    """A Groq chat-completion response body."""
    return {"id": "chatcmpl-test", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 120, "queue_time": 0.01, "total_time": 0.3}}


def reply(text):
    return lambda request: httpx.Response(200, json=completion(text))


class FakeGroq:
    """httpx transport standing in for api.groq.com; records every request. `respond` is a
    function request -> httpx.Response (or raises an httpx exception)."""

    def __init__(self, respond=reply("ok")):
        self.respond, self.requests = respond, []

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    def body(self, i=-1) -> dict:
        return json.loads(self.requests[i].content)

    def user_prompt(self, i=-1) -> str:
        return self.body(i)["messages"][1]["content"]

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def provider(respond=reply("ok"), api_key=FAKE_KEY, model=DEFAULT_GROQ_MODEL):
    """(GroqProvider on a fake transport, FakeGroq)."""
    fake = FakeGroq(respond)
    return GroqProvider(api_key=api_key, model=model, http_client=fake.client(), sleep=lambda s: None), fake


def install(adapter, respond=reply("ok")) -> FakeGroq:
    """Point an adapter's existing GroqProvider at a new fake transport."""
    fake = FakeGroq(respond)
    adapter.provider._client = fake.client()
    return fake


def block_real_calls(adapter) -> FakeGroq:
    """No request through this adapter can reach the network: each one is recorded and refused.
    The adapter turns the refusal into an error result, so tests assert `.requests == []`."""
    def refuse(request):
        raise AssertionError("unexpected Groq call in a test")
    return install(adapter, refuse)
