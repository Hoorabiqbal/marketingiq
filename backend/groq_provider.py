"""
Groq (hosted, OpenAI-compatible chat completions API): MarketingIQ's only LLM provider,
behind the LLMProvider contract (llm_adapter.py).

The system prompt is GROUNDING_INSTRUCTIONS; the user message is the question plus the typed
compact analysis rendered by build_user_prompt(). Every answer then goes through the adapter's
grounding validator.

Retry / timeout ownership for Groq: every attempt is bounded by the adapter's deadline.
Transient 5xx / network errors get one more attempt if enough of the deadline is left.
429 (rate limit) is never retried: it is reported as rate_limited so quota is not burnt
waiting it out. The API key is only ever sent in the Authorization header; it is never
logged, and error bodies from auth failures are not logged either.
"""
import json
import logging
import time

import httpx

from llm_adapter import (GROUNDING_INSTRUCTIONS, LLMProvider, LLMProviderError, ProviderAuthError,
                         ProviderNotConfiguredError, ProviderRateLimitError, ProviderResponseError,
                         ProviderTimeoutError, ProviderUnavailableError, build_user_prompt)

logger = logging.getLogger("marketingiq.llm_adapter")  # shares the adapter's handler

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Production model (not preview) on Groq, 131k context. Chosen over openai/gpt-oss-20b for its
# better grounding accuracy in the provider benchmark (benchmark_results/); on the free plan both
# have the same rate limits. Override with GROQ_MODEL.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
CONNECT_TIMEOUT_S = 5.0
MAX_ATTEMPTS = 2
MIN_ATTEMPT_S = 1.0
# The answer itself is ~150 words; the rest is headroom so a reasoning model's (hidden)
# reasoning tokens can't use up the budget before the answer is written.
MAX_COMPLETION_TOKENS = 1024
RATE_LIMIT_HEADERS = ("retry-after", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                      "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                      "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str = None, model: str = DEFAULT_GROQ_MODEL, base_url: str = DEFAULT_GROQ_BASE_URL,
                 http_client=None, sleep=None):
        self._api_key = (api_key or "").strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client()
        self._sleep = sleep or time.sleep

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def __repr__(self):  # never show the key
        return f"GroqProvider(model={self.model!r}, configured={self.configured})"

    def request_body(self, question: str, analysis: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": GROUNDING_INSTRUCTIONS},
                         {"role": "user", "content": build_user_prompt(question, analysis)}],
            "temperature": 0.2,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "stream": False,
        }
        if _is_reasoning_model(self.model):
            body["reasoning_effort"] = "low"
            body["include_reasoning"] = False
        return body

    def generate_explanation(self, question: str, analysis: dict, timeout_s: float) -> str:
        if not self.configured:
            raise ProviderNotConfiguredError("no GROQ_API_KEY")
        payload = self.request_body(question, analysis)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        deadline = time.monotonic() + timeout_s
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining < MIN_ATTEMPT_S:
                raise last_error or ProviderTimeoutError("groq deadline reached")
            timeout = httpx.Timeout(remaining, connect=min(CONNECT_TIMEOUT_S, remaining))
            try:
                response = self._client.post(f"{self.base_url}/chat/completions", json=payload,
                                             headers=headers, timeout=timeout)
            except httpx.TimeoutException as e:
                raise ProviderTimeoutError("groq request timed out") from e
            except httpx.TransportError as e:
                last_error = ProviderUnavailableError(f"groq network error ({type(e).__name__})")
                last_error.__cause__ = e
            else:
                if response.status_code < 500:
                    return self._handle(response)
                logger.warning("groq server error %s: %s", response.status_code, response.text[:300])
                last_error = ProviderUnavailableError(f"groq HTTP {response.status_code}")
            if attempt < MAX_ATTEMPTS - 1:
                self._sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise last_error

    def _handle(self, response) -> str:
        status = response.status_code
        if status in (401, 403):
            raise ProviderAuthError(f"groq HTTP {status}")  # body not logged: may echo the key
        if status == 429:
            limits = {h: response.headers[h] for h in RATE_LIMIT_HEADERS if h in response.headers}
            logger.warning(json.dumps({"event": "groq_rate_limited", "model": self.model, **limits}))
            raise ProviderRateLimitError("groq HTTP 429")
        if status == 404:  # users see the generic "unavailable" message; the log names the model
            logger.warning(json.dumps({"event": "groq_model_not_found", "model": self.model}))
            raise ProviderNotConfiguredError(f"groq model {self.model} not found")
        if status != 200:
            logger.warning("groq client error %s: %s", status, response.text[:300])
            raise LLMProviderError(f"groq HTTP {status}")
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"]["content"]
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as e:
            raise ProviderResponseError("malformed groq response") from e
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError(f"empty groq response (finish_reason={choice.get('finish_reason')})")
        self._log_usage(body, choice)
        return text.strip()

    def _log_usage(self, body: dict, choice: dict):
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        record = {"event": "groq_generation", "model": body.get("model", self.model),
                  "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                  "queue_time_s": usage.get("queue_time"), "total_time_s": usage.get("total_time"),
                  "finish_reason": choice.get("finish_reason")}
        logger.info(json.dumps(record))
