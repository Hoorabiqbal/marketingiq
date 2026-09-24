"""
Gemini calls for MarketingIQ: the shared call policy, and the LLMProvider
implementation used by the LLM Adapter (llm_adapter.py).

Retry / timeout ownership for every Gemini request in the app:
  - quota errors -> key rotation ............ GeminiKeyRotator (gemini_rotator.py) only
  - per-request timeout + overall deadline .. call_gemini() only
  - transient 5xx / network retry ........... call_gemini() only (bounded attempts, backoff,
                                              never past the deadline)
  - error -> user message ................... the caller: GeminiProvider -> adapter error codes;
                                              /api/chat's fallback loop -> _message_for_error
The SDK's own retries stay off (the rotator's clients set no retry_options), and
no endpoint or adapter retries on top, so one failure is retried in one place.
"""
import logging
import time

import httpx
from google.genai import types

from llm_adapter import (GROUNDING_INSTRUCTIONS, LLMProvider, LLMProviderError, ProviderAuthError,
                         ProviderNotConfiguredError, ProviderRateLimitError, ProviderResponseError,
                         ProviderTimeoutError, ProviderUnavailableError, build_user_prompt)

logger = logging.getLogger("marketingiq.llm_adapter")  # shares the adapter's handler

MIN_ATTEMPT_S = 1.0  # don't start an attempt with less time than this left
MAX_ATTEMPTS = 3     # transient-error attempts per call (1.5s, then 3s backoff)
BACKOFF_S = 1.5


def call_gemini(rotator, model: str, contents, config: dict, deadline: float,
                max_attempts: int = MAX_ATTEMPTS, backoff_s: float = BACKOFF_S, sleep=None):
    """One Gemini generate_content call under the shared policy above.

    config: GenerateContentConfig fields (the HTTP timeout is added here, per attempt, from
    `deadline`, a time.monotonic() value). Returns (response, error_tag) like the rotator,
    where error_tag may also be 'timeout:...'. Worst case: max_attempts x (number of keys)
    HTTP requests, all inside the deadline."""
    sleep = sleep or time.sleep  # looked up per call so tests can patch time.sleep
    err = None
    for attempt in range(max_attempts):
        remaining = deadline - time.monotonic()
        if remaining < MIN_ATTEMPT_S:
            return None, "timeout:deadline reached"
        cfg = types.GenerateContentConfig(**config, http_options=types.HttpOptions(timeout=int(remaining * 1000)))
        try:
            response, err = rotator.call(model, contents, cfg)
        except (httpx.TimeoutException, TimeoutError):
            return None, "timeout:request timed out"
        except httpx.TransportError as e:
            response, err = None, f"server:network error ({type(e).__name__})"
        if response is not None:
            return response, None
        if not (err and err.startswith("server:")) or attempt == max_attempts - 1:
            return None, err
        sleep(min(backoff_s * (attempt + 1), max(0.0, deadline - time.monotonic())))
    return None, err


def _error_for_tag(tag: str) -> LLMProviderError:
    """Maps error tags onto provider-neutral errors. Tag detail text is kept only as the
    exception message (server logs), never shown to users."""
    tag = tag or ""
    kind = tag.split(":", 1)[0]
    cls = {"no_key": ProviderNotConfiguredError, "auth": ProviderAuthError,
           "quota_exhausted": ProviderRateLimitError, "server": ProviderUnavailableError,
           "timeout": ProviderTimeoutError, "client": LLMProviderError}.get(kind, LLMProviderError)
    if kind == "server":
        # Gemini's 5xx text (e.g. "model overloaded") is useful when diagnosing; it's logged
        # server-side only. Auth/client error text is never logged, as it could echo a key.
        logger.warning("gemini server error: %s", tag[len("server:"):][:300])
    return cls(kind)


def _extract_text(response) -> str:
    try:
        parts = response.candidates[0].content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None)).strip()
    except (AttributeError, IndexError, TypeError) as e:
        raise ProviderResponseError("malformed response") from e
    if not text:
        raise ProviderResponseError("empty response")
    return text


class GeminiProvider(LLMProvider):
    """Explanations via the app's existing GeminiKeyRotator (shared keys, rotation state and
    quota handling with the /api/chat fallback). One call per explanation; no tools."""
    name = "gemini"

    def __init__(self, rotator, model: str, max_attempts: int = MAX_ATTEMPTS, backoff_s: float = BACKOFF_S,
                 sleep=None):
        self.rotator = rotator
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._sleep = sleep

    def generate_explanation(self, question: str, analysis: dict, timeout_s: float) -> str:
        if not self.rotator.configured:
            raise ProviderNotConfiguredError("no_key")
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=build_user_prompt(question, analysis))])]
        config = {"system_instruction": GROUNDING_INSTRUCTIONS, "temperature": 0.2}
        response, err = call_gemini(self.rotator, self.model, contents, config, time.monotonic() + timeout_s,
                                    self.max_attempts, self.backoff_s, self._sleep)
        if response is None:
            raise _error_for_tag(err)
        return _extract_text(response)
