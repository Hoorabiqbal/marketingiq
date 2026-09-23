"""
Gemini implementation of the LLMProvider contract (llm_adapter.py).

Reuses the app's existing GeminiKeyRotator instance, so API-key configuration,
key rotation on quota errors and the "remember the last good key" state are
shared with /api/chat rather than duplicated. On top of that it adds what an
explanation call needs:
  - a hard per-request HTTP timeout, bounded by an overall deadline
  - the same bounded server-error retry policy /api/chat uses (3 attempts,
    1.5s then 3s backoff), but never past the deadline
  - mapping of the rotator's error tags onto provider-neutral errors
One Gemini call per explanation; no tools, so no multi-turn loop.
"""
import time

import httpx
from google.genai import types

from llm_adapter import (GROUNDING_INSTRUCTIONS, LLMProvider, LLMProviderError, ProviderAuthError,
                         ProviderNotConfiguredError, ProviderRateLimitError, ProviderResponseError,
                         ProviderTimeoutError, ProviderUnavailableError, build_user_prompt)

MIN_ATTEMPT_S = 1.0  # don't start an attempt with less time than this left


def _error_for_tag(tag: str) -> LLMProviderError:
    """Maps gemini_rotator's error tags. Their detail text is kept only as the
    exception message (server logs), never shown to users."""
    tag = tag or ""
    kind = tag.split(":", 1)[0]
    cls = {"no_key": ProviderNotConfiguredError, "auth": ProviderAuthError,
           "quota_exhausted": ProviderRateLimitError, "server": ProviderUnavailableError,
           "client": LLMProviderError}.get(kind, LLMProviderError)
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
    name = "gemini"

    def __init__(self, rotator, model: str, max_attempts: int = 3, backoff_s: float = 1.5, sleep=time.sleep):
        self.rotator = rotator
        self.model = model
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._sleep = sleep

    def generate_explanation(self, question: str, analysis: dict, timeout_s: float) -> str:
        if not self.rotator.configured:
            raise ProviderNotConfiguredError("no_key")

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=build_user_prompt(question, analysis))])]
        deadline = time.monotonic() + timeout_s

        for attempt in range(self.max_attempts):
            remaining = deadline - time.monotonic()
            if remaining < MIN_ATTEMPT_S:
                raise ProviderTimeoutError("deadline reached before attempt")
            config = types.GenerateContentConfig(
                system_instruction=GROUNDING_INSTRUCTIONS,
                temperature=0.2,
                http_options=types.HttpOptions(timeout=int(remaining * 1000)),  # milliseconds
            )
            try:
                response, err = self.rotator.call(self.model, contents, config)
            except (httpx.TimeoutException, TimeoutError) as e:
                raise ProviderTimeoutError("request timed out") from e
            except httpx.TransportError as e:
                raise ProviderUnavailableError("network error") from e

            if response is not None:
                return _extract_text(response)
            if err and err.startswith("server:") and attempt < self.max_attempts - 1:
                self._sleep(min(self.backoff_s * (attempt + 1), max(0.0, deadline - time.monotonic())))
                continue
            raise _error_for_tag(err)
        raise ProviderUnavailableError("retries exhausted")
