"""
Qwen (via a local Ollama server) implementation of the LLMProvider contract (llm_adapter.py).

Receives exactly what the Gemini provider receives — the question plus the same compact
analysis block, rendered by the shared build_user_prompt() — with a shorter system prompt
suited to a small local model (same grounding rules).

Retry / timeout ownership for Qwen: one HTTP request per explanation, bounded by the
adapter's timeout (fast connect timeout so a stopped Ollama fails immediately). No retries:
retrying a local model that timed out or ran out of memory only makes the machine slower.
"""
import json
import logging
from urllib.parse import urlsplit

import httpx

from llm_adapter import (LLMProvider, LLMProviderError, ProviderNotConfiguredError, ProviderResponseError,
                         ProviderTimeoutError, ProviderUnavailableError, build_user_prompt)

logger = logging.getLogger("marketingiq.llm_adapter")  # shares the adapter's handler

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b-instruct"
CONNECT_TIMEOUT_S = 3.0

QWEN_SYSTEM_PROMPT = """You explain marketing analytics results for MarketingIQ.
Rules:
- Use only the numbers in ANALYSIS. Never invent, estimate or calculate new numbers.
- You only know what is in ANALYSIS. If it lacks what the question needs, say so.
- Answer the question directly in two short parts:
  What the data shows: the relevant supplied figures.
  Interpretation: possible reasons, worded as possibilities, not facts.
- Maximum 120 words. No generic marketing advice.
- If filters_applied is not empty, say the figures reflect that filtered view."""

# Generation settings: low temperature for faithful numbers; output capped so a local CPU
# model can't run on; a context window comfortably above the ~16 KB analysis cap.
GENERATION_OPTIONS = {"temperature": 0.2, "num_predict": 300, "num_ctx": 4096}


def normalize_host(host: str) -> str:
    """Accepts Ollama-style values ('127.0.0.1:11434', 'localhost' -> default port) or full URLs (used as-is)."""
    host = (host or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if "://" not in host:
        host = "http://" + host
        if urlsplit(host).port is None:
            host += ":11434"
    return host


def _unavailable(internal: str, public: str) -> ProviderUnavailableError:
    e = ProviderUnavailableError(internal)
    e.public_message = public
    return e


class QwenProvider(LLMProvider):
    name = "qwen"

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_OLLAMA_MODEL, http_client=None):
        self.host = normalize_host(host)
        self.model = model
        self._client = http_client or httpx.Client()

    def generate_explanation(self, question: str, analysis: dict, timeout_s: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": QWEN_SYSTEM_PROMPT},
                         {"role": "user", "content": build_user_prompt(question, analysis)}],
            "stream": False,
            "options": GENERATION_OPTIONS,
        }
        timeout = httpx.Timeout(timeout_s, connect=min(CONNECT_TIMEOUT_S, timeout_s))
        try:
            response = self._client.post(f"{self.host}/api/chat", json=payload, timeout=timeout)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError("ollama request timed out") from e
        except httpx.TransportError as e:
            raise _unavailable(f"ollama unreachable at {self.host}",
                               "The local AI model server (Ollama) isn't reachable. Make sure Ollama is running.") from e

        body = self._json(response)
        if response.status_code == 404 or "not found" in str(body.get("error", "")).lower():
            e = ProviderNotConfiguredError(f"model {self.model} not found")
            e.public_message = (f"The local AI model '{self.model}' isn't installed in Ollama "
                                f"(run: ollama pull {self.model}).")
            raise e
        if response.status_code >= 500:
            logger.warning("ollama server error %s: %s", response.status_code, str(body.get("error", ""))[:300])
            raise _unavailable(f"ollama HTTP {response.status_code}",
                               "The local AI model failed to respond (it may be out of memory). Please try again.")
        if response.status_code != 200:
            raise LLMProviderError(f"ollama HTTP {response.status_code}")

        text = body.get("message", {}).get("content") if isinstance(body.get("message"), dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseError("malformed or empty ollama response")
        self._log_timings(body)
        return text.strip()

    @staticmethod
    def _json(response) -> dict:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            if response.status_code == 200:
                raise ProviderResponseError("ollama returned non-JSON") from e
            return {}
        if not isinstance(body, dict):
            raise ProviderResponseError("ollama returned an unexpected JSON shape")
        return body

    def _log_timings(self, body: dict):
        """Ollama reports its own timings (nanoseconds): model load, prompt processing, generation."""
        ms = lambda k: round(body[k] / 1e6, 1) if isinstance(body.get(k), (int, float)) else None
        eval_count, eval_ms = body.get("eval_count"), ms("eval_duration")
        record = {"event": "qwen_generation", "model": self.model, "load_ms": ms("load_duration"),
                  "prompt_tokens": body.get("prompt_eval_count"), "prompt_eval_ms": ms("prompt_eval_duration"),
                  "output_tokens": eval_count, "eval_ms": eval_ms, "total_ms": ms("total_duration"),
                  "tokens_per_s": round(eval_count / (eval_ms / 1000), 2) if eval_count and eval_ms else None,
                  "done_reason": body.get("done_reason")}
        logger.info(json.dumps(record))
