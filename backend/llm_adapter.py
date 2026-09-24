"""
MarketingIQ LLM Adapter — provider-independent explanation layer.

    question + compact analysis block  ->  LLMAdapter  ->  LLMProvider  ->  explanation

The analytics tools remain the source of numerical truth; the LLM only explains
results it is handed. The adapter guarantees what reaches a provider is small,
plain JSON (never a DataFrame, the CSV or the full campaign table) and turns
every provider failure into a structured result, so callers never crash
because an LLM is slow, rate-limited or misconfigured.

Nothing in this module is specific to any vendor. Providers (gemini_provider.py
today; others later) subclass LLMProvider and raise LLMProviderError subclasses.
"""
import json
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger("marketingiq.llm_adapter")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

DEFAULT_TIMEOUT_S = 25.0
MAX_ANALYSIS_BYTES = 16_000

# Shared by every provider so answers are grounded the same way regardless of model.
GROUNDING_INSTRUCTIONS = """You are the MarketingIQ explanation layer for a marketing analytics product.

You receive a user QUESTION and an ANALYSIS block: JSON results computed from the real
MarketingIQ campaign dataset by the analytics engine. The ANALYSIS block is your only source
of facts.

Rules, no exceptions:
1. Use only numbers and facts that appear in ANALYSIS. Never invent, estimate, extrapolate or
   calculate new figures. You may say one supplied value is higher or lower than another.
2. If ANALYSIS does not contain what the question needs, say plainly which part cannot be
   answered from the available data. Do not fill the gap from general knowledge.
3. Keep facts and interpretation visibly separate:
   - "What the data shows:" the relevant supplied figures.
   - "Interpretation:" possible business explanations, worded as hypotheses ("may", "could",
     "one possible reason"). The data shows correlation only — never claim a proven cause.
4. If filters_applied is non-empty, say the figures reflect that filtered view.
5. Be concise: at most about 150 words, plain business language, no preamble or sign-off.
   No generic marketing advice unless the question asks for recommendations.
"""


def build_user_prompt(question: str, analysis: dict) -> str:
    return f"QUESTION:\n{question.strip()}\n\nANALYSIS (JSON):\n{serialize_analysis(analysis)}"


def serialize_analysis(analysis) -> str:
    """Compact JSON. Raises TypeError for anything that isn't plain data — a DataFrame,
    Series or array can never be silently stringified into a prompt."""
    def scalar_only(o):
        if getattr(o, "ndim", None) == 0 and hasattr(o, "item"):
            return o.item()  # numpy scalar
        raise TypeError(f"{type(o).__name__} is not allowed in an LLM analysis block")
    return json.dumps(analysis, separators=(",", ":"), default=scalar_only)


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    """Base for provider failures. `code` is machine-readable; `public_message` is user-safe."""
    code = "provider_error"
    public_message = "The AI explanation service hit an unexpected error."


class ProviderNotConfiguredError(LLMProviderError):
    code = "not_configured"
    public_message = "The AI explanation service is not configured (no API key)."


class ProviderAuthError(LLMProviderError):
    code = "auth_error"
    public_message = "The AI explanation service rejected its credentials. This is a configuration issue."


class ProviderRateLimitError(LLMProviderError):
    code = "rate_limited"
    public_message = "The AI explanation service is at its usage limit right now. Please try again shortly."


class ProviderTimeoutError(LLMProviderError):
    code = "timeout"
    public_message = "The AI explanation took too long and was cancelled. Please try again."


class ProviderUnavailableError(LLMProviderError):
    code = "unavailable"
    public_message = "The AI explanation service is temporarily overloaded. Please try again in a moment."


class ProviderResponseError(LLMProviderError):
    code = "bad_response"
    public_message = "The AI explanation service returned an unusable response."


class LLMProvider(ABC):
    name: str = "provider"
    model: str = None

    @abstractmethod
    def generate_explanation(self, question: str, analysis: dict, timeout_s: float) -> str:
        """Return explanation text for `question`, grounded only in `analysis`.

        Must finish (or raise ProviderTimeoutError) within `timeout_s` seconds, and raise
        an LLMProviderError subclass for any failure."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

_RESULT_LIST_KEYS = ("results", "series", "campaigns", "buckets")


def _is_empty_result(result) -> bool:
    if not result:
        return True
    if not isinstance(result, dict):
        return False
    if result.get("campaign_count") == 0 or result.get("matched_count") == 0:
        return True
    lists = [result[k] for k in _RESULT_LIST_KEYS if isinstance(result.get(k), list)]
    return bool(lists) and not any(lists)


class LLMAdapter:
    def __init__(self, provider: LLMProvider, timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_analysis_bytes: int = MAX_ANALYSIS_BYTES):
        self.provider = provider
        self.timeout_s = timeout_s
        self.max_analysis_bytes = max_analysis_bytes

    def explain(self, question: str, analysis: dict) -> dict:
        """analysis: {"results": [{"tool", "tool_input", "result"}, ...], plus optional context
        such as "filters_applied" / "focus_metrics"}. Never raises for provider failures."""
        started = time.perf_counter()
        meta = {"provider": self.provider.name, "model": self.provider.model}

        def finish(status, **fields):
            out = {"status": status, **meta, **{k: v for k, v in fields.items() if v is not None},
                   "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
            log = {k: v for k, v in out.items() if k not in ("text", "message")}
            (logger.info if status == "ok" else logger.warning)(json.dumps({"event": "llm_explain", **log}))
            return out

        results = analysis.get("results") if isinstance(analysis, dict) else None
        if not isinstance(question, str) or not question.strip() or not isinstance(results, list):
            return finish("error", error="invalid_request",
                          message="The explanation request was malformed.")
        if all(_is_empty_result(r.get("result") if isinstance(r, dict) else None) for r in results):
            return finish("skipped", error="empty_analysis",
                          message="No campaign data matched this question, so there is nothing to explain.")
        try:
            size = len(serialize_analysis(analysis).encode("utf-8"))
        except (TypeError, ValueError):
            return finish("error", error="invalid_analysis",
                          message="The analysis could not be prepared for explanation.")
        if size > self.max_analysis_bytes:
            return finish("error", error="analysis_too_large", analysis_bytes=size,
                          message="The analysis is too large to send for explanation.")
        meta["analysis_bytes"] = size

        try:
            text = self.provider.generate_explanation(question, analysis, self.timeout_s)
        except LLMProviderError as e:
            return finish("error", error=e.code, message=e.public_message,
                          cause=type(e.__cause__).__name__ if e.__cause__ else None)
        except Exception:
            logger.exception("llm_adapter unexpected provider error")
            return finish("error", error=LLMProviderError.code, message=LLMProviderError.public_message)
        return finish("ok", text=text)
