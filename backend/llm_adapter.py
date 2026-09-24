"""
MarketingIQ LLM Adapter — provider-independent explanation layer.

    question + compact analysis  ->  LLMAdapter  ->  typed context (grounding.py)
        ->  LLMProvider  ->  numeric-claim validation (grounding.py)  ->  explanation

The analytics tools remain the source of numerical truth; the LLM only explains
results it is handed. An answer containing a number the data doesn't support is
replaced by a deterministic summary of the data, the same way for every provider. The adapter guarantees what reaches a provider is small,
plain JSON (never a DataFrame, the CSV or the full campaign table) and turns
every provider failure into a structured result, so callers never crash
because an LLM is slow, rate-limited or misconfigured.

Nothing in this module is specific to any vendor. Providers (gemini_provider.py
today; others later) subclass LLMProvider and raise LLMProviderError subclasses.
"""
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext

import grounding

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

You receive a QUESTION and an ANALYSIS block: JSON computed from the real MarketingIQ campaign
data. ANALYSIS is your only source of facts.

Rules, no exceptions:
1. Every number you write must appear in ANALYSIS (rounding is fine). Never invent, estimate or
   calculate new figures: no differences, ratios, shares, averages or counts of your own.
2. Keep each value's unit from metric_units. "x" values such as ROAS are ratios, never
   percentages. "%" values are already percentages. "$" values are money. Never convert units.
3. Attach each value to its own metric, entity, month and scope. "aggregate"/"aggregates" are
   group figures; "examples" are individual campaigns, never averages. A filtered_subset's
   figures describe only that subset. creative_age is the age of the ad creative in days, not
   audience age.
4. analysis_scope says what this data can and cannot establish. The data is observational:
   describe reasons only as hypotheses ("may", "could"), never as proven causes. If the question
   needs something the data cannot establish (a cause, or a change over time with no time
   series), say so plainly.
5. If active_filters is non-empty, say the figures reflect that filtered view.
6. Format: "What the data shows:" (the relevant supplied figures), then "Interpretation:"
   (hypotheses). At most about 150 words, plain business language, no preamble or sign-off,
   no generic marketing advice unless the question asks for recommendations.
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


BUSY_MESSAGE = ("Too many AI explanations are running right now. The figures are shown without an "
                "explanation; please try again in a moment.")


class ProviderSlots:
    """Caps how many requests wait on an external LLM at once. Provider calls block a server worker
    thread for seconds; without a cap, enough of them (40+) take every worker and even
    DIRECT_DATABASE answers wait. Never queues: when full, the caller is told at once."""

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self._sem = threading.BoundedSemaphore(self.limit)

    @contextmanager
    def acquire(self):
        """Yields True with a slot held, or False immediately when all slots are taken."""
        got = self._sem.acquire(blocking=False)
        try:
            yield got
        finally:
            if got:
                self._sem.release()


class LLMAdapter:
    def __init__(self, provider: LLMProvider, timeout_s: float = DEFAULT_TIMEOUT_S,
                 max_analysis_bytes: int = MAX_ANALYSIS_BYTES, slots: ProviderSlots = None):
        self.provider = provider
        self.timeout_s = timeout_s
        self.max_analysis_bytes = max_analysis_bytes
        self.slots = slots

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
            local_started = time.perf_counter()
            context = grounding.build_llm_context(question, analysis)
            size = len(serialize_analysis(context).encode("utf-8"))
            local_ms = (time.perf_counter() - local_started) * 1000
        except (TypeError, ValueError):
            return finish("error", error="invalid_analysis",
                          message="The analysis could not be prepared for explanation.")
        if size > self.max_analysis_bytes:
            return finish("error", error="analysis_too_large", analysis_bytes=size,
                          message="The analysis is too large to send for explanation.")
        meta["analysis_bytes"] = size

        with (self.slots.acquire() if self.slots else nullcontext(True)) as got_slot:
            if not got_slot:
                return finish("error", error="busy", message=BUSY_MESSAGE)
            try:
                text = self.provider.generate_explanation(question, context, self.timeout_s)
            except LLMProviderError as e:
                return finish("error", error=e.code, message=e.public_message,
                              cause=type(e.__cause__).__name__ if e.__cause__ else None)
            except Exception:
                logger.exception("llm_adapter unexpected provider error")
                return finish("error", error=LLMProviderError.code, message=LLMProviderError.public_message)
        text, check = self._ground(question, context, text)
        check["local_ms"] = round(local_ms + check.pop("validation_ms"), 2)
        return finish("ok", text=text, grounding=check)

    def _ground(self, question: str, context: dict, text: str):
        """Every provider's answer passes the same deterministic check. An answer with a number the
        data doesn't support is never shown: it is replaced by a summary built from the data itself
        (no second LLM call). An unhedged causal claim keeps the answer but adds a caveat."""
        started = time.perf_counter()
        try:
            report = grounding.validate_explanation(text, context, question)
        except Exception:  # a validator bug must not expose an unchecked answer
            logger.exception("grounding validator error")
            report = grounding.ValidationReport(False, [{"category": "validator_error", "claim": ""}])
        if not report.passed:
            action, text = "replaced_with_data_summary", grounding.safe_summary(context)
        elif report.causal_flags:
            action, text = "caveat_added", f"{text.rstrip()}\n\n{grounding.CAUSAL_CAVEAT}"
        else:
            action = "none"
        ms = (time.perf_counter() - started) * 1000
        # Sanitized: the provider, the categories and the offending numbers only — never the
        # question, the answer text or any credential.
        logger.log(logging.INFO if report.passed else logging.WARNING, json.dumps({
            "event": "grounding_check", "provider": self.provider.name, "model": self.provider.model,
            "passed": report.passed, "action": action, "numbers_checked": report.numbers_checked,
            "causal_flags": report.causal_flags,
            "issues": [{k: str(v)[:40] for k, v in i.items()} for i in report.issues[:10]],
            "validation_ms": round(ms, 2)}))
        return text, {"passed": report.passed, "action": action, "issues": report.categories(),
                      "numbers_checked": report.numbers_checked, "validation_ms": ms}
