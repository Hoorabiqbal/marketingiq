"""
The analytics planner: the fine-tuned Qwen3-0.6B (Q4_K_M GGUF) on a persistent local llama-server.
It is the only component that interprets language; it never answers, explains or supplies a number.

    question (+ the previous validated plan, for a follow-up only)
      -> exact Phase 4 training prompt (experiments/tiny_model: plan_format + llama_client, reused, not copied)
      -> ONE greedy call to the running llama-server (/completion, thinking OFF, n_predict from
         marketingiq_runtime.json); nothing is started or loaded per request
      -> parse JSON -> canonicalize (plan_format.to_canonical, unknown keys rejected)
      -> query_planner.validate_plan()   nothing unvalidated runs
      -> query_planner.execute_plan()    data tools on DuckDB, deterministic answer + charts

A model that is down, a malformed or truncated output, unknown keys and a rejected plan are each
reported as such and never executed; there is no retry and no other model.

Configuration (environment):
    MARKETINGIQ_LLAMA_URL       llama-server base URL (default http://127.0.0.1:8089)
    MARKETINGIQ_LLAMA_TIMEOUT_S request timeout in seconds (default 8)
    MARKETINGIQ_TINY_MODEL_DIR  experiments/tiny_model (prompt contract + export/marketingiq_runtime.json)
"""
import copy
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import analytics_executor as ax
import chart_builder as cb
import query_planner as qp

logger = logging.getLogger("marketingiq.analytics")

DEFAULT_URL = "http://127.0.0.1:8089"
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_TINY_DIR = Path(__file__).resolve().parent.parent / "experiments" / "tiny_model"
UNAVAILABLE = "The local analytics model is temporarily unavailable. Please try again in a moment."
EXECUTION_FAILED = "I couldn't compute that from the campaign data. Try asking it a different way."

# validate_plan() rejections of a field outside the allowlists -> the validator's own refusal for that
# reason. Wording only: the plan is rejected either way.
_REFUSAL_FOR = {"unknown metric": "unknown_metric", "unknown dimension": "unknown_dimension",
                "unknown dimension2": "unknown_dimension", "unknown filter dimension": "unknown_dimension"}
_FIELD_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z _-]{1,29}$")

# Loss-making campaigns, by the dashboard's own rule (data_tools.UNSUCCESSFUL_RULE: profit below $0).
LOSS_RE = re.compile(r"\bloss[- ]making\b|\blos(?:ing|t|e) money\b|\bmoney[- ]losing\b|\blosing campaigns?\b"
                     r"|\b(?:at|for|making) a loss\b|\bnegative[- ]profits?\b|\bunprofitable\b|\bin the red\b"
                     r"|\bunsuccessful\b", re.IGNORECASE)
LOSS_CONDITION = {"field": "profit", "operator": "<", "value": 0.0}
COMPARE_RE = re.compile(r"\b(?:compare|comparison|vs\.?|versus|against)\b", re.IGNORECASE)
PROFIT_PROXIES = ("profit", "revenue", "roas", "roi_pct")   # conditions a loss rule replaces


def _refusal_texts() -> dict:
    """validate_plan()'s own not_answerable wording per reason (computed once, at startup)."""
    texts = {}
    for reason in set(_REFUSAL_FOR.values()):
        try:
            qp.validate_plan(json.dumps({**qp.plan_to_raw(qp.Plan("total", ["revenue"])), "intent": "not_answerable",
                                         "reason": reason}))
        except qp.PlanError as e:
            texts[reason] = e.public
    return texts


def _contract(tiny_dir: Path):
    """The Phase 4 prompt contract and Phase 6 client, imported from the experiment (single source).
    plan_format.backend() would reload the CSV; it is pointed at the already loaded query_planner."""
    for sub in ("dataset", "benchmark"):
        path = str(tiny_dir / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    import llama_client
    import plan_format
    plan_format._qp = qp
    return plan_format, llama_client


@dataclass
class LocalAnswer(qp.PlannedAnswer):
    model_ms: float = None       # llama-server round trip (client side)
    raw_output: str = None       # what the model returned (logged truncated; never shown to the user)


def apply_loss_rule(plan: qp.Plan, question: str) -> qp.Plan:
    """A campaign list asked for with loss-making wording ("loss-making", "losing", "at a loss",
    "negative-profit", "unprofitable") means profit below $0, whatever proxy the model chose
    (revenue < 0, ROAS < 1, ...). The edited plan is validated again."""
    if plan.intent != "campaign_list" or not LOSS_RE.search(question or ""):
        return plan
    conditions = [c for c in plan.conditions if c["field"] not in PROFIT_PROXIES] + [dict(LOSS_CONDITION)]
    if conditions == plan.conditions and plan.order == "asc":
        return plan
    edited = copy.deepcopy(plan)
    edited.conditions, edited.order = conditions[:qp.MAX_CONDITIONS], "asc"
    if edited.metrics[0] not in cb.CAMPAIGN_METRICS:
        edited.metrics = ["profit"]
    checked = qp.validate_plan(json.dumps(qp.plan_to_raw(edited)))
    checked.visualization = plan.visualization
    return checked


class LocalPlanner:
    def __init__(self, url=DEFAULT_URL, timeout_s=DEFAULT_TIMEOUT_S, tiny_dir=DEFAULT_TINY_DIR):
        self.url, self.timeout_s, tiny_dir = url.rstrip("/"), float(timeout_s), Path(tiny_dir)
        self.pf, self.lc = _contract(tiny_dir)
        runtime = json.loads((tiny_dir / "export" / "marketingiq_runtime.json").read_text(encoding="utf-8"))
        if runtime["system_prompt"] != self.pf.SYSTEM_PROMPT or runtime["eos_token"] != self.lc.EOS:
            raise ValueError("marketingiq_runtime.json does not match the plan_format prompt contract")
        self.max_new_tokens = int(runtime["max_new_tokens"])
        self.refusals = _refusal_texts()
        self.model = runtime.get("model")

    @classmethod
    def from_env(cls, env=os.environ):
        return cls(env.get("MARKETINGIQ_LLAMA_URL", DEFAULT_URL),
                   float(env.get("MARKETINGIQ_LLAMA_TIMEOUT_S", DEFAULT_TIMEOUT_S)),
                   Path(env.get("MARKETINGIQ_TINY_MODEL_DIR", DEFAULT_TINY_DIR)))

    # ---- model call
    def prompt(self, question: str, previous: qp.Plan = None) -> str:
        """The exact training prompt; a follow-up carries the previous validated plan in sparse form."""
        return self.lc.render_prompt(question, qp.plan_to_raw(previous) if previous is not None else None)

    def complete(self, prompt: str) -> dict:
        """One greedy completion from the running llama-server. Raises OSError/ValueError when the
        server can't be reached or answers with something that isn't a completion."""
        body = json.dumps(self.lc.greedy_request(prompt, self.max_new_tokens, cache_prompt=True)).encode()
        req = urllib.request.Request(self.url + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(resp.read())
        if not isinstance(out, dict) or not isinstance(out.get("content"), str):
            raise ValueError("llama-server response has no completion text")
        return out

    def healthy(self, timeout_s: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout_s) as resp:
                return json.loads(resp.read()).get("status") == "ok"
        except (OSError, ValueError):
            return False

    # ---- question -> validated plan
    def plan(self, question: str, previous: qp.Plan = None, chart_question: str = None) -> LocalAnswer:
        """ONE model call -> parse -> canonicalize -> validate_plan() -> the same post-processing every
        plan gets. status "ok" carries the validated plan (nothing executed yet); anything else
        carries a safe message. chart_question: the user's own wording, for the chart rules."""
        t0 = time.perf_counter()
        try:
            out = self.complete(self.prompt(question, previous))
        except (OSError, ValueError) as e:   # refused / timeout / HTTP error / not JSON
            self._log("model_unavailable", t0, error=f"{type(e).__name__}: {e}"[:160])
            return LocalAnswer([UNAVAILABLE], status="model_unavailable")
        model_ms = (time.perf_counter() - t0) * 1000
        raw, timings = out["content"].strip(), out.get("timings") or {}
        stats = {"model_ms": round(model_ms, 1), "prompt_tokens": timings.get("prompt_n"),
                 "generated_tokens": timings.get("predicted_n"), "follow_up": previous is not None}

        def reject(status, detail, public=qp.FAILED):
            self._log(status, t0, detail=detail, raw=raw[:200], **stats)
            return LocalAnswer([public], status=status, model_ms=model_ms, raw_output=raw)

        if out.get("stop_type") != "eos":                       # cut off at n_predict, or stopped otherwise
            return reject("malformed_output", f"stop_type {out.get('stop_type')!r}")
        try:
            sparse = json.loads(raw)
        except ValueError:
            return reject("malformed_output", "not JSON")
        if not isinstance(sparse, dict):
            return reject("malformed_output", "not a JSON object")
        unknown = sorted(set(sparse) - set(self.pf.field_order()))
        if unknown or "intent" not in sparse:
            return reject("canonicalization_failed", f"unknown keys {unknown[:5]}" if unknown else "no intent")
        try:
            canonical = self.pf.to_canonical(sparse)
        except (TypeError, ValueError) as e:
            return reject("canonicalization_failed", f"{type(e).__name__}: {e}"[:120])
        try:
            plan = qp.validate_plan(canonical)
        except qp.PlanError as e:
            status = ("ambiguous" if str(e) == "not answerable (ambiguous)" else
                      "not_answerable" if str(e).startswith("not answerable") else "invalid_plan")
            return reject(status, str(e)[:120], self._refusal(str(e), sparse) or e.public)

        wording = chart_question or question
        charted_before = previous is not None and previous.visualization
        if not cb.VIZ_RE.search(wording) and not (charted_before and plan.visualization):
            # A chart only when the user asked for one, or when a follow-up continues a charted answer
            # ("now spend", "make it 100% stacked") and the plan keeps a chart type.
            plan.visualization = None
            if cb.SHOW_RE.search(wording) and plan.intent in ("breakdown", "trend", "relationship") \
                    and not plan.dimension2:
                # "Show ..." answers that are a ranking, a series or a relationship carry a chart.
                plan.visualization = {"trend": "line", "relationship": "scatter"}.get(plan.intent, "bar")
            elif COMPARE_RE.search(wording) and plan.intent == "trend" and len(plan.metrics) >= 2:
                plan.visualization = "line"      # "compare revenue and spend over time": one line per metric
        qp.normalize_cross_year(plan, question)
        try:
            plan = apply_loss_rule(plan, wording)
        except qp.PlanError as e:
            return reject("invalid_plan", f"loss rule: {e}"[:120], e.public)
        self._log("planned", t0, intent=plan.intent, metrics=plan.metrics, dimension=plan.dimension, **stats)
        return LocalAnswer([], status="ok", plan=plan, model_ms=model_ms, raw_output=raw)

    def _refusal(self, detail: str, sparse: dict):
        """Deterministic wording for a field outside the dataset ("Browser-level analysis isn't
        available ..."), built from the rejected value and the validator's own list."""
        reason = _REFUSAL_FOR.get(detail)
        text = self.refusals.get(reason)
        if text is None:
            return None
        if reason == "unknown_dimension":
            names = [sparse.get("dimension"), sparse.get("dimension2"),
                     *[f.get("dimension") for f in sparse.get("filters") or [] if isinstance(f, dict)]]
            bad = next((n for n in names if isinstance(n, str) and n not in qp.DIMENSIONS + ["none"]
                        and _FIELD_NAME_RE.match(n)), None)
            if bad:
                return (f"{bad.replace('_', ' ').capitalize()}-level analysis isn't available in the current "
                        f"MarketingIQ dataset. {text.split('. ', 1)[1]}")
        return text

    # ---- question -> validated plan -> DuckDB answer
    def answer(self, question: str, filters: dict, previous: qp.Plan = None) -> LocalAnswer:
        """At most one model call, then the validator and data tools. Never raises: model, plan and
        execution problems come back as a status with a safe message (details are logged)."""
        p = self.plan(question, previous)
        if p.status != "ok":
            return p
        result, failure = execute(p.plan, filters)
        if failure:
            return LocalAnswer([failure[1]], status=failure[0], model_ms=p.model_ms, raw_output=p.raw_output)
        return LocalAnswer(result.lines, result.chart, result.status, p.plan, result.charts,
                           model_ms=p.model_ms, raw_output=p.raw_output)

    def _log(self, status, t0, **fields):
        event = {"event": "local_plan", "status": status, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                 **fields}
        (logger.info if status in ("planned", "ok") else logger.warning)(json.dumps(event, default=str))


def execute(plan: qp.Plan, filters: dict):
    """(result, None) or (None, (status, safe message)). Never a stack trace to the user."""
    try:
        return qp.execute_plan(plan, filters), None
    except qp.PlanError as e:
        return None, ("invalid_plan", e.public)
    except ax.AnalyticsError as e:     # the data layer's own safe error
        logger.warning(json.dumps({"event": "execution_error", "error": type(e).__name__}))
        return None, ("execution_error", e.public_message)
    except Exception:  # noqa: BLE001 - e.g. the known campaign_list KeyError
        logger.exception("plan execution failed")
        return None, ("execution_error", EXECUTION_FAILED)
