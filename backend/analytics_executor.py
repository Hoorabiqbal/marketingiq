"""
Deterministic data-tool execution for validated analytics plans (no question parsing, no LLM).

    query_planner.execute_plan(validated plan) -> ToolCall -> execute() -> data_tools on DuckDB

Every number in an answer comes from a data_tools function run here against the real dataset.
Moved unchanged from the former router module (whose question classification was removed when
the local Qwen model became the only planner): the tool dispatch with the dashboard filters, the
dimension -> filter-key map, and the data-layer errors with their user-safe messages.
"""
import inspect
import logging
from dataclasses import dataclass, field

import data_tools as dt

logger = logging.getLogger("marketingiq.analytics")
if not logger.handlers:
    # uvicorn doesn't configure non-uvicorn loggers, so INFO lines would be dropped.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Errors — each carries an HTTP status and a user-safe message (no internals).
# ---------------------------------------------------------------------------

class AnalyticsError(Exception):
    status_code = 500
    public_message = "The analytics service hit an unexpected error. Please try again."

    def __init__(self, public_message: str = None):
        if public_message:
            self.public_message = public_message
        super().__init__(self.public_message)


class DataUnavailableError(AnalyticsError):
    status_code = 503
    public_message = "The MarketingIQ dataset is not available right now. Please try again shortly."


class ToolUnavailableError(AnalyticsError):
    status_code = 503
    public_message = "The analytics tool needed for this question is not available right now."


class ToolExecutionError(AnalyticsError):
    status_code = 500
    public_message = "The analytics query failed. Please try again or rephrase the question."


# Tools that accept the dashboard's active filters — derived from the real signatures.
FILTERABLE_TOOLS = {name for name, fn in dt.TOOL_REGISTRY.items()
                    if "filters" in inspect.signature(fn).parameters}

# Dimension -> key understood by dt.apply_global_filters (for scoping a query to an entity).
ENTITY_FILTER_KEYS = {
    "platform": "platform", "objective": "objective", "vertical": "vertical",
    "budget_tier": "budget", "device": "device", "age_group": "age", "gender": "gender",
    "creative_format": "creative", "creative_emotion": "emotion", "placement": "placement",
    "income_bracket": "income",
}


@dataclass
class ToolCall:
    tool: str
    tool_input: dict = field(default_factory=dict)
    period: str = None           # "2025-01" or "2025": narrows a trend_over_time series
    extra_filters: dict = None   # entity scoping merged into the dashboard filters

    def describe(self) -> dict:
        out = {"tool": self.tool, "tool_input": self.tool_input}
        if self.period:
            out["period"] = self.period
        if self.extra_filters:
            out["scope"] = self.extra_filters
        return out


def execute(call: ToolCall, filters: dict) -> dict:
    fn = dt.TOOL_REGISTRY.get(call.tool)
    if fn is None:
        raise ToolUnavailableError()
    kwargs = dict(call.tool_input)
    if call.tool in FILTERABLE_TOOLS:
        kwargs["filters"] = {**filters, **(call.extra_filters or {})}
    try:
        result = fn(**kwargs)
    except RuntimeError as e:
        if "not loaded" in str(e).lower():
            raise DataUnavailableError()
        raise ToolExecutionError() from e
    except Exception as e:
        raise ToolExecutionError() from e
    if isinstance(result, dict) and "error" in result:
        raise ToolExecutionError() from ValueError(f"{call.tool}: {result['error']}")
    if call.period and call.tool == "trend_over_time":
        result = {**result, "period": call.period,
                  "series": [p for p in result["series"] if p["month"].startswith(call.period)]}
    return result


def run(tool: str, filters: dict, **kwargs) -> dict:
    """One data tool with the given filters (the call query_planner and the findings engine make)."""
    return execute(ToolCall(tool, kwargs), filters or {})
