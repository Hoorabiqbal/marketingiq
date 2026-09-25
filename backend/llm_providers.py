"""
LLM configuration for the LLM Adapter. Groq is MarketingIQ's only LLM provider: there is no
provider selection and no failover.

    GROQ_API_KEY          required for explanations. A missing key doesn't stop startup:
                          explanations then report the AI service as unavailable.
    GROQ_MODEL            optional (default openai/gpt-oss-120b)
    LLM_TIMEOUT_SECONDS   optional per-explanation deadline (default 25)

LLM_PROVIDER is no longer read. Any other value left over from older configurations is
logged as ignored at startup.
"""
import json
import logging
import os

from groq_provider import DEFAULT_GROQ_MODEL, GroqProvider

DEFAULT_TIMEOUT_S = 25.0
logger = logging.getLogger("marketingiq.llm_adapter")


def build_provider(env=os.environ) -> GroqProvider:
    legacy = (env.get("LLM_PROVIDER") or "").strip().lower()
    if legacy and legacy != "groq":
        logger.warning(json.dumps({"event": "llm_provider_ignored", "value": legacy[:20],
                                   "reason": "Groq is the only LLM provider"}))
    return GroqProvider(api_key=env.get("GROQ_API_KEY"), model=env.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL)


def timeout_seconds(env=os.environ) -> float:
    return float(env.get("LLM_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_S)
