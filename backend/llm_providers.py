"""
Explicit LLM provider selection for the LLM Adapter.

    LLM_PROVIDER=gemini   (default — the existing behaviour)
    LLM_PROVIDER=qwen     local Qwen through Ollama (OLLAMA_HOST, OLLAMA_MODEL)
    LLM_PROVIDER=groq     hosted Groq API (GROQ_API_KEY, GROQ_MODEL)

There is deliberately no automatic failover between providers: exactly one provider
answers every LLM_REQUIRED question, and an unknown LLM_PROVIDER value stops startup
instead of silently choosing something else.
"""
import os

PROVIDERS = ("gemini", "qwen", "groq")
DEFAULT_PROVIDER = "gemini"
# Explanation deadline when LLM_TIMEOUT_SECONDS is not set. Local CPU inference (including a
# cold model load) needs longer than a hosted API.
DEFAULT_TIMEOUT_S = {"gemini": 25.0, "qwen": 90.0, "groq": 25.0}


class ProviderConfigError(ValueError):
    pass


def selected_provider(env=os.environ) -> str:
    name = (env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ProviderConfigError(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got '{name}').")
    return name


def timeout_seconds(provider_name: str, env=os.environ) -> float:
    return float(env.get("LLM_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_S[provider_name])


def build_provider(name: str, gemini_rotator=None, gemini_model: str = None, env=os.environ):
    if name == "gemini":
        from gemini_provider import GeminiProvider
        return GeminiProvider(gemini_rotator, gemini_model)
    if name == "qwen":
        from qwen_provider import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, QwenProvider
        return QwenProvider(host=env.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST,
                            model=env.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL)
    if name == "groq":
        from groq_provider import DEFAULT_GROQ_MODEL, GroqProvider
        # A missing key doesn't stop startup: explanations report not_configured, like Gemini.
        return GroqProvider(api_key=env.get("GROQ_API_KEY"), model=env.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL)
    raise ProviderConfigError(f"Unknown LLM provider '{name}'.")
