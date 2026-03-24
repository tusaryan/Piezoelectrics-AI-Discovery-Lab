"""
Model-agnostic LLM provider for Piezo.AI Agent.

Uses LangChain's init_chat_model to support any provider:
  - openai (GPT-4o, GPT-4-turbo, etc.)
  - anthropic (Claude Sonnet, Opus, etc.)
  - google (Gemini 2.0 Flash, Pro, etc.)
  - ollama (Gemma, Qwen, DeepSeek, any local model)

Configured entirely via environment variables — no code changes to switch providers.
"""
import logging
from functools import lru_cache

from apps.api.app.core.config import settings

logger = logging.getLogger("piezo.agent.llm_provider")

# Provider → required LangChain package
_PROVIDER_PACKAGES = {
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google": "langchain-google-genai",
    "ollama": "langchain-ollama",
}


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Initialize a chat model from env vars using init_chat_model.
    
    Returns a LangChain BaseChatModel that works identically regardless of provider.
    Cached so the same instance is reused across requests.
    """
    provider = settings.llm_provider.lower()
    if provider == "google":
        provider = "google_genai"
        
    model = settings.llm_model

    logger.info(
        "llm_provider.init",
        extra={"provider": provider, "model": model},
    )

    if not settings.llm_api_key and provider != "ollama":
        raise RuntimeError(
            f"LLM_API_KEY is required for provider '{provider}'. "
            f"Set it in your .env file or environment variables."
        )

    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.globals import set_llm_cache
        from langchain_core.caches import InMemoryCache
        
        # Use an in-memory cache to prevent duplicate LLM calls
        set_llm_cache(InMemoryCache())
    except ImportError:
        raise RuntimeError(
            "langchain is not installed. Install with: pip install langchain"
        )

    # Build kwargs
    kwargs = {
        "model": model,
        "model_provider": provider,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }

    # API key
    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key

    # Custom base URL (used by ollama and custom deployments)
    if settings.llm_base_url:
        cleaned_url = str(settings.llm_base_url).split("#")[0].strip()
        if cleaned_url and cleaned_url.startswith("http"):
            kwargs["base_url"] = cleaned_url

    try:
        llm = init_chat_model(**kwargs)
        logger.info("llm_provider.ready", extra={"provider": provider, "model": model})
        return llm
    except Exception as e:
        pkg = _PROVIDER_PACKAGES.get(provider, "unknown")
        logger.error("llm_provider.init_failed", extra={"error": str(e)})
        raise RuntimeError(
            f"Failed to initialize LLM provider '{provider}' with model '{model}'. "
            f"Make sure '{pkg}' is installed and the API key is valid. Error: {e}"
        ) from e


def get_provider_info() -> dict:
    """Return current LLM provider configuration (for debugging/display)."""
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "has_api_key": bool(settings.llm_api_key),
        "base_url": settings.llm_base_url,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
