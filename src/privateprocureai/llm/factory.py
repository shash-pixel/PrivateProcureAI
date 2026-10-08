"""Construction of the configured LLM provider adapter."""

from __future__ import annotations

from privateprocureai.config import AppSettings, get_settings
from privateprocureai.llm.base import StructuredLLM
from privateprocureai.llm.exceptions import (
    LLMProviderError,
    ProviderNotImplementedError,
)
from privateprocureai.llm.gemini_provider import GeminiProvider
from privateprocureai.llm.mock import MockLLMProvider


def create_llm_provider(settings: AppSettings | None = None) -> StructuredLLM:
    """Create the provider selected by validated application settings."""
    active_settings = settings or get_settings()

    if active_settings.llm_provider == "mock":
        return MockLLMProvider()

    if active_settings.llm_provider == "gemini":
        if active_settings.llm_api_key is None:
            raise LLMProviderError(
                "PP_AI_LLM_API_KEY is required when PP_AI_LLM_PROVIDER=gemini."
            )

        return GeminiProvider(
            api_key=active_settings.llm_api_key,
            model=active_settings.llm_model,
            timeout_seconds=active_settings.request_timeout_seconds,
        )

    if active_settings.llm_provider == "openai":
        raise ProviderNotImplementedError(
            "The OpenAI adapter is not implemented. Gemini is the current runtime provider."
        )

    raise ProviderNotImplementedError(
        f"Unsupported LLM provider: {active_settings.llm_provider}"
    )