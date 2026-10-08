"""Tests for application configuration."""

import pytest
from pydantic import ValidationError

from privateprocureai.config.settings import AppSettings


def test_settings_use_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The project can run locally with a mock provider and no API key."""
    for variable_name in (
        "PP_AI_ENVIRONMENT",
        "PP_AI_LOG_LEVEL",
        "PP_AI_LLM_PROVIDER",
        "PP_AI_LLM_MODEL",
        "PP_AI_LLM_API_KEY",
        "PP_AI_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.environment == "development"
    assert settings.llm_provider == "mock"
    assert settings.uses_remote_llm is False
    assert settings.llm_api_key is None


def test_settings_read_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefixed environment variables override safe defaults."""
    monkeypatch.setenv("PP_AI_LLM_PROVIDER", "openai")
    monkeypatch.setenv("PP_AI_LLM_MODEL", "gpt-example")
    monkeypatch.setenv("PP_AI_REQUEST_TIMEOUT_SECONDS", "45")

    settings = AppSettings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-example"
    assert settings.request_timeout_seconds == 45
    assert settings.uses_remote_llm is True


def test_settings_reject_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid timeout is rejected during configuration loading."""
    monkeypatch.setenv("PP_AI_REQUEST_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_settings_reject_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only explicitly supported LLM providers are allowed."""
    monkeypatch.setenv("PP_AI_LLM_PROVIDER", "unknown-provider")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)