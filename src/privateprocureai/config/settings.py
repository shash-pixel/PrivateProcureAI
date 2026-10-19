"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LLMProvider = Literal["mock", "openai", "gemini"]


class AppSettings(BaseSettings):
    """Runtime settings for PrivateProcureAI.

    Values are loaded from environment variables prefixed with ``PP_AI_``.
    A local .env file may provide development-only values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PP_AI_",
        extra="ignore",
    )

    environment: Environment = "development"
    log_level: LogLevel = "INFO"

    llm_provider: LLMProvider = "mock"
    llm_model: str = Field(default="mock-planner-v1",min_length=1,max_length=200)
    llm_api_key: SecretStr | None = None

    request_timeout_seconds: int = Field(default=30, ge=1, le=300)

    embedding_model: str = Field(
    default="gemini-embedding-2",
    min_length=1,
    max_length=200,
)

    chroma_persist_directory: Path = Path("data/chroma")
    chroma_collection_name: str = Field(
    default="privateprocureai_chunks",
    min_length=3,
    max_length=100,
)

    @property
    def uses_remote_llm(self) -> bool:
        """Return whether the chosen provider makes external API calls."""
        return self.llm_provider != "mock"


@lru_cache
def get_settings() -> AppSettings:
    """Return one validated settings object for the current process."""
    return AppSettings()