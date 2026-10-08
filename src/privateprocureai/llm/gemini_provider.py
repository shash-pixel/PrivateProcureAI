"""Gemini adapter isolated behind the provider-neutral LLM interface."""

from __future__ import annotations

from pydantic import SecretStr

from privateprocureai.llm.base import (
    ResponseModelT,
    StructuredGenerationRequest,
    StructuredLLM,
)
from privateprocureai.llm.exceptions import LLMProviderError, StructuredOutputError


class GeminiProvider(StructuredLLM):
    """Generate Pydantic-validated structured output through Gemini."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        timeout_seconds: int,
    ) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._types = types
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Request JSON matching the requested Pydantic schema."""
        config_options: dict[str, object] = {
            "system_instruction": request.system_instruction,
            "response_mime_type": "application/json",
            "response_schema": response_model,
        }

        if request.temperature is not None:
            config_options["temperature"] = request.temperature

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=request.user_prompt,
                config=self._types.GenerateContentConfig(**config_options),
            )
        except Exception as error:
            raise LLMProviderError("Gemini structured generation request failed.") from error

        if not response.text:
            raise StructuredOutputError(
                "Gemini returned no text for the requested structured response."
            )

        try:
            return response_model.model_validate_json(response.text)
        except Exception as error:
            raise StructuredOutputError(
                f"Gemini output did not match {response_model.__name__}."
            ) from error