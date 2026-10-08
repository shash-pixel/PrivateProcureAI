"""Deterministic mock provider for local development and tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from privateprocureai.llm.base import (
    ResponseModelT,
    StructuredGenerationRequest,
    StructuredLLM,
)
from privateprocureai.llm.exceptions import LLMProviderError, StructuredOutputError


class MockLLMProvider(StructuredLLM):
    """Return queued payloads after validating them against requested schemas.

    The mock never calls an external API. Tests inject payloads representing
    simulated LLM output, then receive the same validation guarantees as a
    real provider adapter.
    """

    def __init__(self, responses: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._responses: deque[Mapping[str, Any]] = deque(responses or [])

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Validate the next queued payload as the requested response model."""
        del request  # The mock is deterministic and does not interpret prompts.

        if not self._responses:
            raise LLMProviderError(
                "MockLLMProvider has no queued response. "
                "Provide a payload when constructing the provider."
            )

        payload = self._responses.popleft()

        try:
            return response_model.model_validate(payload)
        except ValidationError as error:
            raise StructuredOutputError(
                f"Mock response did not match {response_model.__name__}."
            ) from error