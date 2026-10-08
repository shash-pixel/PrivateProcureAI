"""Tests for structured LLM provider abstractions."""

import pytest
from pydantic import BaseModel, Field

from privateprocureai.llm import MockLLMProvider, StructuredGenerationRequest
from privateprocureai.llm.exceptions import LLMProviderError, StructuredOutputError


class ExampleStructuredResponse(BaseModel):
    """A generic schema used only to test structured output behavior."""

    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


@pytest.fixture
def generation_request() -> StructuredGenerationRequest:
    """Create a provider-neutral request for tests."""
    return StructuredGenerationRequest(
        system_instruction="Return a response that follows the requested schema.",
        user_prompt="Generate one generic example.",
    )


def test_mock_provider_returns_validated_structured_output(
    generation_request: StructuredGenerationRequest,
) -> None:
    """Mock output becomes an instance of the requested Pydantic model."""
    provider = MockLLMProvider(
        responses=[
            {
                "label": "example",
                "confidence": 0.8,
            }
        ]
    )

    result = provider.generate_structured(
        generation_request,
        ExampleStructuredResponse,
    )

    assert result.label == "example"
    assert result.confidence == 0.8


def test_mock_provider_rejects_invalid_structured_output(
    generation_request: StructuredGenerationRequest,
) -> None:
    """Malformed provider output cannot enter the application unchecked."""
    provider = MockLLMProvider(
        responses=[
            {
                "label": "example",
                "confidence": 1.5,
            }
        ]
    )

    with pytest.raises(StructuredOutputError):
        provider.generate_structured(
            generation_request,
            ExampleStructuredResponse,
        )


def test_mock_provider_requires_a_queued_response(
    generation_request: StructuredGenerationRequest,
) -> None:
    """Tests fail clearly if they forget to define simulated model output."""
    provider = MockLLMProvider()

    with pytest.raises(LLMProviderError, match="no queued response"):
        provider.generate_structured(
            generation_request,
            ExampleStructuredResponse,
        )