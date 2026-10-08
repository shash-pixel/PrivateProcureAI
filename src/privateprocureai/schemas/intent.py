"""Domain-general schemas for extracted user intent."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from privateprocureai.schemas.models import DomainModel


class ConstraintStrength(StrEnum):
    """Whether a constraint is mandatory or a preference."""

    HARD = "hard"
    SOFT = "soft"


class IntentSource(StrEnum):
    """Whether information was stated directly or inferred by the model."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"


class IntentConstraint(DomainModel):
    """A generic restriction or preference identified in a user goal."""

    description: str = Field(min_length=3, max_length=1_000)
    strength: ConstraintStrength
    source: IntentSource


class IntentDetails(DomainModel):
    """The LLM-generated interpretation of a user's planning goal."""

    objective: str = Field(min_length=3, max_length=2_000)
    constraints: list[IntentConstraint] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_clarification_state(self) -> Self:
        """Keep clarification state and questions consistent."""
        if self.clarification_needed and not self.clarification_questions:
            raise ValueError(
                "Clarification questions are required when clarification is needed."
            )

        if not self.clarification_needed and self.clarification_questions:
            raise ValueError(
                "Clarification questions must be empty when clarification is not needed."
            )

        return self


class ExtractedIntent(IntentDetails):
    """Intent details associated with the original planning request."""

    request_id: UUID