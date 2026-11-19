"""Schemas for final recommendation selection and validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from privateprocureai.schemas.budget import BudgetAssessment
from privateprocureai.schemas.intent import ExtractedIntent
from privateprocureai.schemas.models import (
    AgentResult,
    BudgetConstraint,
    DomainModel,
    Recommendation,
)


class FinalRecommendationStatus(StrEnum):
    """Whether a final recommendation is safe to present."""

    READY = "ready"
    BLOCKED = "blocked"


class OptimizationIssueCode(StrEnum):
    """Known reasons final recommendation creation may be blocked."""

    UNKNOWN_RECOMMENDATION = "unknown_recommendation"
    MISSING_COST_ESTIMATE = "missing_cost_estimate"
    HARD_BUDGET_VIOLATION = "hard_budget_violation"
    BUDGET_UNEVALUABLE = "budget_unevaluable"


class OptimizationIssue(DomainModel):
    """A deterministic validation concern from final selection."""

    code: OptimizationIssueCode
    message: str = Field(min_length=3, max_length=2_000)


class OptimizationDecision(DomainModel):
    """Structured LLM output selecting candidates from known recommendations."""

    summary: str = Field(min_length=3, max_length=2_000)
    selected_recommendation_ids: list[UUID] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_recommendations(self) -> Self:
        """Prevent accidental duplicate selection of one recommendation."""
        if len(self.selected_recommendation_ids) != len(set(self.selected_recommendation_ids)):
            raise ValueError("selected_recommendation_ids must not contain duplicates.")

        return self


class FinalOptimizationInput(DomainModel):
    """Inputs required to choose and validate a final recommendation."""

    intent: ExtractedIntent
    agent_results: list[AgentResult] = Field(min_length=1)
    budget: BudgetConstraint | None = None


class FinalRecommendation(DomainModel):
    """Validated final selection ready for future UI, signing, and persistence."""

    status: FinalRecommendationStatus
    summary: str = Field(min_length=3, max_length=2_000)
    selected_recommendations: list[Recommendation] = Field(default_factory=list)
    budget_assessment: BudgetAssessment | None = None
    issues: list[OptimizationIssue] = Field(default_factory=list)
    requires_replanning: bool
