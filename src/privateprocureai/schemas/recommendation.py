"""Schemas for structured recommendation-agent output."""

from __future__ import annotations

from pydantic import Field

from privateprocureai.schemas.models import DomainModel, Recommendation


class RecommendationAgentOutput(DomainModel):
    """Candidate recommendations grounded in supplied evidence."""

    summary: str = Field(min_length=3, max_length=2_000)
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)