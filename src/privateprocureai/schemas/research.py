"""Schemas for evidence-grounded research agent output."""

from __future__ import annotations

from pydantic import Field

from privateprocureai.schemas.models import DomainModel, Finding


class ResearchAgentOutput(DomainModel):
    """Structured LLM output for one evidence-grounded research task."""

    summary: str = Field(min_length=3, max_length=2_000)
    findings: list[Finding] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)