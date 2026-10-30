"""Schemas for RAG retrieval configuration and results."""

from __future__ import annotations

from pydantic import Field

from privateprocureai.schemas.models import DomainModel, Evidence


class RetrievalConfig(DomainModel):
    """Settings that control vector-search breadth and relevance filtering."""

    candidate_limit: int = Field(default=8, ge=1, le=50)
    minimum_relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class RetrievalResult(DomainModel):
    """Validated evidence returned for one retrieval query."""

    query: str = Field(min_length=1, max_length=10_000)
    evidence: list[Evidence] = Field(default_factory=list)
    discarded_match_count: int = Field(ge=0)