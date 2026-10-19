"""Schemas for vector indexing and similarity-search results."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.models import DomainModel


class VectorSearchMatch(DomainModel):
    """One chunk returned by vector similarity search."""

    chunk_id: UUID
    document_id: UUID
    text: str = Field(min_length=1)
    distance: float = Field(ge=0.0)
    source_name: str = Field(min_length=1)
    source_uri: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)