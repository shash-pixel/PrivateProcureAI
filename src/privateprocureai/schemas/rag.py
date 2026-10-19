"""Schemas for document ingestion and future retrieval."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from privateprocureai.schemas.models import DomainModel


class DocumentType(StrEnum):
    """Supported document formats for the initial ingestion pipeline."""

    TEXT = "text"
    MARKDOWN = "markdown"


class ChunkingConfig(DomainModel):
    """Settings controlling deterministic character-based text chunking."""

    chunk_size: int = Field(default=1_000, ge=5, le=10_000)
    chunk_overlap: int = Field(default=150, ge=0, le=2_000)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        """Prevent an overlap that would stop chunking progress."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        return self


class DocumentMetadata(DomainModel):
    """Traceability information for one ingested source document."""

    document_id: UUID = Field(default_factory=uuid4)
    source_name: str = Field(min_length=1, max_length=500)
    document_type: DocumentType
    source_uri: str | None = Field(default=None, max_length=2_000)
    attributes: dict[str, str] = Field(default_factory=dict)


class DocumentChunk(DomainModel):
    """A clean, bounded text fragment ready for future embedding."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    metadata: DocumentMetadata


class IngestionResult(DomainModel):
    """The document record and all chunks created from one source."""

    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(min_length=1)