"""Deterministic ingestion, cleaning, metadata creation, and chunking."""

from __future__ import annotations

import re
from pathlib import Path

from privateprocureai.schemas import (
    ChunkingConfig,
    DocumentChunk,
    DocumentMetadata,
    DocumentType,
    IngestionResult,
)


class DocumentIngestionError(ValueError):
    """Base exception for invalid document-ingestion input."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when a source file is not supported by this module."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when cleaning leaves a document with no usable text."""


class DocumentIngestor:
    """Convert supported local documents into clean, overlapping text chunks."""

    _SUPPORTED_SUFFIXES = {
        ".txt": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
    }

    def __init__(self, chunking: ChunkingConfig | None = None) -> None:
        self._chunking = chunking or ChunkingConfig()

    def ingest_file(
        self,
        path: Path,
        *,
        attributes: dict[str, str] | None = None,
    ) -> IngestionResult:
        """Read a UTF-8 text or Markdown file and create traceable chunks."""
        if not path.is_file():
            raise DocumentIngestionError(f"Document does not exist: {path}")

        document_type = self._SUPPORTED_SUFFIXES.get(path.suffix.lower())

        if document_type is None:
            supported = ", ".join(sorted(self._SUPPORTED_SUFFIXES))
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{path.suffix}'. Supported types: {supported}."
            )

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise DocumentIngestionError(
                f"Document must use UTF-8 encoding: {path}"
            ) from error

        return self.ingest_text(
            raw_text,
            source_name=path.name,
            document_type=document_type,
            source_uri=str(path.resolve()),
            attributes=attributes,
        )

    def ingest_text(
        self,
        raw_text: str,
        *,
        source_name: str,
        document_type: DocumentType,
        source_uri: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> IngestionResult:
        """Clean supplied text and produce deterministic overlapping chunks."""
        cleaned_text = self._clean_text(raw_text)

        if not cleaned_text:
            raise EmptyDocumentError(
                "Document contains no usable text after cleaning."
            )

        metadata = DocumentMetadata(
            source_name=source_name,
            document_type=document_type,
            source_uri=source_uri,
            attributes=attributes or {},
        )

        chunks = self._chunk_text(cleaned_text, metadata)

        return IngestionResult(
            metadata=metadata,
            chunks=chunks,
        )

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Normalize line endings, remove trailing whitespace, and trim blanks."""
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()

    def _chunk_text(
        self,
        text: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """Split text near paragraph or word boundaries with configured overlap."""
        chunks: list[DocumentChunk] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            maximum_end = min(start + self._chunking.chunk_size, text_length)
            end = maximum_end

            if maximum_end < text_length:
                paragraph_boundary = text.rfind("\n\n", start + 1, maximum_end + 1)
                word_boundary = text.rfind(" ", start + 1, maximum_end + 1)

                if paragraph_boundary > start:
                    end = paragraph_boundary
                elif word_boundary > start:
                    end = word_boundary

            raw_chunk = text[start:end]
            stripped_chunk = raw_chunk.strip()

            if not stripped_chunk:
                end = min(start + self._chunking.chunk_size, text_length)
                stripped_chunk = text[start:end].strip()

            leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
            chunk_start = start + leading_whitespace
            chunk_end = chunk_start + len(stripped_chunk)

            chunks.append(
                DocumentChunk(
                    document_id=metadata.document_id,
                    chunk_index=len(chunks),
                    text=stripped_chunk,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=metadata,
                )
            )

            if end >= text_length:
                break

            start = max(end - self._chunking.chunk_overlap, start + 1)

        return chunks