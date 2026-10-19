"""Tests for document ingestion, cleaning, metadata, and chunking."""

from pathlib import Path

import pytest

from privateprocureai.rag import DocumentIngestor
from privateprocureai.rag.ingestion import (
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from privateprocureai.schemas import ChunkingConfig, DocumentType


def test_ingest_text_creates_clean_traceable_chunks() -> None:
    """Text is cleaned and attached to one shared document record."""
    ingestor = DocumentIngestor(ChunkingConfig(chunk_size=40, chunk_overlap=10))

    result = ingestor.ingest_text(
        "First paragraph.   \n\n\nSecond paragraph with useful information.",
        source_name="example.txt",
        document_type=DocumentType.TEXT,
        attributes={"owner": "test"},
    )

    assert len(result.chunks) >= 2
    assert result.chunks[0].metadata.document_id == result.metadata.document_id
    assert result.chunks[0].metadata.attributes["owner"] == "test"
    assert "\n\n\n" not in result.chunks[0].text


def test_chunks_include_character_overlap() -> None:
    """Adjacent chunks overlap so context is not lost at boundaries."""
    ingestor = DocumentIngestor(ChunkingConfig(chunk_size=30, chunk_overlap=8))

    result = ingestor.ingest_text(
        "One two three four five six seven eight nine ten eleven twelve.",
        source_name="words.txt",
        document_type=DocumentType.TEXT,
    )

    first_chunk, second_chunk = result.chunks[:2]

    assert second_chunk.start_char < first_chunk.end_char


def test_ingest_file_detects_markdown(tmp_path: Path) -> None:
    """Markdown files receive the correct source type metadata."""
    document_path = tmp_path / "notes.md"
    document_path.write_text("# Notes\n\nUseful generic information.", encoding="utf-8")

    result = DocumentIngestor().ingest_file(document_path)

    assert result.metadata.document_type is DocumentType.MARKDOWN
    assert result.metadata.source_name == "notes.md"


def test_ingest_file_rejects_unsupported_format(tmp_path: Path) -> None:
    """Unsupported formats fail clearly instead of being parsed incorrectly."""
    document_path = tmp_path / "data.csv"
    document_path.write_text("column,value", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError):
        DocumentIngestor().ingest_file(document_path)


def test_ingestion_rejects_empty_document() -> None:
    """Whitespace-only sources do not produce empty chunks."""
    with pytest.raises(EmptyDocumentError):
        DocumentIngestor().ingest_text(
            " \n\n  ",
            source_name="empty.txt",
            document_type=DocumentType.TEXT,
        )