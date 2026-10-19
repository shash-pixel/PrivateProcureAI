"""Tests for embedding-independent ChromaDB vector storage."""

from privateprocureai.rag import DocumentIngestor, RagIndexer
from privateprocureai.rag.vector_store import ChromaVectorStore
from privateprocureai.schemas import ChunkingConfig, DocumentType


class FakeEmbeddingProvider:
    """Deterministic embeddings for tests without Gemini API calls."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(index + 1)]
            for index, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_indexer_stores_chunks_and_vector_store_returns_matches() -> None:
    """Chunks can be embedded, persisted in memory, and searched."""
    ingestion_result = DocumentIngestor(
        ChunkingConfig(chunk_size=100, chunk_overlap=10)
    ).ingest_text(
        "Generic information for retrieval testing.",
        source_name="sample.txt",
        document_type=DocumentType.TEXT,
    )

    store = ChromaVectorStore(
        collection_name="test_collection",
        embedding_model="test-embedding-model",
    )

    indexed_count = RagIndexer(FakeEmbeddingProvider(), store).index(
        ingestion_result
    )

    matches = store.search(
        FakeEmbeddingProvider().embed_query("generic retrieval"),
        limit=1,
    )

    assert indexed_count == 1
    assert len(matches) == 1
    assert matches[0].source_name == "sample.txt"


def test_vector_store_rejects_embedding_count_mismatch() -> None:
    """Every stored chunk must have exactly one vector."""
    ingestion_result = DocumentIngestor().ingest_text(
        "One valid document.",
        source_name="sample.txt",
        document_type=DocumentType.TEXT,
    )

    store = ChromaVectorStore(
        collection_name="mismatch_collection",
        embedding_model="test-embedding-model",
    )

    try:
        store.upsert(ingestion_result.chunks, [])
    except Exception as error:
        assert "number of chunks" in str(error)
    else:
        raise AssertionError("Expected an embedding-count validation error.")