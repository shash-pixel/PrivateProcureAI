"""Indexing service that joins ingestion, embeddings, and vector storage."""

from __future__ import annotations

from privateprocureai.rag.embeddings import EmbeddingProvider
from privateprocureai.rag.vector_store import VectorStore
from privateprocureai.schemas import IngestionResult


class RagIndexer:
    """Create embeddings for ingested chunks and store them in a vector database."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def index(self, ingestion_result: IngestionResult) -> int:
        """Embed and store every chunk in one ingestion result."""
        chunks = ingestion_result.chunks
        embeddings = self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )

        self._vector_store.upsert(chunks, embeddings)

        return len(chunks)