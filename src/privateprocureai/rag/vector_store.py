"""ChromaDB-backed local vector storage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import UUID

from privateprocureai.schemas import DocumentChunk, VectorSearchMatch


class VectorStoreError(RuntimeError):
    """Raised when vector storage cannot safely complete an operation."""


class VectorStore(Protocol):
    """Contract for storage engines that index and search chunk embeddings."""

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store chunks with their precomputed embeddings."""

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorSearchMatch]:
        """Return the nearest stored chunks."""


class ChromaVectorStore:
    """Persistent local ChromaDB storage with application-owned embeddings."""

    def __init__(
        self,
        *,
        collection_name: str,
        embedding_model: str,
        persist_directory: Path | None = None,
    ) -> None:
        import chromadb

        if persist_directory is None:
            self._client = chromadb.EphemeralClient()
        else:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={
                "embedding_model": embedding_model,
                "hnsw:space": "cosine",
            },
        )

        existing_model = (self._collection.metadata or {}).get("embedding_model")

        if existing_model != embedding_model:
            raise VectorStoreError(
                "This collection was created with a different embedding model. "
                "Use a new collection name or remove the old local collection."
            )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update chunks with explicit Gemini-generated embeddings."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "The number of chunks must equal the number of embeddings."
            )

        if not chunks:
            return

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) != 1 or 0 in dimensions:
            raise VectorStoreError(
                "All embeddings must be non-empty vectors with the same dimension."
            )

        self._collection.upsert(
            ids=[str(chunk.chunk_id) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[self._to_chroma_metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorSearchMatch]:
        """Search stored chunks using an explicit query embedding."""
        if not query_embedding:
            raise VectorStoreError("A query embedding cannot be empty.")

        if limit < 1:
            raise VectorStoreError("limit must be at least 1.")

        count = self._collection.count()

        if count == 0:
            return []

        response = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        return [
            VectorSearchMatch(
                chunk_id=UUID(chunk_id),
                document_id=UUID(metadata["document_id"]),
                text=document,
                distance=float(distance),
                source_name=metadata["source_name"],
                source_uri=metadata.get("source_uri"),
                attributes={
                    key.removeprefix("attribute_"): value
                    for key, value in metadata.items()
                    if key.startswith("attribute_")
                },
            )
            for chunk_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

    @staticmethod
    def _to_chroma_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        """Flatten project metadata into Chroma-compatible primitive values."""
        metadata: dict[str, str | int] = {
            "document_id": str(chunk.document_id),
            "source_name": chunk.metadata.source_name,
            "document_type": chunk.metadata.document_type.value,
            "chunk_index": chunk.chunk_index,
        }

        if chunk.metadata.source_uri is not None:
            metadata["source_uri"] = chunk.metadata.source_uri

        for key, value in chunk.metadata.attributes.items():
            metadata[f"attribute_{key}"] = value

        return metadata