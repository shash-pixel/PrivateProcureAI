"""Vector retrieval and conversion of matches into agent-ready evidence."""

from __future__ import annotations

from typing import Protocol

from privateprocureai.rag.embeddings import EmbeddingProvider
from privateprocureai.rag.vector_store import VectorStore
from privateprocureai.schemas import (
    Evidence,
    ExtractedIntent,
    PlanTask,
    RetrievalConfig,
    RetrievalResult,
)


class EvidenceRetriever(Protocol):
    """Contract used by the agent executor to obtain relevant evidence."""

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant evidence for a generic text query."""


class RagRetriever:
    """Retrieve, filter, and normalize vector-search matches into Evidence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        """Embed a query, retrieve matches, and discard weak evidence."""
        if not query.strip():
            raise ValueError("A retrieval query cannot be empty.")

        query_embedding = self._embedding_provider.embed_query(query)

        matches = self._vector_store.search(
            query_embedding,
            limit=self._config.candidate_limit,
        )

        evidence: list[Evidence] = []
        discarded_match_count = 0

        for match in matches:
            relevance_score = self._distance_to_relevance(match.distance)

            if relevance_score < self._config.minimum_relevance_score:
                discarded_match_count += 1
                continue

            evidence.append(
                Evidence(
                    source_id=str(match.chunk_id),
                    source_url=match.source_uri,
                    excerpt=match.text,
                    relevance_score=relevance_score,
                )
            )

        return RetrievalResult(
            query=query,
            evidence=evidence,
            discarded_match_count=discarded_match_count,
        )

    @staticmethod
    def _distance_to_relevance(distance: float) -> float:
        """Convert Chroma cosine distance to an approximate 0-to-1 relevance score."""
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


def build_retrieval_query(
    intent: ExtractedIntent,
    task: PlanTask,
) -> str:
    """Create a generic retrieval query from validated planning state."""
    constraints = "\n".join(
        f"- {constraint.description}"
        for constraint in intent.constraints
    ) or "- No explicit constraints provided."

    preferences = "\n".join(
        f"- {preference}"
        for preference in intent.preferences
    ) or "- No explicit preferences provided."

    return (
        f"Objective:\n{intent.objective}\n\n"
        f"Task:\n{task.title}\n{task.instructions}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Preferences:\n{preferences}"
    )