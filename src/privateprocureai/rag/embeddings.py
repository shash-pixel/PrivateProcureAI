"""Provider-neutral embeddings interface and Gemini implementation."""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot create valid vectors."""


class EmbeddingProvider(Protocol):
    """Contract implemented by text embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding vector for every document text."""

    def embed_query(self, text: str) -> list[float]:
        """Create one embedding vector for a similarity-search query."""


class GeminiEmbeddingProvider:
    """Create text embeddings through Gemini's official Python SDK."""

    def __init__(self, *, api_key: SecretStr, model: str) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._types = types
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding per supplied document chunk."""
        if not texts:
            return []

        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Create an embedding for one non-empty search query."""
        if not text.strip():
            raise EmbeddingProviderError("A query cannot be empty.")

        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call Gemini and validate the returned vector count."""
        contents = [
            self._types.Content(
                parts=[self._types.Part.from_text(text=text)]
            )
            for text in texts
        ]

        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=contents,
            )
        except Exception as error:
            raise EmbeddingProviderError("Gemini embedding request failed.") from error

        vectors = [
            list(embedding.values)
            for embedding in response.embeddings
        ]

        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "Gemini returned a different number of embeddings than requested."
            )

        if not all(vector for vector in vectors):
            raise EmbeddingProviderError("Gemini returned an empty embedding vector.")

        return vectors