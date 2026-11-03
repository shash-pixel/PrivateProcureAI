"""Schemas for reusable user preference memory."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field

from privateprocureai.schemas.models import DomainModel


class PreferenceKind(StrEnum):
    """How strongly a remembered preference should influence planning."""

    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"
    AVOID = "avoid"


class PreferenceSource(StrEnum):
    """Where a stored preference originated."""

    EXPLICIT = "explicit"
    LEARNED = "learned"


class PreferenceMemoryItem(DomainModel):
    """One reusable, domain-general preference."""

    preference_id: UUID = Field(default_factory=uuid4)
    value: str = Field(min_length=1, max_length=500)
    kind: PreferenceKind
    source: PreferenceSource
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def normalized_value(self) -> str:
        """Return a deterministic key used for preference conflict detection."""
        return " ".join(self.value.casefold().split())


class UserPreferenceMemory(DomainModel):
    """All stored preference items for one anonymous application profile."""

    profile_id: UUID
    preferences: list[PreferenceMemoryItem] = Field(default_factory=list)
    revision: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))