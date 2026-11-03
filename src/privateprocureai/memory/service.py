"""Conflict-aware preference memory management."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from privateprocureai.memory.repository import PreferenceMemoryRepository
from privateprocureai.schemas import (
    PlanningRequest,
    PreferenceKind,
    PreferenceMemoryItem,
    PreferenceSource,
    UserPreferenceMemory,
    UserPreferences,
)


class PreferenceMemoryService:
    """Store explicit preferences and merge memory into new planning requests."""

    def __init__(self, repository: PreferenceMemoryRepository) -> None:
        self._repository = repository

    def remember_explicit_preferences(
        self,
        *,
        profile_id: UUID,
        preferences: UserPreferences,
    ) -> UserPreferenceMemory:
        """Save current explicit preferences, replacing older conflicting entries."""
        existing = self._repository.get(profile_id) or UserPreferenceMemory(
            profile_id=profile_id
        )

        items_by_value = {
            item.normalized_value: item
            for item in existing.preferences
        }

        for value, kind in self._iter_preferences(preferences):
            item = PreferenceMemoryItem(
                value=value,
                kind=kind,
                source=PreferenceSource.EXPLICIT,
                confidence_score=1.0,
            )
            items_by_value[item.normalized_value] = item

        updated_memory = UserPreferenceMemory(
            profile_id=profile_id,
            preferences=list(items_by_value.values()),
            revision=existing.revision + 1,
            updated_at=datetime.now(UTC),
        )

        self._repository.save(updated_memory)

        return updated_memory

    def enrich_request(
        self,
        *,
        profile_id: UUID,
        planning_request: PlanningRequest,
    ) -> PlanningRequest:
        """Apply stored preferences while preserving current-request precedence."""
        memory = self._repository.get(profile_id)

        if memory is None:
            return planning_request

        effective_preferences = self._merge_preferences(
            current=planning_request.preferences,
            remembered=memory,
        )

        return planning_request.model_copy(
            update={"preferences": effective_preferences}
        )

    @staticmethod
    def _merge_preferences(
        *,
        current: UserPreferences,
        remembered: UserPreferenceMemory,
    ) -> UserPreferences:
        """Merge memory, skipping every value explicitly mentioned in this request."""
        current_values = {
            " ".join(value.casefold().split())
            for values in (
                current.must_have,
                current.nice_to_have,
                current.avoid,
            )
            for value in values
        }

        must_have = list(current.must_have)
        nice_to_have = list(current.nice_to_have)
        avoid = list(current.avoid)

        for item in remembered.preferences:
            if item.normalized_value in current_values:
                continue

            if item.kind is PreferenceKind.MUST_HAVE:
                must_have.append(item.value)
            elif item.kind is PreferenceKind.NICE_TO_HAVE:
                nice_to_have.append(item.value)
            else:
                avoid.append(item.value)

        return UserPreferences(
            must_have=PreferenceMemoryService._unique_values(must_have),
            nice_to_have=PreferenceMemoryService._unique_values(nice_to_have),
            avoid=PreferenceMemoryService._unique_values(avoid),
            notes=current.notes,
        )

    @staticmethod
    def _iter_preferences(
        preferences: UserPreferences,
    ):
        """Yield every current preference with its corresponding kind."""
        for value in preferences.must_have:
            yield value, PreferenceKind.MUST_HAVE

        for value in preferences.nice_to_have:
            yield value, PreferenceKind.NICE_TO_HAVE

        for value in preferences.avoid:
            yield value, PreferenceKind.AVOID

    @staticmethod
    def _unique_values(values: list[str]) -> list[str]:
        """Remove duplicate display values while preserving first-seen order."""
        seen: set[str] = set()
        unique_values: list[str] = []

        for value in values:
            normalized = " ".join(value.casefold().split())

            if normalized not in seen:
                seen.add(normalized)
                unique_values.append(value)

        return unique_values