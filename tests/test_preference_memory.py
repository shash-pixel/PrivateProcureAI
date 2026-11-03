"""Tests for local, conflict-aware user preference memory."""

from uuid import uuid4

from privateprocureai.memory import (
    JsonPreferenceMemoryRepository,
    PreferenceMemoryService,
)
from privateprocureai.schemas import (
    PlanningRequest,
    PreferenceKind,
    PreferenceMemoryItem,
    PreferenceSource,
    UserPreferenceMemory,
    UserPreferences,
)


def test_explicit_preferences_are_saved_and_reloaded(tmp_path) -> None:
    """Explicit preferences persist through the local JSON repository."""
    repository = JsonPreferenceMemoryRepository(tmp_path / "preferences.json")
    service = PreferenceMemoryService(repository)
    profile_id = uuid4()

    saved_memory = service.remember_explicit_preferences(
        profile_id=profile_id,
        preferences=UserPreferences(
            must_have=["Accessible documentation"],
            avoid=["Unverified sources"],
        ),
    )

    loaded_memory = repository.get(profile_id)

    assert loaded_memory is not None
    assert loaded_memory.revision == 1
    assert len(saved_memory.preferences) == 2
    assert loaded_memory.preferences[0].source is PreferenceSource.EXPLICIT


def test_current_request_preference_overrides_saved_memory(tmp_path) -> None:
    """Current user instructions override conflicting remembered preferences."""
    repository = JsonPreferenceMemoryRepository(tmp_path / "preferences.json")
    service = PreferenceMemoryService(repository)
    profile_id = uuid4()

    repository.save(
        UserPreferenceMemory(
            profile_id=profile_id,
            preferences=[
                PreferenceMemoryItem(
                    value="Low maintenance",
                    kind=PreferenceKind.NICE_TO_HAVE,
                    source=PreferenceSource.LEARNED,
                    confidence_score=0.7,
                )
            ],
        )
    )

    enriched_request = service.enrich_request(
        profile_id=profile_id,
        planning_request=PlanningRequest(
            goal="Create a plan using the current requirements.",
            preferences=UserPreferences(
                avoid=["low maintenance"],
            ),
        ),
    )

    assert enriched_request.preferences.avoid == ["low maintenance"]
    assert enriched_request.preferences.nice_to_have == []


def test_memory_adds_non_conflicting_preferences(tmp_path) -> None:
    """Remembered preferences enrich a request only when not contradicted."""
    repository = JsonPreferenceMemoryRepository(tmp_path / "preferences.json")
    service = PreferenceMemoryService(repository)
    profile_id = uuid4()

    service.remember_explicit_preferences(
        profile_id=profile_id,
        preferences=UserPreferences(
            nice_to_have=["Clear evidence trail"],
        ),
    )

    enriched_request = service.enrich_request(
        profile_id=profile_id,
        planning_request=PlanningRequest(
            goal="Prepare a supported plan for the stated objective."
        ),
    )

    assert enriched_request.preferences.nice_to_have == [
        "Clear evidence trail"
    ]