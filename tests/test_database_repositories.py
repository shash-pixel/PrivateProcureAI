"""Tests for SQLAlchemy-backed persistence repositories."""

from uuid import uuid4

from sqlalchemy import create_engine

from privateprocureai.database import (
    PostgresPreferenceMemoryRepository,
    PostgresSignedRecommendationRepository,
    create_session_factory,
    initialize_database,
)
from privateprocureai.schemas import (
    FinalRecommendation,
    FinalRecommendationStatus,
    IntegrityRecord,
    PreferenceKind,
    PreferenceMemoryItem,
    PreferenceSource,
    SignedFinalRecommendation,
    UserPreferenceMemory,
)


def make_session_factory():
    """Create an isolated in-memory database for repository tests."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    initialize_database(engine)

    return create_session_factory(engine)


def test_preference_memory_repository_round_trip() -> None:
    """Preference memory persists and reloads through the repository contract."""
    session_factory = make_session_factory()
    repository = PostgresPreferenceMemoryRepository(session_factory)
    profile_id = uuid4()

    repository.save(
        UserPreferenceMemory(
            profile_id=profile_id,
            revision=1,
            preferences=[
                PreferenceMemoryItem(
                    value="Traceable evidence",
                    kind=PreferenceKind.MUST_HAVE,
                    source=PreferenceSource.EXPLICIT,
                    confidence_score=1.0,
                )
            ],
        )
    )

    loaded_memory = repository.get(profile_id)

    assert loaded_memory is not None
    assert loaded_memory.revision == 1
    assert loaded_memory.preferences[0].value == "Traceable evidence"


def test_signed_recommendation_repository_preserves_integrity_payload() -> None:
    """Signed final results can be stored and loaded without modification."""
    session_factory = make_session_factory()
    repository = PostgresSignedRecommendationRepository(session_factory)
    request_id = uuid4()

    signed_recommendation = SignedFinalRecommendation(
        recommendation=FinalRecommendation(
            status=FinalRecommendationStatus.READY,
            summary="A validated final recommendation.",
            selected_recommendations=[],
            requires_replanning=False,
        ),
        integrity=IntegrityRecord(
            payload_sha256="a" * 64,
            signature_base64="test-signature",
            public_key_fingerprint_sha256="b" * 64,
        ),
    )

    result_id = repository.save(
        request_id=request_id,
        signed_recommendation=signed_recommendation,
    )

    loaded_recommendation = repository.get(result_id)

    assert loaded_recommendation == signed_recommendation
    assert repository.list_result_ids(request_id) == [result_id]
