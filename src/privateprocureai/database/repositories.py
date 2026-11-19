"""PostgreSQL-backed repositories for memory and signed recommendations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from privateprocureai.database.exceptions import DatabaseUnavailableError
from privateprocureai.database.models import (
    PreferenceMemoryRecord,
    SignedRecommendationRecord,
)
from privateprocureai.observability import redact_secrets, timed_step
from privateprocureai.schemas import (
    SignedFinalRecommendation,
    UserPreferenceMemory,
)


class PostgresPreferenceMemoryRepository:
    """Replace the local JSON preference repository with database persistence."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get(self, profile_id: UUID) -> UserPreferenceMemory | None:
        """Load stored preference memory for one profile."""
        with timed_step("persistence"):
            try:
                with self._session_factory() as session:
                    record = session.get(PreferenceMemoryRecord, profile_id)

                    if record is None:
                        return None

                    return UserPreferenceMemory(
                        profile_id=record.profile_id,
                        preferences=record.preferences,
                        revision=record.revision,
                        updated_at=record.updated_at,
                    )
            except SQLAlchemyError as error:
                safe_msg = redact_secrets(str(error))
                raise DatabaseUnavailableError(
                    f"Failed to load preference memory: {safe_msg}"
                ) from error

    def save(self, memory: UserPreferenceMemory) -> None:
        """Insert or update all preference memory for one profile."""
        with timed_step("persistence"):
            try:
                with self._session_factory.begin() as session:
                    record = session.get(
                        PreferenceMemoryRecord,
                        memory.profile_id,
                    )

                    preference_payload = [
                        item.model_dump(mode="json") for item in memory.preferences
                    ]

                    if record is None:
                        session.add(
                            PreferenceMemoryRecord(
                                profile_id=memory.profile_id,
                                preferences=preference_payload,
                                revision=memory.revision,
                                updated_at=memory.updated_at,
                            )
                        )
                        return

                    record.preferences = preference_payload
                    record.revision = memory.revision
                    record.updated_at = memory.updated_at
            except SQLAlchemyError as error:
                safe_msg = redact_secrets(str(error))
                raise DatabaseUnavailableError(
                    f"Failed to save preference memory: {safe_msg}"
                ) from error


class PostgresSignedRecommendationRepository:
    """Persist signed results as append-only integrity records."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(
        self,
        *,
        request_id: UUID,
        signed_recommendation: SignedFinalRecommendation,
    ) -> UUID:
        """Store one immutable signed recommendation and return its record ID."""
        with timed_step("persistence"):
            try:
                with self._session_factory.begin() as session:
                    record = SignedRecommendationRecord(
                        request_id=request_id,
                        status=signed_recommendation.recommendation.status.value,
                        signed_payload=signed_recommendation.model_dump(mode="json"),
                    )
                    session.add(record)
                    session.flush()

                    return record.result_id
            except SQLAlchemyError as error:
                safe_msg = redact_secrets(str(error))
                raise DatabaseUnavailableError(
                    f"Failed to save signed recommendation: {safe_msg}"
                ) from error

    def get(
        self,
        result_id: UUID,
    ) -> SignedFinalRecommendation | None:
        """Load and validate one stored signed recommendation."""
        with timed_step("persistence"):
            try:
                with self._session_factory() as session:
                    record = session.get(SignedRecommendationRecord, result_id)

                    if record is None:
                        return None

                    return SignedFinalRecommendation.model_validate(record.signed_payload)
            except SQLAlchemyError as error:
                safe_msg = redact_secrets(str(error))
                raise DatabaseUnavailableError(
                    f"Failed to load signed recommendation: {safe_msg}"
                ) from error

    def list_result_ids(
        self,
        request_id: UUID,
    ) -> list[UUID]:
        """List result history for a planning request, newest first."""
        with timed_step("persistence"):
            try:
                with self._session_factory() as session:
                    statement = (
                        select(SignedRecommendationRecord.result_id)
                        .where(SignedRecommendationRecord.request_id == request_id)
                        .order_by(SignedRecommendationRecord.created_at.desc())
                    )

                    return list(session.scalars(statement))
            except SQLAlchemyError as error:
                safe_msg = redact_secrets(str(error))
                raise DatabaseUnavailableError(
                    f"Failed to list recommendation IDs: {safe_msg}"
                ) from error
