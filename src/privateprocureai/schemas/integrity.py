"""Schemas for cryptographic recommendation integrity records."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.models import DomainModel
from privateprocureai.schemas.optimization import FinalRecommendation


class SignatureAlgorithm(StrEnum):
    """Supported result-signing algorithm."""

    ED25519 = "ed25519"


class IntegrityRecord(DomainModel):
    """Hash and signature metadata for one final recommendation."""

    algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    signature_base64: str = Field(min_length=1)
    public_key_fingerprint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    signed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: UUID | None = None
    intent_hash_sha256: str | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)
    budget_status: str | None = None
    evaluation_status: str | None = None


class SignedFinalRecommendation(DomainModel):
    """A final recommendation with detached integrity metadata."""

    recommendation: FinalRecommendation
    integrity: IntegrityRecord


class IntegrityVerification(DomainModel):
    """Result of verifying a signed recommendation."""

    is_valid: bool
    reason: str | None = None
