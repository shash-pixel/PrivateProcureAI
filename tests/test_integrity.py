"""Tests for Ed25519 recommendation integrity verification."""

from pathlib import Path

import pytest

from privateprocureai.integrity import Ed25519ResultSigner
from privateprocureai.integrity.signing import (
    SigningError,
    generate_ed25519_keypair,
)
from privateprocureai.schemas import (
    FinalRecommendation,
    FinalRecommendationStatus,
)


def make_signer(tmp_path: Path) -> Ed25519ResultSigner:
    """Create an isolated key pair for one test."""
    private_key_path = tmp_path / "private.pem"
    public_key_path = tmp_path / "public.pem"

    generate_ed25519_keypair(
        private_key_path=private_key_path,
        public_key_path=public_key_path,
        passphrase=b"test-passphrase",
    )

    return Ed25519ResultSigner.from_pem_files(
        private_key_path=private_key_path,
        public_key_path=public_key_path,
        passphrase=b"test-passphrase",
    )


def make_ready_recommendation() -> FinalRecommendation:
    """Create a minimal final recommendation suitable for signing."""
    return FinalRecommendation(
        status=FinalRecommendationStatus.READY,
        summary="A validated final recommendation.",
        selected_recommendations=[],
        requires_replanning=False,
    )


def test_signature_verifies_for_unchanged_recommendation(tmp_path: Path) -> None:
    """A signed, unchanged recommendation passes verification."""
    signer = make_signer(tmp_path)

    signed_recommendation = signer.sign(make_ready_recommendation())
    verification = signer.verify(signed_recommendation)

    assert verification.is_valid is True
    assert verification.reason is None


def test_signature_detects_tampered_recommendation(tmp_path: Path) -> None:
    """Changing a signed payload invalidates its stored hash and signature."""
    signer = make_signer(tmp_path)

    signed_recommendation = signer.sign(make_ready_recommendation())
    tampered_recommendation = signed_recommendation.model_copy(
        update={
            "recommendation": signed_recommendation.recommendation.model_copy(
                update={"summary": "A modified recommendation."}
            )
        }
    )

    verification = signer.verify(tampered_recommendation)

    assert verification.is_valid is False


def test_blocked_recommendation_cannot_be_signed(tmp_path: Path) -> None:
    """Only validated ready recommendations can receive integrity signatures."""
    signer = make_signer(tmp_path)

    blocked_recommendation = FinalRecommendation(
        status=FinalRecommendationStatus.BLOCKED,
        summary="Recommendation requires re-planning.",
        selected_recommendations=[],
        requires_replanning=True,
    )

    with pytest.raises(SigningError, match="Only a ready"):
        signer.sign(blocked_recommendation)