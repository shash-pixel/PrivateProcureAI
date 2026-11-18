"""Ed25519 signing and verification for validated final recommendations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from privateprocureai.schemas import (
    FinalRecommendation,
    FinalRecommendationStatus,
    IntegrityRecord,
    IntegrityVerification,
    SignedFinalRecommendation,
)


class SigningError(RuntimeError):
    """Raised when key loading or signing cannot safely complete."""


def canonicalize_recommendation(recommendation: FinalRecommendation) -> bytes:
    """Create deterministic bytes for hashing and signing."""
    payload = recommendation.model_dump(mode="json")

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def generate_ed25519_keypair(
    *,
    private_key_path: Path,
    public_key_path: Path,
    passphrase: bytes | None,
) -> None:
    """Generate encrypted-or-local Ed25519 PEM files without overwriting keys."""
    if private_key_path == public_key_path:
        raise SigningError("Private and public key paths must be different.")

    if private_key_path.exists() or public_key_path.exists():
        raise SigningError(
            "Refusing to overwrite existing key material. Choose new paths."
        )

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    encryption = (
        serialization.BestAvailableEncryption(passphrase)
        if passphrase
        else serialization.NoEncryption()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.write_bytes(private_pem)
    public_key_path.write_bytes(public_pem)


class Ed25519ResultSigner:
    """Sign and verify ready final recommendations using Ed25519."""

    def __init__(
        self,
        *,
        private_key: Ed25519PrivateKey,
        public_key: Ed25519PublicKey,
    ) -> None:
        self._private_key = private_key
        self._public_key = public_key

    @classmethod
    def from_pem_files(
        cls,
        *,
        private_key_path: Path,
        public_key_path: Path,
        passphrase: bytes | None = None,
    ) -> Ed25519ResultSigner:
        """Load a matching Ed25519 key pair from PEM files."""
        try:
            loaded_private_key = serialization.load_pem_private_key(
                private_key_path.read_bytes(),
                password=passphrase,
            )
            loaded_public_key = serialization.load_pem_public_key(
                public_key_path.read_bytes()
            )
        except (OSError, ValueError, TypeError) as error:
            raise SigningError("Could not load signing key files.") from error

        if not isinstance(loaded_private_key, Ed25519PrivateKey):
            raise SigningError("Private key must be an Ed25519 key.")

        if not isinstance(loaded_public_key, Ed25519PublicKey):
            raise SigningError("Public key must be an Ed25519 key.")

        derived_public_key = loaded_private_key.public_key()

        if (
            derived_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            != loaded_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ):
            raise SigningError("Private and public key files do not match.")

        return cls(
            private_key=loaded_private_key,
            public_key=loaded_public_key,
        )

    def sign(
        self,
        recommendation: FinalRecommendation,
    ) -> SignedFinalRecommendation:
        """Hash and sign a validated recommendation ready for presentation."""
        if recommendation.status is not FinalRecommendationStatus.READY:
            raise SigningError(
                "Only a ready final recommendation may receive a signature."
            )

        payload = canonicalize_recommendation(recommendation)
        payload_hash = hashlib.sha256(payload).hexdigest()
        signature = self._private_key.sign(payload)

        return SignedFinalRecommendation(
            recommendation=recommendation,
            integrity=IntegrityRecord(
                payload_sha256=payload_hash,
                signature_base64=base64.b64encode(signature).decode("ascii"),
                public_key_fingerprint_sha256=self._public_key_fingerprint(),
            ),
        )

    def verify(
        self,
        signed_recommendation: SignedFinalRecommendation,
    ) -> IntegrityVerification:
        """Verify payload hash, public-key identity, and Ed25519 signature."""
        payload = canonicalize_recommendation(
            signed_recommendation.recommendation
        )
        computed_hash = hashlib.sha256(payload).hexdigest()

        if not hmac.compare_digest(
            computed_hash,
            signed_recommendation.integrity.payload_sha256,
        ):
            return IntegrityVerification(
                is_valid=False,
                reason="The recommendation payload hash does not match.",
            )

        if not hmac.compare_digest(
            self._public_key_fingerprint(),
            signed_recommendation.integrity.public_key_fingerprint_sha256,
        ):
            return IntegrityVerification(
                is_valid=False,
                reason="The signing public-key fingerprint does not match.",
            )

        try:
            self._public_key.verify(
                base64.b64decode(
                    signed_recommendation.integrity.signature_base64,
                    validate=True,
                ),
                payload,
            )
        except (InvalidSignature, ValueError):
            return IntegrityVerification(
                is_valid=False,
                reason="The Ed25519 signature is invalid.",
            )

        return IntegrityVerification(is_valid=True)

    def _public_key_fingerprint(self) -> str:
        """Return a SHA-256 identifier for the public key, not the private key."""
        public_key_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        return hashlib.sha256(public_key_bytes).hexdigest()