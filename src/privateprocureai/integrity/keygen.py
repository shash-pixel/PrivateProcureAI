"""Command-line utility for local Ed25519 key-pair generation."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from privateprocureai.integrity.signing import generate_ed25519_keypair


def main() -> None:
    """Generate key files without exposing a passphrase in command history."""
    parser = argparse.ArgumentParser(
        description="Generate an Ed25519 signing key pair."
    )
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--public-key", required=True, type=Path)
    parser.add_argument(
        "--unencrypted",
        action="store_true",
        help="Create an unencrypted private key for local development only.",
    )
    arguments = parser.parse_args()

    passphrase = None

    if not arguments.unencrypted:
        entered_passphrase = getpass.getpass("Private-key passphrase: ")
        confirmation = getpass.getpass("Confirm passphrase: ")

        if not entered_passphrase:
            raise SystemExit("Passphrase cannot be empty.")

        if entered_passphrase != confirmation:
            raise SystemExit("Passphrases do not match.")

        passphrase = entered_passphrase.encode("utf-8")

    generate_ed25519_keypair(
        private_key_path=arguments.private_key,
        public_key_path=arguments.public_key,
        passphrase=passphrase,
    )

    print("Ed25519 key pair created successfully.")


if __name__ == "__main__":
    main()