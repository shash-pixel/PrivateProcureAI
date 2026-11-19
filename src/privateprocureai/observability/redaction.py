"""Redaction of secrets, API keys, private keys, and sensitive inputs in logs."""

from __future__ import annotations

import logging
import re

# Regex patterns for common secrets and sensitive tokens
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Gemini API keys
    (re.compile(r"AIza[0-9A-Za-z\-_]{30,}"), "[REDACTED_API_KEY]"),
    # OpenAI style keys
    (re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"), "[REDACTED_API_KEY]"),
    # Authorization header / Bearer tokens
    (re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    # Explicit password / passphrase keywords
    (
        re.compile(r"((?:password|passphrase|secret)\s*[:= ]\s*)[^\s,;'\"]+", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    # Database connection string passwords
    (
        re.compile(
            r"((?:postgresql|postgres|mysql|sqlite)(?:\+[a-zA-Z0-9]+)?://[^:]+:)([^@]+)(@)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED_PASSWORD]\3",
    ),
    # PEM private keys
    (
        re.compile(
            r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]*?"
            r"-----END (?:[A-Z ]+ )?PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
]

_REGISTERED_SECRETS: set[str] = set()


def register_secret(secret: str | None) -> None:
    """Register a known secret value (such as API key or passphrase) to mask in all logs."""
    if secret and len(secret.strip()) >= 4:
        _REGISTERED_SECRETS.add(secret.strip())


def clear_registered_secrets() -> None:
    """Clear all registered dynamic secrets (mainly for test isolation)."""
    _REGISTERED_SECRETS.clear()


def redact_secrets(text: str) -> str:
    """Redact known secrets and sensitive patterns from a string."""
    if not text:
        return text

    redacted = text

    # Redact dynamic registered secrets first
    for secret in _REGISTERED_SECRETS:
        if secret in redacted:
            redacted = redacted.replace(secret, "[REDACTED]")

    # Redact regex patterns
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)

    return redacted


def sanitize_user_input(text: str, max_length: int = 100) -> str:
    """Sanitize and truncate user input for safe log output."""
    if not text:
        return ""

    cleaned = " ".join(text.split())
    cleaned = redact_secrets(cleaned)

    if len(cleaned) > max_length:
        return f"{cleaned[:max_length]}... [truncated {len(cleaned)} chars]"

    return cleaned


class SecretMaskingFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials, keys, and passphrases."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)

        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_secrets(arg) if isinstance(arg, str) else arg for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    key: redact_secrets(val) if isinstance(val, str) else val
                    for key, val in record.args.items()
                }

        if record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)

        return True
