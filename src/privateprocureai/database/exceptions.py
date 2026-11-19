"""Database exceptions for persistence layers."""

from __future__ import annotations


class DatabaseError(RuntimeError):
    """Base exception for database failures."""


class DatabaseUnavailableError(DatabaseError):
    """Raised when the PostgreSQL database is unreachable or connection fails."""
