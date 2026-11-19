"""Database engine, session factory, and schema initialization helpers."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from privateprocureai.database.exceptions import DatabaseUnavailableError
from privateprocureai.database.models import Base
from privateprocureai.observability import redact_secrets


def create_database_engine(database_url: str) -> Engine:
    """Create a synchronous PostgreSQL-compatible SQLAlchemy engine."""
    try:
        return create_engine(
            database_url,
            pool_pre_ping=True,
        )
    except SQLAlchemyError as error:
        safe_msg = redact_secrets(str(error))
        raise DatabaseUnavailableError(f"Database engine creation failed: {safe_msg}") from error


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    """Create a reusable factory for short-lived database sessions."""
    return sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )


def initialize_database(engine: Engine) -> None:
    """Create database tables for local development and early project stages."""
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError as error:
        safe_msg = redact_secrets(str(error))
        raise DatabaseUnavailableError(f"Database initialization failed: {safe_msg}") from error
