"""Centralized application logging configuration and context filters."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from privateprocureai.observability.context import (
    get_agent_name,
    get_request_id,
    get_workflow_step,
)
from privateprocureai.observability.redaction import (
    SecretMaskingFilter,
    register_secret,
)

if TYPE_CHECKING:
    from privateprocureai.config import AppSettings


class RequestContextFilter(logging.Filter):
    """Inject contextvars (request_id, step, agent_name) into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id") or record.request_id is None:
            record.request_id = get_request_id() or "-"

        if not hasattr(record, "step") or record.step is None:
            record.step = get_workflow_step() or "-"

        if not hasattr(record, "agent_name") or record.agent_name is None:
            record.agent_name = get_agent_name() or "-"

        return True


def configure_logging(
    level: str = "INFO",
    settings: AppSettings | None = None,
) -> None:
    """Configure root and application loggers with contextual formatting and secret masking."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Register known secrets from settings if provided
    if settings is not None:
        if settings.llm_api_key:
            register_secret(settings.llm_api_key.get_secret_value())
        if settings.signing_private_key_passphrase:
            register_secret(settings.signing_private_key_passphrase.get_secret_value())
        if settings.database_url:
            register_secret(settings.database_url.get_secret_value())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Check if our handler is already configured
    has_app_handler = any(
        getattr(handler, "_privateprocureai_handler", False) for handler in root_logger.handlers
    )

    if not has_app_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler._privateprocureai_handler = True  # type: ignore[attr-defined]
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s [%(levelname)s] [request_id=%(request_id)s] "
                "[step=%(step)s] [agent=%(agent_name)s] %(name)s: %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        handler.addFilter(SecretMaskingFilter())

        # Clean existing handlers or add ours
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(log_level)
            handler.addFilter(SecretMaskingFilter())
