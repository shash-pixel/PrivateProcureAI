"""Execution timing and latency instrumentation for workflow components."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

from privateprocureai.observability.context import (
    RequestContext,
    get_agent_name,
    get_request_id,
)
from privateprocureai.observability.redaction import redact_secrets

P = ParamSpec("P")
R = TypeVar("R")


class TimingResult:
    """Carries timing outcome and duration for a completed or failed operation."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        self.duration_ms: float = 0.0
        self.status: str = "pending"
        self.error: Exception | None = None

    @property
    def safe_error_message(self) -> str | None:
        """Return a secret-redacted error message if an exception occurred."""
        if self.error is None:
            return None
        return redact_secrets(str(self.error))


@contextmanager
def timed_step(
    step_name: str,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    agent_name: str | None = None,
) -> Iterator[TimingResult]:
    """Measure duration and log structured timing events around an operation."""
    timing = TimingResult(step_name)
    log = logger or logging.getLogger("privateprocureai.timing")
    start = time.perf_counter()

    with RequestContext(step=step_name, agent_name=agent_name):
        req_id = get_request_id() or "none"
        agent = agent_name or get_agent_name() or "none"

        log.log(
            level,
            "Step '%s' started [request_id=%s, agent=%s]",
            step_name,
            req_id,
            agent,
        )

        try:
            yield timing
            timing.status = "success"
        except Exception as error:
            timing.status = "failed"
            timing.error = error
            timing.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.warning(
                "Step '%s' failed after %.2fms [request_id=%s, agent=%s]: %s",
                step_name,
                timing.duration_ms,
                req_id,
                agent,
                timing.safe_error_message,
            )
            raise
        else:
            timing.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.log(
                level,
                "Step '%s' completed in %.2fms [request_id=%s, agent=%s, status=success]",
                step_name,
                timing.duration_ms,
                req_id,
                agent,
            )


def timed_function(
    step_name: str,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to measure and log function execution latency."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_step(step_name, logger=logger, level=level):
                return func(*args, **kwargs)

        return wrapper

    return decorator
