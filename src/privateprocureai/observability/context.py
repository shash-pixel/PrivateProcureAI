"""Request and execution context tracking across asynchronous/concurrent boundaries."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any
from uuid import UUID

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_step_var: ContextVar[str | None] = ContextVar("workflow_step", default=None)
_agent_name_var: ContextVar[str | None] = ContextVar("agent_name", default=None)


def get_request_id() -> str | None:
    """Return the active request ID in the current execution context."""
    return _request_id_var.get()


def set_request_id(request_id: str | UUID | None) -> Token[str | None]:
    """Set the active request ID in the current execution context."""
    value = str(request_id) if request_id is not None else None
    return _request_id_var.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """Reset the request ID to a prior token."""
    _request_id_var.reset(token)


def get_workflow_step() -> str | None:
    """Return the active workflow step in the current execution context."""
    return _step_var.get()


def set_workflow_step(step: str | None) -> Token[str | None]:
    """Set the active workflow step in the current execution context."""
    return _step_var.set(step)


def reset_workflow_step(token: Token[str | None]) -> None:
    """Reset the workflow step to a prior token."""
    _step_var.reset(token)


def get_agent_name() -> str | None:
    """Return the active agent name in the current execution context."""
    return _agent_name_var.get()


def set_agent_name(agent_name: str | None) -> Token[str | None]:
    """Set the active agent name in the current execution context."""
    return _agent_name_var.set(agent_name)


def reset_agent_name(token: Token[str | None]) -> None:
    """Reset the agent name to a prior token."""
    _agent_name_var.reset(token)


class RequestContext:
    """Context manager to scope request ID, workflow step, and agent name."""

    def __init__(
        self,
        request_id: str | UUID | None = None,
        step: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._request_id = str(request_id) if request_id is not None else None
        self._step = step
        self._agent_name = agent_name
        self._tokens: list[tuple[ContextVar[Any], Token[Any]]] = []

    def __enter__(self) -> RequestContext:
        if self._request_id is not None:
            self._tokens.append((_request_id_var, _request_id_var.set(self._request_id)))
        if self._step is not None:
            self._tokens.append((_step_var, _step_var.set(self._step)))
        if self._agent_name is not None:
            self._tokens.append((_agent_name_var, _agent_name_var.set(self._agent_name)))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for var, token in reversed(self._tokens):
            var.reset(token)
