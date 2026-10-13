"""Common protocol implemented by every specialized task agent."""

from __future__ import annotations

from typing import Protocol

from privateprocureai.schemas import AgentResult, AgentTaskContext, PlanTask


class TaskAgent(Protocol):
    """Contract used by the dependency-aware task executor."""

    agent_name: str

    def supports(self, task: PlanTask) -> bool:
        """Return whether this agent can execute the supplied task."""

    def run(self, context: AgentTaskContext) -> AgentResult:
        """Execute one supported task and return a structured result."""