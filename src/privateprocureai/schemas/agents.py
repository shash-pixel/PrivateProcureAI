"""Schemas shared by specialized agents and the task executor."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.intent import ExtractedIntent
from privateprocureai.schemas.models import (
    AgentResult,
    DomainModel,
    Evidence,
    PlanTask,
)


class AgentTaskContext(DomainModel):
    """All validated input available to one specialized agent execution."""

    intent: ExtractedIntent
    task: PlanTask
    evidence: list[Evidence] = Field(default_factory=list)
    dependency_results: list[AgentResult] = Field(default_factory=list)


class PlanExecutionResult(DomainModel):
    """Agent outputs created while executing one validated task plan."""

    plan_id: UUID
    results: list[AgentResult] = Field(default_factory=list)