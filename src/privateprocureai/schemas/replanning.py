"""Schemas for bounded iterative planning and re-planning."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from privateprocureai.schemas.agents import PlanExecutionResult
from privateprocureai.schemas.evaluation import EvaluationReport
from privateprocureai.schemas.models import DomainModel
from privateprocureai.schemas.planning import TaskPlan


class ReplanningStatus(StrEnum):
    """Terminal states for iterative planning."""

    COMPLETED = "completed"
    EXHAUSTED = "exhausted"


class ReplanningConfig(DomainModel):
    """Safety limits for iterative re-planning."""

    max_iterations: int = Field(default=3, ge=1, le=5)


class ReplanningIteration(DomainModel):
    """One complete plan, execution, and evaluation cycle."""

    iteration_number: int = Field(ge=1)
    task_plan: TaskPlan
    execution_result: PlanExecutionResult
    evaluation_report: EvaluationReport
    feedback_for_next_iteration: list[str] = Field(default_factory=list)


class ReplanningOutcome(DomainModel):
    """Final result after a successful or exhausted planning loop."""

    status: ReplanningStatus
    iterations: list[ReplanningIteration] = Field(min_length=1)