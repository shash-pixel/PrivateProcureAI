"""Schemas for generic task decomposition and planning."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from privateprocureai.schemas.models import DomainModel, PlanTask, TaskKind, TaskPriority

NodeT = TypeVar("NodeT", bound=Hashable)


def _has_cycle(graph: Mapping[NodeT, list[NodeT]]) -> bool:
    """Return whether a directed dependency graph contains a cycle."""
    visiting: set[NodeT] = set()
    visited: set[NodeT] = set()

    def visit(node: NodeT) -> bool:
        if node in visiting:
            return True

        if node in visited:
            return False

        visiting.add(node)

        for dependency in graph[node]:
            if visit(dependency):
                return True

        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


class TaskDraft(DomainModel):
    """LLM-generated task before application code assigns permanent IDs."""

    task_key: str = Field(
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        description="A unique, human-readable identifier within one plan.",
    )
    title: str = Field(min_length=3, max_length=200)
    instructions: str = Field(min_length=3, max_length=2_000)
    kind: TaskKind
    priority: TaskPriority = TaskPriority.NORMAL
    depends_on: list[str] = Field(default_factory=list)
    requires_retrieval: bool = False


class TaskPlanDraft(DomainModel):
    """LLM-generated task plan using symbolic task keys."""

    planning_summary: str = Field(min_length=3, max_length=2_000)
    tasks: list[TaskDraft] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> TaskPlanDraft:
        """Reject duplicate, missing, self-referential, and cyclic dependencies."""
        task_keys = [task.task_key for task in self.tasks]

        if len(task_keys) != len(set(task_keys)):
            raise ValueError("Each task_key must be unique.")

        task_key_set = set(task_keys)
        graph: dict[str, list[str]] = {}

        for task in self.tasks:
            if task.task_key in task.depends_on:
                raise ValueError(f"Task '{task.task_key}' cannot depend on itself.")

            if len(task.depends_on) != len(set(task.depends_on)):
                raise ValueError(
                    f"Task '{task.task_key}' contains duplicate dependencies."
                )

            unknown_dependencies = set(task.depends_on) - task_key_set
            if unknown_dependencies:
                raise ValueError(
                    f"Task '{task.task_key}' has unknown dependencies: "
                    f"{sorted(unknown_dependencies)}."
                )

            graph[task.task_key] = task.depends_on

        if _has_cycle(graph):
            raise ValueError("Task dependencies must not contain a cycle.")

        return self


class TaskPlan(DomainModel):
    """Validated, application-owned task plan ready for orchestration."""

    plan_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    planning_summary: str = Field(min_length=3, max_length=2_000)
    tasks: list[PlanTask] = Field(min_length=1, max_length=20)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_task_dependencies(self) -> TaskPlan:
        """Ensure all UUID dependencies exist and form an acyclic graph."""
        task_ids = [task.task_id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Each task_id must be unique.")

        task_id_set = set(task_ids)
        graph: dict[UUID, list[UUID]] = {}

        for task in self.tasks:
            unknown_dependencies = set(task.depends_on) - task_id_set
            if unknown_dependencies:
                raise ValueError(
                    f"Task '{task.task_id}' has dependencies outside this plan."
                )

            graph[task.task_id] = task.depends_on

        if _has_cycle(graph):
            raise ValueError("Task dependencies must not contain a cycle.")

        return self