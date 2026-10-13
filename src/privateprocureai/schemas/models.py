"""Core domain models for planning, agent work, evidence, and recommendations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainModel(BaseModel):
    """Base model with shared validation rules for application data."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class TaskKind(StrEnum):
    """The meaningful responsibilities that specialized agents may perform."""

    RESEARCH = "research"
    RECOMMENDATION = "recommendation"
    BUDGET_ANALYSIS = "budget_analysis"
    EVALUATION = "evaluation"


class TaskPriority(StrEnum):
    """Relative execution importance for a planning task."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class AgentStatus(StrEnum):
    """Lifecycle states for a specialized agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Money(DomainModel):
    """A monetary amount represented safely with Decimal rather than float."""

    amount: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class BudgetConstraint(DomainModel):
    """A spending limit supplied by the user or planning policy."""

    maximum: Money
    is_hard_limit: bool = True


class UserPreferences(DomainModel):
    """Preferences that guide recommendations without replacing hard constraints."""

    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=1_000)


class PlanningRequest(DomainModel):
    """A user's high-level goal converted into a validated planning input."""

    request_id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=3, max_length=2_000)
    budget: BudgetConstraint | None = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlanTask(DomainModel):
    """A specific unit of work produced by the future planner."""

    task_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=3, max_length=200)
    instructions: str = Field(min_length=3, max_length=2_000)
    kind: TaskKind
    priority: TaskPriority = TaskPriority.NORMAL
    depends_on: list[UUID] = Field(default_factory=list)
    requires_retrieval: bool = False

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        """Reject impossible or duplicated dependency definitions."""
        if self.task_id in self.depends_on:
            raise ValueError("A task cannot depend on itself.")

        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("A task cannot list the same dependency more than once.")

        return self


class Evidence(DomainModel):
    """A retrieved or externally supplied fact supporting a recommendation."""

    evidence_id: UUID = Field(default_factory=uuid4)
    source_id: str = Field(min_length=1, max_length=200)
    source_url: str | None = Field(default=None, max_length=2_000)
    excerpt: str = Field(min_length=1, max_length=5_000)
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Recommendation(DomainModel):
    """A candidate recommendation with optional cost and evidence references."""

    recommendation_id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=3, max_length=200)
    rationale: str = Field(min_length=3, max_length=2_000)
    estimated_cost: Money | None = None
    evidence_ids: list[UUID] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)

class Finding(DomainModel):
    """A source-linked statement produced by an agent."""

    finding_id: UUID = Field(default_factory=uuid4)
    statement: str = Field(min_length=3, max_length=2_000)
    evidence_ids: list[UUID] = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)


class AgentResult(DomainModel):
    """The structured output contract used by each future specialized agent."""

    agent_name: str = Field(min_length=1, max_length=100)
    status: AgentStatus
    summary: str | None = Field(default=None, max_length=2_000)
    recommendations: list[Recommendation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    error_message: str | None = Field(default=None, max_length=1_000)
    findings: list[Finding] = Field(default_factory=list)
    task_id: UUID | None=None 
    # the above line connects each agent result to the task that produced it .

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require failure context and prevent errors on successful results."""
        if self.status is AgentStatus.FAILED and not self.error_message:
            raise ValueError("A failed agent result must include an error message.")

        if self.status is not AgentStatus.FAILED and self.error_message is not None:
            raise ValueError("Only failed agent results may include an error message.")

        return self