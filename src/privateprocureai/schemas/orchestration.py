from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.agents import AgentResult
from privateprocureai.schemas.budget import BudgetAssessment
from privateprocureai.schemas.evaluation import EvaluationReport
from privateprocureai.schemas.integrity import IntegrityVerification, SignedFinalRecommendation
from privateprocureai.schemas.intent import ExtractedIntent
from privateprocureai.schemas.models import (
    DomainModel,  # Provides a common base model for your project's schemas.
    Evidence,
)
from privateprocureai.schemas.optimization import FinalRecommendation
from privateprocureai.schemas.planning import TaskPlan
from privateprocureai.schemas.replanning import ReplanningIteration

"""Schemas representing workflow status and observable execution events."""

"""This file defines the structures that allow your workflow to communicate
its progress and return its result in a consistent format."""

# the above block explanation
# This block imports the components required to define the workflow's data structure


class WorkflowStatus(StrEnum):
    """Possible terminal states for the planning workflow."""

    PLANNED = "planned"
    NEEDS_CLARIFICATION = "needs_clarification"


class WorkflowStep(StrEnum):
    """Named workflow stages recorded for observability."""

    INTENT_EXTRACTION = "intent_extraction"
    TASK_PLANNING = "task_planning"
    RETRIEVAL = "retrieval"
    AGENT_EXECUTION = "agent_execution"
    BUDGET_VALIDATION = "budget_validation"
    EVALUATION = "evaluation"
    REPLANNING = "replanning"
    OPTIMIZATION = "optimization"
    SIGNING = "signing"
    PERSISTENCE = "persistence"


class WorkflowEvent(DomainModel):
    """A safe, human-readable record of one completed workflow step."""

    step: WorkflowStep
    message: str = Field(min_length=1, max_length=1_000)
    status: str = Field(default="success")
    event_type: str = Field(default="workflow")
    duration_ms: float | None = None
    agent_name: str | None = None
    input_ref: str | None = None
    output_ref: str | None = None
    error_details: str | None = None
    retry_count: int = Field(default=0, ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkflowResult(DomainModel):  # This is the main output schema of the workflow.
    """The terminal result returned by the Module 6 orchestration workflow."""

    request_id: UUID
    status: WorkflowStatus
    intent: ExtractedIntent
    task_plan: TaskPlan | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)
    # Returns the execution history


class ApplicationWorkflowStatus(StrEnum):
    """Terminal states for the complete end-to-end application workflow."""

    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    REPLANNING_EXHAUSTED = "replanning_exhausted"
    FAILED = "failed"


class ApplicationWorkflowResult(DomainModel):
    """Full end-to-end outcome from user goal to verified, signed recommendation."""

    request_id: UUID
    status: ApplicationWorkflowStatus
    intent: ExtractedIntent
    task_plan: TaskPlan | None = None
    agent_results: list[AgentResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    budget_assessment: BudgetAssessment | None = None
    evaluation_report: EvaluationReport | None = None
    replanning_iterations: list[ReplanningIteration] = Field(default_factory=list)
    final_recommendation: FinalRecommendation | None = None
    signed_recommendation: SignedFinalRecommendation | None = None
    verification: IntegrityVerification | None = None
    persisted_result_id: UUID | None = None
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    events: list[WorkflowEvent] = Field(default_factory=list)
