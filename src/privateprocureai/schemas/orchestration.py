"""Schemas representing workflow status and observable execution events."""

"""This file defines the structures that allow your workflow to communicate its progress and return its result in a consistent format."""

from __future__ import annotations     # This block imports the components required to define the workflow's data structure

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.intent import ExtractedIntent
from privateprocureai.schemas.models import DomainModel   #Provides a common base model for your project's schemas.
from privateprocureai.schemas.planning import TaskPlan


class WorkflowStatus(StrEnum):
    """Possible terminal states for the planning workflow."""

    PLANNED = "planned"
    NEEDS_CLARIFICATION = "needs_clarification"


class WorkflowStep(StrEnum):
    """Named workflow stages recorded for observability."""

    INTENT_EXTRACTION = "intent_extraction"
    TASK_PLANNING = "task_planning"


class WorkflowEvent(DomainModel):
    """A safe, human-readable record of one completed workflow step."""

    step: WorkflowStep             #Identifies the workflow stage associated with the event.
    message: str = Field(min_length=1, max_length=1_000)          #Provides a human-readable description of the event.
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))    #Records when the event occurred.


class WorkflowResult(DomainModel):      #This is the main output schema of the workflow.
    """The terminal result returned by the Module 6 orchestration workflow."""

    request_id: UUID
    status: WorkflowStatus
    intent: ExtractedIntent
    task_plan: TaskPlan | None = None
    events: list[WorkflowEvent] = Field(default_factory=list)  #Returns the execution history