"""Internal LangGraph state shared between workflow nodes."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from privateprocureai.schemas import (
    ExtractedIntent,
    PlanningRequest,
    TaskPlan,
    WorkflowEvent,
)


class PlanningWorkflowState(TypedDict, total=False):
    """State fields read and updated by the Module 6 graph nodes."""

    planning_request: PlanningRequest
    extracted_intent: ExtractedIntent
    task_plan: TaskPlan

# LangGraph uses this reducer to append events from successive nodes.
    execution_log: Annotated[list[WorkflowEvent], add]


##TypedDict describes state without turning it into a separate runtime data model.