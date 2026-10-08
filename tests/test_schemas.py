"""Tests for core domain schemas."""

from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    BudgetConstraint,
    Money,
    PlanningRequest,
    PlanTask,
    TaskKind,
    UserPreferences,
)


def test_planning_request_accepts_goal_budget_and_preferences() -> None:
    """A valid user request is converted into a typed planning input."""
    request = PlanningRequest(
        goal="Plan a three-day Delhi trip under a defined budget.",
        budget=BudgetConstraint(maximum=Money(amount="20000.00", currency="INR")),
        preferences=UserPreferences(
            must_have=["vegetarian food"],
            nice_to_have=["near a metro station"],
        ),
    )

    assert request.budget is not None
    assert request.budget.maximum.amount == Decimal("20000.00")
    assert request.preferences.must_have == ["vegetarian food"]


def test_money_rejects_more_than_two_decimal_places() -> None:
    """Currency amounts must use normal two-decimal-place precision."""
    with pytest.raises(ValidationError):
        Money(amount="100.999", currency="INR")


def test_task_cannot_depend_on_itself() -> None:
    """The planner cannot create an impossible self-referential task."""
    task_id = uuid4()

    with pytest.raises(ValidationError, match="cannot depend on itself"):
        PlanTask(
            task_id=task_id,
            title="Research Delhi hotels",
            instructions="Find budget hotels close to metro stations.",
            kind=TaskKind.RESEARCH,
            depends_on=[task_id],
        )


def test_failed_agent_result_requires_an_error_message() -> None:
    """Failures must remain observable to the future orchestrator."""
    with pytest.raises(ValidationError, match="must include an error message"):
        AgentResult(
            agent_name="research_agent",
            status=AgentStatus.FAILED,
        )


def test_completed_agent_result_cannot_include_an_error_message() -> None:
    """A non-failed result must not misleadingly carry an error."""
    with pytest.raises(ValidationError, match="Only failed agent results"):
        AgentResult(
            agent_name="research_agent",
            status=AgentStatus.COMPLETED,
            error_message="This should not be here.",
        )


def test_unknown_fields_are_rejected() -> None:
    """Unexpected data is rejected instead of silently entering the system."""
    with pytest.raises(ValidationError):
        PlanningRequest(
            goal="Plan a small event.",
            unexpected_field="not allowed",
        )