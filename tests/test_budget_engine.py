"""Tests for deterministic budget evaluation."""

from decimal import Decimal

from privateprocureai.constraints import BudgetEngine
from privateprocureai.schemas import (
    BudgetConstraint,
    BudgetEvaluationInput,
    BudgetStatus,
    BudgetViolation,
    CostLineItem,
    Money,
)


def make_budget(maximum: str, *, hard_limit: bool = True) -> BudgetConstraint:
    """Build a test budget with an explicit currency."""
    return BudgetConstraint(
        maximum=Money(amount=maximum, currency="USD"),
        is_hard_limit=hard_limit,
    )


def test_budget_engine_marks_costs_within_limit_as_compliant() -> None:
    """Total cost below the limit leaves a positive remaining amount."""
    assessment = BudgetEngine().evaluate(
        BudgetEvaluationInput(
            budget=make_budget("100.00"),
            line_items=[
                CostLineItem(label="1st selected item", cost=Money(amount="25.00", currency="USD")),
                CostLineItem(label="2nd selected item", cost=Money(amount="40.00", currency="USD")),
            ],
        )
    )

    assert assessment.status is BudgetStatus.COMPLIANT
    assert assessment.total_cost is not None
    assert assessment.total_cost.amount == Decimal("65.00")
    assert assessment.remaining_amount == Decimal("35.00")
    assert assessment.requires_replanning is False


def test_budget_engine_marks_hard_budget_violation_for_replanning() -> None:
    """A hard budget violation requests future re-planning."""
    assessment = BudgetEngine().evaluate(
        BudgetEvaluationInput(
            budget=make_budget("100.00"),
            line_items=[
                CostLineItem(label="Selected item", cost=Money(amount="125.00", currency="USD")),
            ],
        )
    )

    assert assessment.status is BudgetStatus.EXCEEDED
    assert assessment.remaining_amount == Decimal("-25.00")
    assert assessment.violations == [BudgetViolation.BUDGET_EXCEEDED]
    assert assessment.requires_replanning is True


def test_budget_engine_does_not_convert_mixed_currencies() -> None:
    """Currency mismatches remain visible instead of receiving a fake total."""
    assessment = BudgetEngine().evaluate(
        BudgetEvaluationInput(
            budget=make_budget("100.00"),
            line_items=[
                CostLineItem(label="First item", cost=Money(amount="25.00", currency="USD")),
                CostLineItem(label="Second item", cost=Money(amount="25.00", currency="EUR")),
            ],
        )
    )

    assert assessment.status is BudgetStatus.UNEVALUABLE
    assert assessment.total_cost is None
    assert assessment.remaining_amount is None
    assert assessment.violations == [BudgetViolation.CURRENCY_MISMATCH]


def test_soft_budget_violation_does_not_require_replanning() -> None:
    """A non-hard budget can be exceeded without forcing a workflow restart."""
    assessment = BudgetEngine().evaluate(
        BudgetEvaluationInput(
            budget=make_budget("100.00", hard_limit=False),
            line_items=[
                CostLineItem(label="Selected item", cost=Money(amount="125.00", currency="USD")),
            ],
        )
    )

    assert assessment.status is BudgetStatus.EXCEEDED
    assert assessment.requires_replanning is False