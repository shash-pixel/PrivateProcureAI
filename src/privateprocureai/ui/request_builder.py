"""Convert Streamlit form values into validated planning requests."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from privateprocureai.schemas import BudgetConstraint, Money, PlanningRequest, UserPreferences


def split_lines(value: str) -> list[str]:
    """Return non-empty, trimmed entries from a multi-line text field."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def build_planning_request(
    *,
    goal: str,
    include_budget: bool,
    budget_amount: str,
    budget_currency: str,
    budget_is_hard_limit: bool,
    must_have: str,
    nice_to_have: str,
    avoid: str,
    notes: str,
) -> PlanningRequest:
    """Build one generic planning request from the UI form values.

    Pydantic validates the final models; this function adds a clear error for
    a budget amount that cannot be represented as a decimal monetary value.
    """
    budget = None

    if include_budget:
        try:
            amount = Decimal(budget_amount.strip())
        except (InvalidOperation, ValueError) as error:
            raise ValueError("Enter a valid budget amount.") from error

        budget = BudgetConstraint(
            maximum=Money(
                amount=amount,
                currency=budget_currency.strip().upper(),
            ),
            is_hard_limit=budget_is_hard_limit,
        )

    return PlanningRequest(
        goal=goal.strip(),
        budget=budget,
        preferences=UserPreferences(
            must_have=split_lines(must_have),
            nice_to_have=split_lines(nice_to_have),
            avoid=split_lines(avoid),
            notes=notes.strip() or None,
        ),
    )
