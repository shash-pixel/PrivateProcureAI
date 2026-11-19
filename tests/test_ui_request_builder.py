from decimal import Decimal

import pytest

from privateprocureai.ui.request_builder import build_planning_request, split_lines


def test_split_lines_discards_blank_lines_and_whitespace() -> None:
    assert split_lines(" first  \n\nsecond\n ") == ["first", "second"]


def test_build_planning_request_keeps_generic_preferences_and_budget() -> None:
    request = build_planning_request(
        goal="Plan a suitable outcome for a user-defined objective.",
        include_budget=True,
        budget_amount="125.50",
        budget_currency="usd",
        budget_is_hard_limit=True,
        must_have="accessible\nverified",
        nice_to_have="flexible",
        avoid="unnecessary commitments",
        notes="Use reliable sources.",
    )

    assert request.budget is not None
    assert request.budget.maximum.amount == Decimal("125.50")
    assert request.budget.maximum.currency == "USD"
    assert request.preferences.must_have == ["accessible", "verified"]
    assert request.preferences.nice_to_have == ["flexible"]
    assert request.preferences.avoid == ["unnecessary commitments"]


def test_build_planning_request_rejects_invalid_budget_amount() -> None:
    with pytest.raises(ValueError, match="valid budget amount"):
        build_planning_request(
            goal="Plan a suitable outcome for a user-defined objective.",
            include_budget=True,
            budget_amount="not-a-number",
            budget_currency="USD",
            budget_is_hard_limit=True,
            must_have="",
            nice_to_have="",
            avoid="",
            notes="",
        )
