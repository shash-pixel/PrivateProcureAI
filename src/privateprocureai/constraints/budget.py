"""Deterministic budget evaluation without LLM decision-making."""

from __future__ import annotations

from decimal import Decimal

from privateprocureai.schemas import (
    BudgetAssessment,
    BudgetEvaluationInput,
    BudgetStatus,
    BudgetViolation,
    Money,
)


class BudgetEngine:
    """Evaluate selected cost items against one explicit budget limit."""

    def evaluate(self, evaluation_input: BudgetEvaluationInput) -> BudgetAssessment:
        """Calculate total cost and determine deterministic budget compliance."""
        budget = evaluation_input.budget
        budget_currency = budget.maximum.currency

        has_currency_mismatch = any(
            item.cost.currency != budget_currency
            for item in evaluation_input.line_items
        )

        if has_currency_mismatch:
            return BudgetAssessment(
                budget=budget,
                status=BudgetStatus.UNEVALUABLE,
                total_cost=None,
                remaining_amount=None,
                line_item_count=len(evaluation_input.line_items),
                violations=[BudgetViolation.CURRENCY_MISMATCH],
                requires_replanning=False,
            )

        total_amount = sum(
            (item.cost.amount for item in evaluation_input.line_items),
            start=Decimal("0"),
        )
        remaining_amount = budget.maximum.amount - total_amount

        if remaining_amount >= Decimal("0"):
            return BudgetAssessment(
                budget=budget,
                status=BudgetStatus.COMPLIANT,
                total_cost=Money(
                    amount=total_amount,
                    currency=budget_currency,
                ),
                remaining_amount=remaining_amount,
                line_item_count=len(evaluation_input.line_items),
                violations=[],
                requires_replanning=False,
            )

        return BudgetAssessment(
            budget=budget,
            status=BudgetStatus.EXCEEDED,
            total_cost=Money(
                amount=total_amount,
                currency=budget_currency,
            ),
            remaining_amount=remaining_amount,
            line_item_count=len(evaluation_input.line_items),
            violations=[BudgetViolation.BUDGET_EXCEEDED],
            requires_replanning=budget.is_hard_limit,
        )