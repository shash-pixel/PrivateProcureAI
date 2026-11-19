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
            item.cost.currency != budget_currency for item in evaluation_input.line_items
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

        # Distinguish budget compliance from procurement feasibility and pricing completeness
        feasibility_notes: list[str] = []
        is_procurement_feasible = True
        is_scope_complete = True
        pricing_status = "complete"
        violations: list[BudgetViolation] = []

        # Check for zero or missing pricing in line items
        has_zero_cost = any(
            item.cost.amount == Decimal("0") for item in evaluation_input.line_items
        )
        if has_zero_cost:
            pricing_status = "incomplete"
            is_scope_complete = False
            feasibility_notes.append("One or more line items have unquoted or zero cost.")
            violations.append(BudgetViolation.INCOMPLETE_PRICING)

        # Check for implausibly low pricing (e.g. 25 workstations for under $100/unit)
        for item in evaluation_input.line_items:
            qty = getattr(item, "quantity", 1)
            if qty >= 5 and item.cost.amount < Decimal("100") * qty:
                is_procurement_feasible = False
                pricing_status = "implausible"
                feasibility_notes.append(
                    f"Line item '{item.label}' proposes {item.cost.amount} for quantity {qty}, "
                    "which is implausibly below market minimums."
                )
                if BudgetViolation.UNREALISTIC_COST not in violations:
                    violations.append(BudgetViolation.UNREALISTIC_COST)

        if remaining_amount >= Decimal("0"):
            status = BudgetStatus.COMPLIANT
            requires_replanning = False
        else:
            status = BudgetStatus.EXCEEDED
            violations.append(BudgetViolation.BUDGET_EXCEEDED)
            requires_replanning = budget.is_hard_limit

        return BudgetAssessment(
            budget=budget,
            status=status,
            total_cost=Money(
                amount=total_amount,
                currency=budget_currency,
            ),
            remaining_amount=remaining_amount,
            line_item_count=len(evaluation_input.line_items),
            violations=violations,
            requires_replanning=requires_replanning,
            is_procurement_feasible=is_procurement_feasible,
            is_scope_complete=is_scope_complete,
            pricing_status=pricing_status,
            feasibility_notes=feasibility_notes,
        )
