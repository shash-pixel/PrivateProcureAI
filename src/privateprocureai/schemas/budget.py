"""Schemas for deterministic budget evaluation."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from privateprocureai.schemas.models import BudgetConstraint, DomainModel, Money


class BudgetStatus(StrEnum):
    """Possible outcomes from deterministic budget evaluation."""

    COMPLIANT = "compliant"
    EXCEEDED = "exceeded"
    UNEVALUABLE = "unevaluable"


class BudgetViolation(StrEnum):
    """Reasons a budget assessment may be non-compliant or incomplete."""

    BUDGET_EXCEEDED = "budget_exceeded"
    CURRENCY_MISMATCH = "currency_mismatch"


class CostLineItem(DomainModel):
    """One selected cost that contributes to a candidate plan total."""

    item_id: UUID = Field(default_factory=uuid4)
    label: str = Field(min_length=1, max_length=300)
    cost: Money
    recommendation_id: UUID | None = None


class BudgetEvaluationInput(DomainModel):
    """The explicit cost items to compare against one budget constraint."""

    budget: BudgetConstraint
    line_items: list[CostLineItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_item_ids(self) -> BudgetEvaluationInput:
        """Avoid accidental duplicate counting of one cost item."""
        item_ids = [item.item_id for item in self.line_items]

        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Each cost line item must have a unique item_id.")

        return self


class BudgetAssessment(DomainModel):
    """Result of deterministic budget evaluation."""

    budget: BudgetConstraint
    status: BudgetStatus
    total_cost: Money | None
    remaining_amount: Decimal | None
    line_item_count: int = Field(ge=0)
    violations: list[BudgetViolation] = Field(default_factory=list)
    requires_replanning: bool