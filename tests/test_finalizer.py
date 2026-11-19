"""Tests for final recommendation selection and budget validation."""

from uuid import uuid4

import pytest

from privateprocureai.llm import MockLLMProvider
from privateprocureai.optimization import FinalRecommendationOptimizer
from privateprocureai.optimization.finalizer import UnknownRecommendationError
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    BudgetConstraint,
    ExtractedIntent,
    FinalOptimizationInput,
    FinalRecommendationStatus,
    Money,
    Recommendation,
)


def make_intent() -> ExtractedIntent:
    """Create generic intent for optimization tests."""
    return ExtractedIntent(
        request_id=uuid4(),
        objective="Produce a suitable recommendation.",
        constraints=[],
        preferences=[],
        assumptions=[],
        clarification_needed=False,
        clarification_questions=[],
        confidence_score=0.9,
    )


def make_recommendation(*, cost: str | None) -> Recommendation:
    """Create a generic candidate recommendation."""
    return Recommendation(
        title="Candidate option",
        rationale="This candidate aligns with the supplied requirements.",
        estimated_cost=(Money(amount=cost, currency="USD") if cost is not None else None),
        confidence_score=0.8,
    )


def test_optimizer_returns_budget_compliant_final_recommendation() -> None:
    """A valid selected candidate becomes ready after Python budget validation."""
    candidate = make_recommendation(cost="60.00")

    optimizer = FinalRecommendationOptimizer(
        MockLLMProvider(
            responses=[
                {
                    "summary": "The candidate best satisfies the available evidence.",
                    "selected_recommendation_ids": [str(candidate.recommendation_id)],
                }
            ]
        )
    )

    result = optimizer.optimize(
        FinalOptimizationInput(
            intent=make_intent(),
            agent_results=[
                AgentResult(
                    agent_name="recommendation_agent",
                    status=AgentStatus.COMPLETED,
                    recommendations=[candidate],
                )
            ],
            budget=BudgetConstraint(maximum=Money(amount="100.00", currency="USD")),
        )
    )

    assert result.status is FinalRecommendationStatus.READY
    assert result.budget_assessment is not None
    assert result.requires_replanning is False


def test_optimizer_blocks_selected_candidate_without_cost() -> None:
    """Budgeted selection cannot silently accept an unknown cost."""
    candidate = make_recommendation(cost=None)

    optimizer = FinalRecommendationOptimizer(
        MockLLMProvider(
            responses=[
                {
                    "summary": "The candidate is relevant.",
                    "selected_recommendation_ids": [str(candidate.recommendation_id)],
                }
            ]
        )
    )

    result = optimizer.optimize(
        FinalOptimizationInput(
            intent=make_intent(),
            agent_results=[
                AgentResult(
                    agent_name="recommendation_agent",
                    status=AgentStatus.COMPLETED,
                    recommendations=[candidate],
                )
            ],
            budget=BudgetConstraint(maximum=Money(amount="100.00", currency="USD")),
        )
    )

    assert result.status is FinalRecommendationStatus.BLOCKED
    assert result.requires_replanning is True
    assert result.issues[0].code == "missing_cost_estimate"


def test_optimizer_rejects_unknown_recommendation_id() -> None:
    """The LLM cannot select a candidate that application code never supplied."""
    candidate = make_recommendation(cost="10.00")

    optimizer = FinalRecommendationOptimizer(
        MockLLMProvider(
            responses=[
                {
                    "summary": "An invalid selection.",
                    "selected_recommendation_ids": [str(uuid4())],
                }
            ]
        )
    )

    with pytest.raises(UnknownRecommendationError):
        optimizer.optimize(
            FinalOptimizationInput(
                intent=make_intent(),
                agent_results=[
                    AgentResult(
                        agent_name="recommendation_agent",
                        status=AgentStatus.COMPLETED,
                        recommendations=[candidate],
                    )
                ],
            )
        )
