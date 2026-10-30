"""Tests for critic-agent evaluation and deterministic overrides."""

from uuid import uuid4

from privateprocureai.constraints import BudgetEngine
from privateprocureai.evaluation import CriticAgent
from privateprocureai.llm import MockLLMProvider
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    BudgetConstraint,
    BudgetEvaluationInput,
    CostLineItem,
    EvaluationInput,
    EvaluationStatus,
    ExtractedIntent,
    Money,
)


def make_intent() -> ExtractedIntent:
    """Create generic intent for critic tests."""
    return ExtractedIntent(
        request_id=uuid4(),
        objective="Produce a supported recommendation.",
        constraints=[],
        preferences=[],
        assumptions=[],
        clarification_needed=False,
        clarification_questions=[],
        confidence_score=0.9,
    )


def make_completed_result() -> AgentResult:
    """Create a completed agent result for critic tests."""
    return AgentResult(
        task_id=uuid4(),
        agent_name="research_agent",
        status=AgentStatus.COMPLETED,
        summary="Research completed successfully.",
    )


def make_pass_analysis() -> dict:
    """Create valid simulated critic output."""
    return {
        "summary": "The available results adequately support the objective.",
        "criteria": [
            {
                "name": "Evidence coverage",
                "passed": True,
                "severity": "info",
                "rationale": "Available evidence supports the reviewed output.",
                "related_task_ids": [],
            }
        ],
    }


def test_critic_passes_valid_completed_results() -> None:
    """A successful execution with positive critic analysis passes."""
    critic = CriticAgent(MockLLMProvider(responses=[make_pass_analysis()]))

    report = critic.evaluate(
        EvaluationInput(
            intent=make_intent(),
            agent_results=[make_completed_result()],
        )
    )

    assert report.status is EvaluationStatus.PASS
    assert report.requires_replanning is False
    assert report.issues == []


def test_failed_agent_overrides_positive_critic_analysis() -> None:
    """Python fails evaluation even if the LLM incorrectly says everything passed."""
    critic = CriticAgent(MockLLMProvider(responses=[make_pass_analysis()]))

    failed_result = AgentResult(
        task_id=uuid4(),
        agent_name="research_agent",
        status=AgentStatus.FAILED,
        error_message="The assigned source could not be processed.",
    )

    report = critic.evaluate(
        EvaluationInput(
            intent=make_intent(),
            agent_results=[failed_result],
        )
    )

    assert report.status is EvaluationStatus.FAIL
    assert report.requires_replanning is True
    assert report.issues[0].code == "agent_failure"


def test_hard_budget_violation_overrides_positive_critic_analysis() -> None:
    """A hard budget breach always fails evaluation deterministically."""
    budget_assessment = BudgetEngine().evaluate(
        BudgetEvaluationInput(
            budget=BudgetConstraint(
                maximum=Money(amount="100.00", currency="USD"),
                is_hard_limit=True,
            ),
            line_items=[
                CostLineItem(
                    label="Selected item",
                    cost=Money(amount="125.00", currency="USD"),
                )
            ],
        )
    )

    critic = CriticAgent(MockLLMProvider(responses=[make_pass_analysis()]))

    report = critic.evaluate(
        EvaluationInput(
            intent=make_intent(),
            agent_results=[make_completed_result()],
            budget_assessment=budget_assessment,
        )
    )

    assert report.status is EvaluationStatus.FAIL
    assert report.requires_replanning is True
    assert report.issues[0].code == "hard_budget_violation"