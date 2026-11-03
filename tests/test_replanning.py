"""Tests for bounded iterative re-planning."""

from uuid import uuid4

from privateprocureai.orchestration import ReplanningCoordinator
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    EvaluationReport,
    EvaluationStatus,
    ExtractedIntent,
    PlanExecutionResult,
    PlanTask,
    ReplanningConfig,
    TaskKind,
    TaskPlan,
)


class SequencePlanner:
    """Test planner that records feedback received on each iteration."""

    def __init__(self) -> None:
        self.feedback_history: list[list[str]] = []

    def create_plan(self, intent, *, feedback=None) -> TaskPlan:
        self.feedback_history.append(feedback or [])

        return TaskPlan(
            request_id=intent.request_id,
            planning_summary="Create one generic research task.",
            tasks=[
                PlanTask(
                    title="Research supporting information",
                    instructions="Analyze relevant supplied information.",
                    kind=TaskKind.RESEARCH,
                )
            ],
        )


class SuccessfulExecutor:
    """Test executor that always returns a completed agent result."""

    def execute(self, *, intent, task_plan, evidence) -> PlanExecutionResult:
        del intent, evidence

        return PlanExecutionResult(
            plan_id=task_plan.plan_id,
            results=[
                AgentResult(
                    task_id=task_plan.tasks[0].task_id,
                    agent_name="research_agent",
                    status=AgentStatus.COMPLETED,
                    summary="Research completed.",
                )
            ],
        )


class SequenceCritic:
    """Test critic that fails once and then passes."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, evaluation_input) -> EvaluationReport:
        del evaluation_input
        self.calls += 1

        if self.calls == 1:
            return EvaluationReport(
                status=EvaluationStatus.FAIL,
                summary="Evidence coverage is incomplete.",
                criteria=[],
                issues=[
                    {
                        "code": "critical_quality_concern",
                        "severity": "critical",
                        "message": "Add evidence for the unresolved requirement.",
                    }
                ],
                requires_replanning=True,
            )

        return EvaluationReport(
            status=EvaluationStatus.PASS,
            summary="The revised plan is adequate.",
            criteria=[],
            issues=[],
            requires_replanning=False,
        )


def make_intent() -> ExtractedIntent:
    """Create generic planning intent for loop tests."""
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


def test_replanning_uses_critic_feedback_for_second_attempt() -> None:
    """A failed first attempt creates a feedback-informed second plan."""
    planner = SequencePlanner()

    outcome = ReplanningCoordinator(
        planner=planner,
        executor=SuccessfulExecutor(),
        critic=SequenceCritic(),
        config=ReplanningConfig(max_iterations=3),
    ).run(intent=make_intent())

    assert outcome.status.value == "completed"
    assert len(outcome.iterations) == 2
    assert planner.feedback_history[0] == []
    assert planner.feedback_history[1] == [
        "Add evidence for the unresolved requirement."
    ]


def test_replanning_stops_when_maximum_attempts_are_exhausted() -> None:
    """Repeated failures cannot create an unbounded loop."""
    planner = SequencePlanner()

    outcome = ReplanningCoordinator(
        planner=planner,
        executor=SuccessfulExecutor(),
        critic=SequenceCritic(),
        config=ReplanningConfig(max_iterations=1),
    ).run(intent=make_intent())

    assert outcome.status.value == "exhausted"
    assert len(outcome.iterations) == 1