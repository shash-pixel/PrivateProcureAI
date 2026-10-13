"""Tests for dependency-aware multi-agent execution."""

from uuid import uuid4

from privateprocureai.execution import ParallelAgentExecutor
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    ExtractedIntent,
    PlanTask,
    TaskKind,
    TaskPlan,
)


class TestAgent:
    """Small deterministic agent used to test executor behavior."""

    agent_name = "test_agent"

    def supports(self, task: PlanTask) -> bool:
        return task.kind in {TaskKind.RESEARCH, TaskKind.RECOMMENDATION}

    def run(self, context) -> AgentResult:
        return AgentResult(
            task_id=context.task.task_id,
            agent_name=self.agent_name,
            status=AgentStatus.COMPLETED,
            summary=f"Completed: {context.task.title}",
        )


def test_executor_runs_tasks_and_respects_dependencies() -> None:
    """A dependent task receives the completed result of its dependency."""
    research_task = PlanTask(
        title="Research requirements",
        instructions="Analyze supplied information.",
        kind=TaskKind.RESEARCH,
    )
    recommendation_task = PlanTask(
        title="Create recommendations",
        instructions="Create candidates from research findings.",
        kind=TaskKind.RECOMMENDATION,
        depends_on=[research_task.task_id],
    )

    plan = TaskPlan(
        request_id=uuid4(),
        planning_summary="Research before generating recommendations.",
        tasks=[research_task, recommendation_task],
    )

    result = ParallelAgentExecutor([TestAgent()]).execute(
        intent=ExtractedIntent(
            request_id=plan.request_id,
            objective="Produce a supported recommendation.",
            constraints=[],
            preferences=[],
            assumptions=[],
            clarification_needed=False,
            clarification_questions=[],
            confidence_score=0.9,
        ),
        task_plan=plan,
        evidence=[],
    )

    assert [item.task_id for item in result.results] == [
        research_task.task_id,
        recommendation_task.task_id,
    ]


def test_executor_runs_independent_tasks_in_one_batch() -> None:
    """Independent tasks can be scheduled without waiting for each other."""
    first_task = PlanTask(
        title="Research first information source",
        instructions="Analyze the first supplied source.",
        kind=TaskKind.RESEARCH,
    )
    second_task = PlanTask(
        title="Research second information source",
        instructions="Analyze the second supplied source.",
        kind=TaskKind.RESEARCH,
    )

    plan = TaskPlan(
        request_id=uuid4(),
        planning_summary="Analyze independent sources.",
        tasks=[first_task, second_task],
    )

    result = ParallelAgentExecutor([TestAgent()], max_workers=2).execute(
        intent=ExtractedIntent(
            request_id=plan.request_id,
            objective="Analyze supplied information.",
            constraints=[],
            preferences=[],
            assumptions=[],
            clarification_needed=False,
            clarification_questions=[],
            confidence_score=0.9,
        ),
        task_plan=plan,
        evidence=[],
    )

    assert len(result.results) == 2