"""Integration tests for the Module 6 LangGraph workflow."""

from privateprocureai.llm import MockLLMProvider
from privateprocureai.orchestration import PlanningWorkflow
from privateprocureai.schemas import PlanningRequest, WorkflowStatus


def test_workflow_extracts_intent_and_creates_plan() -> None:
    """A clear request proceeds through both graph nodes."""
    provider = MockLLMProvider(
        responses=[
            {
                "objective": "Produce a recommendation satisfying stated requirements.",
                "constraints": [],
                "preferences": [],
                "assumptions": [],
                "clarification_needed": False,
                "clarification_questions": [],
                "confidence_score": 0.9,
            },
            {
                "planning_summary": "Research information before evaluating options.",
                "tasks": [
                    {
                        "task_key": "research",
                        "title": "Research relevant information",
                        "instructions": "Collect evidence relevant to the objective.",
                        "kind": "research",
                        "priority": "high",
                        "depends_on": [],
                        "requires_retrieval": True,
                    },
                    {
                        "task_key": "evaluate",
                        "title": "Evaluate available options",
                        "instructions": "Compare findings against identified constraints.",
                        "kind": "evaluation",
                        "priority": "normal",
                        "depends_on": ["research"],
                        "requires_retrieval": False,
                    },
                ],
            },
        ]
    )

    request = PlanningRequest(
        goal="Provide a recommendation that meets the supplied requirements."
    )

    result = PlanningWorkflow(provider).run(request)

    assert result.status is WorkflowStatus.PLANNED   # Checks whether the workflow behaved as expected
    assert result.task_plan is not None
    assert len(result.task_plan.tasks) == 2
    assert [event.step for event in result.events] == [
        "intent_extraction",
        "task_planning",
    ]


def test_workflow_stops_when_clarification_is_needed() -> None:
    """A vague request ends safely without creating a speculative plan."""
    provider = MockLLMProvider(
        responses=[
            {
                "objective": "Create a suitable plan.",
                "constraints": [],
                "preferences": [],
                "assumptions": [],
                "clarification_needed": True,
                "clarification_questions": [
                    "What outcome should the plan optimize for?"
                ],
                "confidence_score": 0.4,
            }
        ]
    )

    request = PlanningRequest(goal="Create a plan.")
    result = PlanningWorkflow(provider).run(request)

    assert result.status is WorkflowStatus.NEEDS_CLARIFICATION   # Confirms that the workflow stopped correctly.
    assert result.task_plan is None
    assert [event.step for event in result.events] == ["intent_extraction"]