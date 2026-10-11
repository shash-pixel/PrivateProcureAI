"""Tests for generic task decomposition."""

import pytest
from pydantic import ValidationError

from privateprocureai.llm import MockLLMProvider
from privateprocureai.planning import PlanningBlockedError, TaskPlanner
from privateprocureai.schemas import ExtractedIntent, TaskPlanDraft


def test_planner_creates_uuid_backed_dependency_plan() -> None:
    """Symbolic LLM task dependencies become validated UUID dependencies."""
    intent = ExtractedIntent(
        request_id="12345678-1234-5678-1234-567812345678",
        objective="Produce a recommendation that satisfies stated requirements.",
        constraints=[],
        preferences=[],
        assumptions=[],
        clarification_needed=False,
        clarification_questions=[],
        confidence_score=0.9,
    )

    provider = MockLLMProvider(
        responses=[
            {
                "planning_summary": "Research available options before comparison.",
                "tasks": [
                    {
                        "task_key": "research_options",
                        "title": "Research available options",
                        "instructions": "Collect relevant options and supporting facts.",
                        "kind": "research",
                        "priority": "high",
                        "depends_on": [],
                        "requires_retrieval": True,
                    },
                    {
                        "task_key": "compare_options",
                        "title": "Compare options",
                        "instructions": "Compare research findings against constraints.",
                        "kind": "recommendation",
                        "priority": "normal",
                        "depends_on": ["research_options"],
                        "requires_retrieval": False,
                    },
                ],
            }
        ]
    )

    plan = TaskPlanner(provider).create_plan(intent)

    research_task, comparison_task = plan.tasks

    assert plan.request_id == intent.request_id
    assert comparison_task.depends_on == [research_task.task_id]
    assert research_task.task_id != comparison_task.task_id


def test_planner_stops_when_clarification_is_needed() -> None:
    """The system does not invent a plan when user intent remains unclear."""
    intent = ExtractedIntent(
        request_id="12345678-1234-5678-1234-567812345678",
        objective="Create a suitable plan.",
        constraints=[],
        preferences=[],
        assumptions=[],
        clarification_needed=True,
        clarification_questions=["What outcome should the plan optimize for?"],
        confidence_score=0.4,
    )

    with pytest.raises(PlanningBlockedError, match="Planning is blocked"):
        TaskPlanner(MockLLMProvider()).create_plan(intent)


def test_task_plan_draft_rejects_unknown_dependency() -> None:
    """LLM task references must point to a task in the same plan."""
    with pytest.raises(ValidationError, match="unknown dependencies"):
        TaskPlanDraft(
            planning_summary="A plan with invalid dependencies.",
            tasks=[
                {
                    "task_key": "evaluate",
                    "title": "Evaluate options",
                    "instructions": "Evaluate results against requirements.",
                    "kind": "evaluation",
                    "depends_on": ["missing_task"],
                }
            ],
        )


def test_task_plan_draft_rejects_dependency_cycles() -> None:
    """Mutually dependent tasks cannot be orchestrated safely."""
    with pytest.raises(ValidationError, match="must not contain a cycle"):
        TaskPlanDraft(
            planning_summary="A plan with a cyclic dependency.",
            tasks=[
                {
                    "task_key": "first_task",
                    "title": "First task",
                    "instructions": "Perform the first task.",
                    "kind": "research",
                    "depends_on": ["second_task"],
                },
                {
                    "task_key": "second_task",
                    "title": "Second task",
                    "instructions": "Perform the second task.",
                    "kind": "evaluation",
                    "depends_on": ["first_task"],
                },
            ],
        )