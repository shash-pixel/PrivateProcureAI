"""Tests for generic structured intent extraction."""

from privateprocureai.intent import IntentExtractor
from privateprocureai.llm import MockLLMProvider
from privateprocureai.schemas import PlanningRequest


def test_extractor_returns_validated_intent_for_original_request() -> None:
    """Structured LLM output is linked to the source planning request."""
    planning_request = PlanningRequest(
        goal="Organize a small workshop within the stated budget and timeline."
    )

    provider = MockLLMProvider(
        responses=[
            {
                "objective": "Organize a workshop that meets the stated constraints.",
                "constraints": [
                    {
                        "description": "Remain within the budget supplied by the user.",
                        "strength": "hard",
                        "source": "explicit",
                    }
                ],
                "preferences": [],
                "assumptions": [],
                "clarification_needed": False,
                "clarification_questions": [],
                "confidence_score": 0.9,
            }
        ]
    )

    intent = IntentExtractor(provider).extract(planning_request)

    assert intent.request_id == planning_request.request_id
    assert intent.objective.startswith("Organize a workshop")
    assert intent.constraints[0].strength == "hard"


def test_extractor_preserves_clarification_questions() -> None:
    """Missing details are made visible instead of silently invented."""
    planning_request = PlanningRequest(goal="Create a suitable plan.")

    provider = MockLLMProvider(
        responses=[
            {
                "objective": "Create a plan matching the user's intended outcome.",
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

    intent = IntentExtractor(provider).extract(planning_request)

    assert intent.clarification_needed is True
    assert len(intent.clarification_questions) == 1