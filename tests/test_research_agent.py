"""Tests for the evidence-grounded research agent."""

from uuid import uuid4

import pytest

from privateprocureai.agents import ResearchAgent
from privateprocureai.agents.research import (
    InvalidResearchTaskError,
    UnsupportedEvidenceReferenceError,
)
from privateprocureai.llm import MockLLMProvider
from privateprocureai.schemas import (
    Evidence,
    ExtractedIntent,
    PlanTask,
    TaskKind,
)


def test_research_agent_returns_findings_linked_to_supplied_evidence() -> None:
    """Valid structured output remains traceable to provided evidence."""
    evidence = Evidence(
        source_id="source-a",
        excerpt="The source confirms that the stated requirement is supported.",
        relevance_score=0.9,
    )

    provider = MockLLMProvider(
        responses=[
            {
                "summary": "The supplied evidence supports one requirement.",
                "findings": [
                    {
                        "statement": "The requirement is supported by the source.",
                        "evidence_ids": [str(evidence.evidence_id)],
                        "confidence_score": 0.9,
                    }
                ],
                "evidence_gaps": [],
            }
        ]
    )

    result = ResearchAgent(provider).run(
        intent=ExtractedIntent(
            request_id=uuid4(),
            objective="Determine whether stated requirements are supported.",
            constraints=[],
            preferences=[],
            assumptions=[],
            clarification_needed=False,
            clarification_questions=[],
            confidence_score=0.9,
        ),
        task=PlanTask(
            title="Research available evidence",
            instructions="Assess evidence against the stated requirements.",
            kind=TaskKind.RESEARCH,
        ),
        evidence=[evidence],
    )

    assert result.agent_name == "research_agent"
    assert result.findings[0].evidence_ids == [evidence.evidence_id]


def test_research_agent_rejects_non_research_task() -> None:
    """Each agent has a meaningful, enforced responsibility."""
    agent = ResearchAgent(MockLLMProvider())

    with pytest.raises(InvalidResearchTaskError):
        agent.run(
            intent=ExtractedIntent(
                request_id=uuid4(),
                objective="Evaluate options.",
                constraints=[],
                preferences=[],
                assumptions=[],
                clarification_needed=False,
                clarification_questions=[],
                confidence_score=0.9,
            ),
            task=PlanTask(
                title="Evaluate options",
                instructions="Compare available options.",
                kind=TaskKind.EVALUATION,
            ),
            evidence=[],
        )


def test_research_agent_rejects_unsupported_evidence_reference() -> None:
    """The LLM cannot cite an identifier outside the supplied evidence."""
    supplied_evidence = Evidence(
        source_id="source-a",
        excerpt="A relevant source excerpt.",
        relevance_score=0.8,
    )

    provider = MockLLMProvider(
        responses=[
            {
                "summary": "A summary.",
                "findings": [
                    {
                        "statement": "An unsupported statement.",
                        "evidence_ids": [str(uuid4())],
                        "confidence_score": 0.7,
                    }
                ],
                "evidence_gaps": [],
            }
        ]
    )

    agent = ResearchAgent(provider)

    with pytest.raises(UnsupportedEvidenceReferenceError):
        agent.run(
            intent=ExtractedIntent(
                request_id=uuid4(),
                objective="Analyze evidence.",
                constraints=[],
                preferences=[],
                assumptions=[],
                clarification_needed=False,
                clarification_questions=[],
                confidence_score=0.9,
            ),
            task=PlanTask(
                title="Research evidence",
                instructions="Analyze supplied evidence.",
                kind=TaskKind.RESEARCH,
            ),
            evidence=[supplied_evidence],
        )