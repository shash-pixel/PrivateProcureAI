"""A domain-general agent for evidence-grounded candidate recommendations."""

from __future__ import annotations

from uuid import UUID

from privateprocureai.llm import StructuredGenerationRequest, StructuredLLM
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    AgentTaskContext,
    RecommendationAgentOutput,
    TaskKind,
)

RECOMMENDATION_AGENT_SYSTEM_INSTRUCTION = """
You are an evidence-grounded recommendation agent.

Use the extracted intent, assigned task, prior agent results, and supplied
evidence to produce candidate recommendations.

Every recommendation must cite one or more supplied evidence IDs.
Do not invent facts, sources, prices, locations, categories, or requirements.
Do not claim that a candidate satisfies a constraint unless supplied evidence
supports that claim. Record missing information in evidence_gaps.
""".strip()


class InvalidRecommendationTaskError(ValueError):
    """Raised when the recommendation agent receives an unsupported task."""


class UnsupportedRecommendationEvidenceError(ValueError):
    """Raised when a recommendation cites unavailable evidence."""


class RecommendationAgent:
    """Create evidence-linked candidate recommendations."""

    agent_name = "recommendation_agent"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def supports(self, task) -> bool:
        """Return whether this is a recommendation task."""
        return task.kind is TaskKind.RECOMMENDATION

    def run(self, context: AgentTaskContext) -> AgentResult:
        """Execute one recommendation task."""
        if not self.supports(context.task):
            raise InvalidRecommendationTaskError(
                "RecommendationAgent supports only tasks with kind='recommendation'."
            )

        request = StructuredGenerationRequest(
            system_instruction=RECOMMENDATION_AGENT_SYSTEM_INSTRUCTION,
            user_prompt=(
                f"Intent:\n{context.intent.model_dump_json(indent=2)}\n\n"
                f"Task:\n{context.task.model_dump_json(indent=2)}\n\n"
                "Prior agent results:\n"
                f"{[item.model_dump(mode='json') for item in context.dependency_results]}"
                "\n\nEvidence:\n"
                f"{[item.model_dump(mode='json') for item in context.evidence]}"
            ),
            temperature=0.0,
        )

        output = self._llm.generate_structured(
            request=request,
            response_model=RecommendationAgentOutput,
        )

        self._validate_evidence_references(
            output,
            {item.evidence_id for item in context.evidence},
        )

        return AgentResult(
            task_id=context.task.task_id,
            agent_name=self.agent_name,
            status=AgentStatus.COMPLETED,
            summary=output.summary,
            recommendations=output.recommendations,
            evidence=context.evidence,
        )

    @staticmethod
    def _validate_evidence_references(
        output: RecommendationAgentOutput,
        available_ids: set[UUID],
    ) -> None:
        """Reject recommendations that cite unavailable evidence."""
        cited_ids = {
            evidence_id
            for recommendation in output.recommendations
            for evidence_id in recommendation.evidence_ids
        }

        unsupported_ids = cited_ids - available_ids

        if unsupported_ids:
            raise UnsupportedRecommendationEvidenceError(
                "Recommendations cited unavailable evidence IDs: "
                f"{sorted(str(item) for item in unsupported_ids)}"
            )