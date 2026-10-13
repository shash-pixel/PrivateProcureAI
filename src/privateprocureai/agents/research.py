"""A domain-general agent for synthesizing supplied research evidence."""

from __future__ import annotations

from uuid import UUID

from privateprocureai.llm import StructuredGenerationRequest, StructuredLLM
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    AgentTaskContext,
    ResearchAgentOutput,
    TaskKind,
)

RESEARCH_AGENT_SYSTEM_INSTRUCTION = """
You are an evidence-grounded research agent.

Analyze only supplied evidence in relation to the extracted intent and assigned
research task. Every finding must cite one or more supplied evidence IDs.

Do not invent sources, facts, categories, prices, locations, or final
recommendations. Record unanswered questions in evidence_gaps.
""".strip()


class InvalidResearchTaskError(ValueError):
    """Raised when the research agent receives an unsupported task."""


class UnsupportedEvidenceReferenceError(ValueError):
    """Raised when output cites evidence that was not supplied."""


class ResearchAgent:
    """Produce source-linked findings for generic research tasks."""

    agent_name = "research_agent"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def supports(self, task) -> bool:
        """Return whether this is a research task."""
        return task.kind is TaskKind.RESEARCH

    def run(self, context: AgentTaskContext) -> AgentResult:
        """Execute one research task."""
        if not self.supports(context.task):
            raise InvalidResearchTaskError(
                "ResearchAgent supports only tasks with kind='research'."
            )

        request = StructuredGenerationRequest(
            system_instruction=RESEARCH_AGENT_SYSTEM_INSTRUCTION,
            user_prompt=(
                f"Intent:\n{context.intent.model_dump_json(indent=2)}\n\n"
                f"Task:\n{context.task.model_dump_json(indent=2)}\n\n"
                "Evidence:\n"
                f"{[item.model_dump(mode='json') for item in context.evidence]}"
            ),
            temperature=0.0,
        )

        output = self._llm.generate_structured(
            request=request,
            response_model=ResearchAgentOutput,
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
            findings=output.findings,
            evidence=context.evidence,
        )

    @staticmethod
    def _validate_evidence_references(
        output: ResearchAgentOutput,
        available_ids: set[UUID],
    ) -> None:
        """Reject findings that cite unavailable evidence."""
        cited_ids = {
            evidence_id
            for finding in output.findings
            for evidence_id in finding.evidence_ids
        }

        unsupported_ids = cited_ids - available_ids

        if unsupported_ids:
            raise UnsupportedEvidenceReferenceError(
                f"Unsupported evidence IDs: {sorted(str(item) for item in unsupported_ids)}"
            )