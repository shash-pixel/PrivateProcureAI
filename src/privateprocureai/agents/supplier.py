"""A domain-general agent for evaluating suppliers, warranties, and credentials."""

from __future__ import annotations

from uuid import UUID

from privateprocureai.llm import StructuredGenerationRequest, StructuredLLM
from privateprocureai.observability import timed_step
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    AgentTaskContext,
    EvidenceClassification,
    ResearchAgentOutput,
    TaskKind,
)

SUPPLIER_AGENT_SYSTEM_INSTRUCTION = """
You are a domain-general supplier evaluation and credentials verification agent.

Analyze supplier evidence, business credentials, warranty terms, delivery timelines,
and commercial risk. Every finding must cite one or more supplied evidence IDs.

Explicitly report:
- Whether supplier credentials are confirmed or unverified.
- Warranty coverage duration and terms.
- Commercial exclusions or risk flags (e.g., refurbished equipment, lack of OEM warranty).
- Record missing verification information in evidence_gaps.
""".strip()


class InvalidSupplierTaskError(ValueError):
    """Raised when the supplier agent receives an unsupported task."""


class UnsupportedSupplierEvidenceError(ValueError):
    """Raised when output cites evidence that was not supplied."""


class SupplierAgent:
    """Evaluate supplier credibility, qualifications, and warranty offerings."""

    agent_name = "supplier_agent"

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def supports(self, task) -> bool:
        """Return whether this is a supplier evaluation or research task."""
        return task.kind in (TaskKind.SUPPLIER_EVALUATION, TaskKind.RESEARCH)

    def run(self, context: AgentTaskContext) -> AgentResult:
        """Execute one supplier qualification or warranty review task."""
        with timed_step("agent_execution", agent_name=self.agent_name) as timing:
            if not self.supports(context.task):
                raise InvalidSupplierTaskError(
                    "SupplierAgent supports only tasks with kind='supplier_evaluation' "
                    "or 'research'."
                )

            request = StructuredGenerationRequest(
                system_instruction=SUPPLIER_AGENT_SYSTEM_INSTRUCTION,
                user_prompt=(
                    f"Intent:\n{context.intent.model_dump_json(indent=2)}\n\n"
                    f"Supplier Task:\n{context.task.model_dump_json(indent=2)}\n\n"
                    "Supplied Evidence:\n"
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

            # Classify supplier verification from evidence
            unverified_evidence = [
                ev
                for ev in context.evidence
                if ev.classification
                in (EvidenceClassification.UNVERIFIED, EvidenceClassification.MISSING)
            ]
            warnings: list[str] = [f"Unresolved gap: {gap}" for gap in output.evidence_gaps]
            if unverified_evidence:
                warnings.append(
                    f"{len(unverified_evidence)} supplier source(s) are unverified "
                    "or missing authoritative verification."
                )

            tasks_performed = [
                f"Audited supplier credentials and warranty terms for {context.task.title}",
                f"Inspected {len(context.evidence)} source(s) for business qualification",
                f"Formulated {len(output.findings)} finding(s)",
            ]

            return AgentResult(
                task_id=context.task.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.COMPLETED,
                summary=output.summary,
                findings=output.findings,
                evidence=context.evidence,
                input_requirements=[c.description for c in context.intent.constraints],
                tasks_performed=tasks_performed,
                decisions_made=[
                    f"Qualified {len(output.findings)} finding(s) with supplier accreditation"
                ],
                warnings=warnings,
                unresolved_issues=list(output.evidence_gaps),
                duration_ms=timing.duration_ms,
            )

    @staticmethod
    def _validate_evidence_references(
        output: ResearchAgentOutput,
        available_ids: set[UUID],
    ) -> None:
        """Reject findings that cite unavailable evidence."""
        cited_ids = {
            evidence_id for finding in output.findings for evidence_id in finding.evidence_ids
        }
        unsupported_ids = cited_ids - available_ids
        if unsupported_ids:
            raise UnsupportedSupplierEvidenceError(
                f"Unsupported evidence IDs: {sorted(str(item) for item in unsupported_ids)}"
            )
