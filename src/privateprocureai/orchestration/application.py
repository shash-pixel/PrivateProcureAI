"""End-to-end application workflow integrating planning, agents, RAG, and signing."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from uuid import UUID

from privateprocureai.agents.base import TaskAgent
from privateprocureai.agents.recommendation import RecommendationAgent
from privateprocureai.agents.research import ResearchAgent
from privateprocureai.agents.supplier import SupplierAgent
from privateprocureai.constraints.budget import BudgetEngine
from privateprocureai.database.repositories import PostgresSignedRecommendationRepository
from privateprocureai.evaluation.critic import CriticAgent
from privateprocureai.execution.executor import ParallelAgentExecutor
from privateprocureai.integrity.signing import Ed25519ResultSigner
from privateprocureai.llm.base import StructuredLLM
from privateprocureai.observability import RequestContext, timed_step
from privateprocureai.optimization.finalizer import FinalRecommendationOptimizer
from privateprocureai.orchestration.replanning import ReplanningCoordinator
from privateprocureai.orchestration.workflow import PlanningWorkflow
from privateprocureai.planning.planner import TaskPlanner
from privateprocureai.rag.retrieval import EvidenceRetriever
from privateprocureai.schemas import (
    ApplicationWorkflowResult,
    ApplicationWorkflowStatus,
    BudgetAssessment,
    BudgetConstraint,
    BudgetEvaluationInput,
    CostLineItem,
    EvaluationInput,
    EvaluationIssueCode,
    EvaluationReport,
    Evidence,
    EvidenceClassification,
    FinalOptimizationInput,
    FinalRecommendationStatus,
    IntentDetails,
    PlanExecutionResult,
    PlanningRequest,
    ReplanningConfig,
    ReplanningIteration,
    ReplanningStatus,
    WorkflowEvent,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


class ApplicationWorkflow:
    """End-to-end multi-agent AI procurement and planning coordinator.

    Composes:
    - PlanningWorkflow (intent extraction & validated task planning)
    - Early exit when clarification is needed
    - Parallel specialist execution with selective RAG retrieval
    - Deterministic budget validation
    - Critic evaluation and bounded re-planning
    - Final recommendation optimization
    - Ed25519 cryptographic signing & verification
    - Optional PostgreSQL persistence
    """

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        planning_workflow: PlanningWorkflow | None = None,
        retriever: EvidenceRetriever | None = None,
        agents: Sequence[TaskAgent] | None = None,
        budget_engine: BudgetEngine | None = None,
        critic: CriticAgent | None = None,
        optimizer: FinalRecommendationOptimizer | None = None,
        signer: Ed25519ResultSigner | None = None,
        repository: PostgresSignedRecommendationRepository | None = None,
        max_replanning_iterations: int = 3,
    ) -> None:
        self._llm = llm
        self._planning_workflow = planning_workflow or PlanningWorkflow(llm)
        self._retriever = retriever
        self._budget_engine = budget_engine or BudgetEngine()
        self._critic = critic or CriticAgent(llm)
        self._signer = signer
        self._repository = repository
        self._max_replanning_iterations = max_replanning_iterations

        registered_agents = (
            list(agents)
            if agents is not None
            else [
                ResearchAgent(llm),
                SupplierAgent(llm),
                RecommendationAgent(llm),
            ]
        )
        self._executor = ParallelAgentExecutor(
            agents=registered_agents,
            retriever=retriever,
        )
        self._optimizer = optimizer or FinalRecommendationOptimizer(
            llm,
            budget_engine=self._budget_engine,
        )

    def run(self, request: PlanningRequest) -> ApplicationWorkflowResult:
        """Execute the full end-to-end procurement and planning workflow."""
        with RequestContext(request_id=request.request_id):
            events: list[WorkflowEvent] = []

            # Step 1 & 2: Intent extraction and planning via composed PlanningWorkflow
            planning_result = self._planning_workflow.run(request)
            events.extend(planning_result.events)

            if planning_result.status is WorkflowStatus.NEEDS_CLARIFICATION:
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.INTENT_EXTRACTION,
                        message="Workflow paused: user clarification required.",
                        status="needs_clarification",
                    )
                )
                return ApplicationWorkflowResult(
                    request_id=request.request_id,
                    status=ApplicationWorkflowStatus.NEEDS_CLARIFICATION,
                    intent=planning_result.intent,
                    task_plan=None,
                    events=events,
                )

            intent = planning_result.intent
            task_plan = planning_result.task_plan

            if task_plan is None or not task_plan.tasks:
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.TASK_PLANNING,
                        message="Workflow blocked: no executable tasks generated.",
                        status="blocked",
                    )
                )
                return ApplicationWorkflowResult(
                    request_id=request.request_id,
                    status=ApplicationWorkflowStatus.BLOCKED,
                    intent=intent,
                    task_plan=task_plan,
                    events=events,
                )

            # Step 3: Initial specialist agent execution with RAG retrieval
            try:
                with timed_step("agent_execution") as timing:
                    execution_result = self._executor.execute(
                        intent=intent,
                        task_plan=task_plan,
                        evidence=[],
                    )
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.AGENT_EXECUTION,
                        message=f"Executed {len(execution_result.results)} agent task(s).",
                        duration_ms=timing.duration_ms,
                        status="success",
                        event_type="agent",
                    )
                )
            except Exception as error:
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.AGENT_EXECUTION,
                        message=f"Agent execution failed: {error}",
                        status="failed",
                        event_type="agent",
                        error_details=str(error),
                    )
                )
                return ApplicationWorkflowResult(
                    request_id=request.request_id,
                    status=ApplicationWorkflowStatus.FAILED,
                    intent=intent,
                    task_plan=task_plan,
                    events=events,
                )

            # Collect evidence from all completed agent results
            evidence_by_id: dict[str, Evidence] = {}
            for agent_res in execution_result.results:
                for ev in agent_res.evidence:
                    evidence_by_id.setdefault(ev.source_id, ev)
            evidence_list = list(evidence_by_id.values())

            # Step 4: Deterministic budget validation
            budget_assessment = self._assess_budget(request.budget, execution_result)
            if budget_assessment is not None:
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.BUDGET_VALIDATION,
                        message=(
                            f"Budget assessment: {budget_assessment.status.value} "
                            f"(total: {budget_assessment.total_cost.amount} "
                            f"{budget_assessment.total_cost.currency})"
                        ),
                        status="success",
                    )
                )

            # Step 5: Critic evaluation
            with timed_step("evaluation") as timing:
                evaluation_report = self._critic.evaluate(
                    EvaluationInput(
                        intent=intent,
                        agent_results=execution_result.results,
                        budget_assessment=budget_assessment,
                    )
                )
            eval_status_str = "success" if not evaluation_report.requires_replanning else "failed"
            events.append(
                WorkflowEvent(
                    step=WorkflowStep.EVALUATION,
                    message=(
                        f"Critic evaluation {evaluation_report.status.value}: "
                        f"{evaluation_report.summary[:200]}"
                    ),
                    duration_ms=timing.duration_ms,
                    status=eval_status_str,
                )
            )

            # Step 6: Bounded re-planning loop when critic requests revisions
            replanning_iterations: list[ReplanningIteration] = []
            if evaluation_report.requires_replanning:
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.REPLANNING,
                        message="Critic required plan revisions; starting bounded loop.",
                        status="started",
                    )
                )

                def budget_builder(exec_res: PlanExecutionResult) -> BudgetAssessment | None:
                    return self._assess_budget(request.budget, exec_res)

                coordinator = ReplanningCoordinator(
                    planner=TaskPlanner(self._llm),
                    executor=self._executor,
                    critic=self._critic,
                    config=ReplanningConfig(max_iterations=self._max_replanning_iterations),
                    budget_assessment_builder=budget_builder,
                )

                with timed_step("replanning") as timing:
                    replanning_outcome = coordinator.run(
                        intent=intent,
                        evidence=evidence_list,
                    )
                replanning_iterations = replanning_outcome.iterations

                latest_iteration = replanning_iterations[-1]
                task_plan = latest_iteration.task_plan
                execution_result = latest_iteration.execution_result
                evaluation_report = latest_iteration.evaluation_report
                budget_assessment = self._assess_budget(request.budget, execution_result)

                # Re-harvest evidence
                for agent_res in execution_result.results:
                    for ev in agent_res.evidence:
                        evidence_by_id.setdefault(ev.source_id, ev)
                evidence_list = list(evidence_by_id.values())

                if (
                    replanning_outcome.status is ReplanningStatus.EXHAUSTED
                    and evaluation_report.requires_replanning
                ):
                    events.append(
                        WorkflowEvent(
                            step=WorkflowStep.REPLANNING,
                            message=(
                                f"Re-planning exhausted after "
                                f"{len(replanning_iterations)} iteration(s)."
                            ),
                            duration_ms=timing.duration_ms,
                            status="failed",
                        )
                    )
                    return ApplicationWorkflowResult(
                        request_id=request.request_id,
                        status=ApplicationWorkflowStatus.REPLANNING_EXHAUSTED,
                        intent=intent,
                        task_plan=task_plan,
                        agent_results=execution_result.results,
                        evidence=evidence_list,
                        budget_assessment=budget_assessment,
                        evaluation_report=evaluation_report,
                        replanning_iterations=replanning_iterations,
                        events=events,
                    )

                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.REPLANNING,
                        message=(
                            f"Re-planning succeeded after "
                            f"{len(replanning_iterations)} iteration(s)."
                        ),
                        duration_ms=timing.duration_ms,
                        status="success",
                    )
                )

            # Step 7: Final candidate optimization
            with timed_step("optimization") as timing:
                final_recommendation = self._optimizer.optimize(
                    FinalOptimizationInput(
                        intent=intent,
                        agent_results=execution_result.results,
                        budget=request.budget,
                    )
                )
            opt_status_str = (
                "success"
                if final_recommendation.status == FinalRecommendationStatus.READY
                else "blocked"
            )
            events.append(
                WorkflowEvent(
                    step=WorkflowStep.OPTIMIZATION,
                    message=(
                        f"Final selection {final_recommendation.status.value}: "
                        f"{final_recommendation.summary[:200]}"
                    ),
                    duration_ms=timing.duration_ms,
                    status=opt_status_str,
                )
            )

            # Step 8: Ed25519 signing and verification (only for READY recommendations)
            signed_recommendation = None
            verification = None
            if (
                final_recommendation.status is FinalRecommendationStatus.READY
                and self._signer is not None
            ):
                with timed_step("signing") as timing:
                    intent_hash = hashlib.sha256(
                        intent.model_dump_json().encode("utf-8")
                    ).hexdigest()
                    signed_recommendation = self._signer.sign(
                        final_recommendation,
                        request_id=request.request_id,
                        intent_hash_sha256=intent_hash,
                        evidence_ids=[ev.evidence_id for ev in evidence_list],
                        budget_status=budget_assessment.status.value if budget_assessment else None,
                        evaluation_status=evaluation_report.status.value
                        if evaluation_report
                        else None,
                    )
                    verification = self._signer.verify(signed_recommendation)
                fp = signed_recommendation.integrity.public_key_fingerprint_sha256[:12]
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.SIGNING,
                        message=(
                            f"Signed with Ed25519 (fingerprint: {fp}...), "
                            f"valid={verification.is_valid}"
                        ),
                        duration_ms=timing.duration_ms,
                        status="success" if verification.is_valid else "failed",
                    )
                )

            # Step 9: Optional persistence to PostgreSQL
            persisted_id: UUID | None = None
            if signed_recommendation is not None and self._repository is not None:
                with timed_step("persistence") as timing:
                    persisted_id = self._repository.save(
                        request_id=request.request_id,
                        signed_recommendation=signed_recommendation,
                    )
                events.append(
                    WorkflowEvent(
                        step=WorkflowStep.PERSISTENCE,
                        message=f"Saved signed recommendation record {persisted_id} to database.",
                        duration_ms=timing.duration_ms,
                        status="success",
                    )
                )

            terminal_status = (
                ApplicationWorkflowStatus.COMPLETED
                if final_recommendation.status is FinalRecommendationStatus.READY
                else ApplicationWorkflowStatus.BLOCKED
            )

            confidence_breakdown = self._compute_confidence_breakdown(
                intent=intent,
                evidence=evidence_list,
                evaluation_report=evaluation_report,
                budget_assessment=budget_assessment,
            )

            return ApplicationWorkflowResult(
                request_id=request.request_id,
                status=terminal_status,
                intent=intent,
                task_plan=task_plan,
                agent_results=execution_result.results,
                evidence=evidence_list,
                budget_assessment=budget_assessment,
                evaluation_report=evaluation_report,
                replanning_iterations=replanning_iterations,
                final_recommendation=final_recommendation,
                signed_recommendation=signed_recommendation,
                verification=verification,
                persisted_result_id=persisted_id,
                confidence_breakdown=confidence_breakdown,
                events=events,
            )

    def _assess_budget(
        self,
        budget: BudgetConstraint | None,
        execution_result: PlanExecutionResult,
    ) -> BudgetAssessment | None:
        """Extract recommendation costs and perform deterministic budget evaluation."""
        if budget is None:
            return None

        line_items: list[CostLineItem] = []
        for result in execution_result.results:
            for recommendation in result.recommendations:
                if recommendation.estimated_cost is not None:
                    line_items.append(
                        CostLineItem(
                            label=recommendation.title,
                            cost=recommendation.estimated_cost,
                            recommendation_id=recommendation.recommendation_id,
                        )
                    )

        if not line_items:
            return None

        return self._budget_engine.evaluate(
            BudgetEvaluationInput(
                budget=budget,
                line_items=line_items,
            )
        )

    @staticmethod
    def _compute_confidence_breakdown(
        *,
        intent: IntentDetails,
        evidence: list[Evidence],
        evaluation_report: EvaluationReport | None,
        budget_assessment: BudgetAssessment | None,
    ) -> dict[str, float]:
        """Compute an explainable multi-factor confidence score."""
        intent_score = 1.0 if not intent.clarification_needed else 0.4
        if intent.confidence_score:
            intent_score = min(intent_score, intent.confidence_score)

        if evidence:
            rel_score = sum(ev.relevance_score for ev in evidence) / len(evidence)
            verified_count = sum(
                1 for ev in evidence if ev.classification == EvidenceClassification.VERIFIED
            )
            rel_reliability = verified_count / len(evidence)
        else:
            rel_score = 0.0
            rel_reliability = 0.0

        if evaluation_report is not None:
            has_req_failure = any(
                iss.code == EvaluationIssueCode.REQUIREMENT_MISMATCH
                for iss in evaluation_report.issues
            )
            has_critic_fail = evaluation_report.requires_replanning
            req_coverage = 0.3 if has_req_failure else (0.6 if has_critic_fail else 1.0)
        else:
            req_coverage = 0.5

        overall = round(
            0.25 * intent_score + 0.25 * rel_score + 0.25 * rel_reliability + 0.25 * req_coverage,
            2,
        )

        return {
            "intent_extraction": round(intent_score, 2),
            "evidence_relevance": round(rel_score, 2),
            "evidence_reliability": round(rel_reliability, 2),
            "requirement_coverage": round(req_coverage, 2),
            "overall_confidence": overall,
        }
