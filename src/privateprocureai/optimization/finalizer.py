"""Final candidate selection with deterministic budget validation."""

from __future__ import annotations

from privateprocureai.constraints import BudgetEngine
from privateprocureai.llm import StructuredGenerationRequest, StructuredLLM
from privateprocureai.schemas import (
    BudgetEvaluationInput,
    BudgetStatus,
    CostLineItem,
    FinalOptimizationInput,
    FinalRecommendation,
    FinalRecommendationStatus,
    OptimizationDecision,
    OptimizationIssue,
    OptimizationIssueCode,
    Recommendation,
)

FINALIZER_SYSTEM_INSTRUCTION = """
You are a domain-general optimization agent.

Select the best candidate recommendations for the extracted intent. Use only
the supplied candidates, evidence references, constraints, and preferences.

Do not invent candidates, facts, prices, categories, locations, or sources.
Return only IDs from the supplied candidate recommendation list.
Do not assume that budget compliance has been validated; application code will
verify costs and hard constraints separately.
""".strip()


class UnknownRecommendationError(ValueError):
    """Raised when LLM output selects an ID that was never supplied."""


class FinalRecommendationOptimizer:
    """Select final candidates and independently validate budget feasibility."""

    def __init__(
        self,
        llm: StructuredLLM,
        budget_engine: BudgetEngine | None = None,
    ) -> None:
        self._llm = llm
        self._budget_engine = budget_engine or BudgetEngine()

    def optimize(
        self,
        optimization_input: FinalOptimizationInput,
    ) -> FinalRecommendation:
        """Select candidates through the LLM and validate them in Python."""
        candidates = self._collect_candidates(optimization_input)

        if not candidates:
            return FinalRecommendation(
                status=FinalRecommendationStatus.BLOCKED,
                summary="No candidate recommendations are available for selection.",
                selected_recommendations=[],
                issues=[
                    OptimizationIssue(
                        code=OptimizationIssueCode.UNKNOWN_RECOMMENDATION,
                        message="No agent produced a candidate recommendation.",
                    )
                ],
                requires_replanning=True,
            )

        decision = self._select_candidates(
            optimization_input=optimization_input,
            candidates=candidates,
        )

        selected_recommendations = self._resolve_selected_candidates(
            decision=decision,
            candidates=candidates,
        )

        return self._validate_final_selection(
            summary=decision.summary,
            selected_recommendations=selected_recommendations,
            budget=optimization_input.budget,
        )

    def _select_candidates(
        self,
        *,
        optimization_input: FinalOptimizationInput,
        candidates: list[Recommendation],
    ) -> OptimizationDecision:
        """Ask the LLM to rank and select only known candidate IDs."""
        request = StructuredGenerationRequest(
            system_instruction=FINALIZER_SYSTEM_INSTRUCTION,
            user_prompt=(
                f"Extracted intent:\n{optimization_input.intent.model_dump_json(indent=2)}"
                "\n\nCandidate recommendations:\n"
                f"{[candidate.model_dump(mode='json') for candidate in candidates]}"
            ),
            temperature=0.0,
        )

        return self._llm.generate_structured(
            request=request,
            response_model=OptimizationDecision,
        )

    @staticmethod
    def _collect_candidates(
        optimization_input: FinalOptimizationInput,
    ) -> list[Recommendation]:
        """Collect unique recommendations from every completed agent result."""
        candidates_by_id: dict = {}

        for result in optimization_input.agent_results:
            for recommendation in result.recommendations:
                candidates_by_id.setdefault(
                    recommendation.recommendation_id,
                    recommendation,
                )

        return list(candidates_by_id.values())

    @staticmethod
    def _resolve_selected_candidates(
        *,
        decision: OptimizationDecision,
        candidates: list[Recommendation],
    ) -> list[Recommendation]:
        """Map LLM-selected IDs back to application-owned candidate objects."""
        candidates_by_id = {
            candidate.recommendation_id: candidate
            for candidate in candidates
        }

        unknown_ids = [
            recommendation_id
            for recommendation_id in decision.selected_recommendation_ids
            if recommendation_id not in candidates_by_id
        ]

        if unknown_ids:
            raise UnknownRecommendationError(
                "The optimizer selected unknown recommendation IDs: "
                f"{', '.join(str(item) for item in unknown_ids)}"
            )

        return [
            candidates_by_id[recommendation_id]
            for recommendation_id in decision.selected_recommendation_ids
        ]

    def _validate_final_selection(
        self,
        *,
        summary: str,
        selected_recommendations: list[Recommendation],
        budget,
    ) -> FinalRecommendation:
        """Enforce deterministic cost and hard-budget rules."""
        if budget is None:
            return FinalRecommendation(
                status=FinalRecommendationStatus.READY,
                summary=summary,
                selected_recommendations=selected_recommendations,
                requires_replanning=False,
            )

        recommendations_without_cost = [
            recommendation
            for recommendation in selected_recommendations
            if recommendation.estimated_cost is None
        ]

        if recommendations_without_cost:
            return FinalRecommendation(
                status=FinalRecommendationStatus.BLOCKED,
                summary=summary,
                selected_recommendations=selected_recommendations,
                issues=[
                    OptimizationIssue(
                        code=OptimizationIssueCode.MISSING_COST_ESTIMATE,
                        message=(
                            "Every selected recommendation needs an estimated "
                            "cost when a budget constraint is present."
                        ),
                    )
                ],
                requires_replanning=True,
            )

        line_items = [
            CostLineItem(
                label=recommendation.title,
                cost=recommendation.estimated_cost,
                recommendation_id=recommendation.recommendation_id,
            )
            for recommendation in selected_recommendations
        ]

        budget_assessment = self._budget_engine.evaluate(
            BudgetEvaluationInput(
                budget=budget,
                line_items=line_items,
            )
        )

        if budget_assessment.status is BudgetStatus.UNEVALUABLE:
            return FinalRecommendation(
                status=FinalRecommendationStatus.BLOCKED,
                summary=summary,
                selected_recommendations=selected_recommendations,
                budget_assessment=budget_assessment,
                issues=[
                    OptimizationIssue(
                        code=OptimizationIssueCode.BUDGET_UNEVALUABLE,
                        message=(
                            "Selected costs cannot be compared because their "
                            "currencies do not match the budget currency."
                        ),
                    )
                ],
                requires_replanning=True,
            )

        if budget_assessment.requires_replanning:
            return FinalRecommendation(
                status=FinalRecommendationStatus.BLOCKED,
                summary=summary,
                selected_recommendations=selected_recommendations,
                budget_assessment=budget_assessment,
                issues=[
                    OptimizationIssue(
                        code=OptimizationIssueCode.HARD_BUDGET_VIOLATION,
                        message="Selected recommendations exceed a hard budget limit.",
                    )
                ],
                requires_replanning=True,
            )

        return FinalRecommendation(
            status=FinalRecommendationStatus.READY,
            summary=summary,
            selected_recommendations=selected_recommendations,
            budget_assessment=budget_assessment,
            requires_replanning=False,
        )