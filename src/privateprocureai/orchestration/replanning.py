"""Bounded LangGraph workflow for iterative plan improvement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from privateprocureai.evaluation import CriticAgent
from privateprocureai.execution import AgentExecutionError, ParallelAgentExecutor
from privateprocureai.planning import TaskPlanner
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    BudgetAssessment,
    EvaluationInput,
    Evidence,
    ExtractedIntent,
    PlanExecutionResult,
    ReplanningConfig,
    ReplanningIteration,
    ReplanningOutcome,
    ReplanningStatus,
    TaskPlan,
)

BudgetAssessmentBuilder = Callable[[PlanExecutionResult], BudgetAssessment | None]


class ReplanningState(TypedDict, total=False):
    """Internal state shared by LangGraph re-planning nodes."""

    intent: ExtractedIntent
    evidence: list[Evidence]
    iteration_number: int
    feedback: list[str]

    current_plan: TaskPlan
    current_execution: PlanExecutionResult
    current_evaluation: object

    history: list[ReplanningIteration]


class ReplanningCoordinator:
    """Create, execute, evaluate, and revise plans within a safe attempt limit."""

    def __init__(
        self,
        *,
        planner: TaskPlanner,
        executor: ParallelAgentExecutor,
        critic: CriticAgent,
        config: ReplanningConfig | None = None,
        budget_assessment_builder: BudgetAssessmentBuilder | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._critic = critic
        self._config = config or ReplanningConfig()
        self._budget_assessment_builder = budget_assessment_builder
        self._graph = self._build_graph()

    def run(
        self,
        *,
        intent: ExtractedIntent,
        evidence: Sequence[Evidence] = (),
    ) -> ReplanningOutcome:
        """Run the bounded plan-execute-evaluate loop."""
        final_state = cast(
            ReplanningState,
            self._graph.invoke(
                {
                    "intent": intent,
                    "evidence": list(evidence),
                    "iteration_number": 0,
                    "feedback": [],
                    "history": [],
                }
            ),
        )

        history = final_state["history"]
        latest_evaluation = history[-1].evaluation_report

        return ReplanningOutcome(
            status=(
                ReplanningStatus.COMPLETED
                if not latest_evaluation.requires_replanning
                else ReplanningStatus.EXHAUSTED
            ),
            iterations=history,
        )

    def _build_graph(self):
        """Build the deterministic LangGraph re-planning loop."""
        builder = StateGraph(ReplanningState)

        builder.add_node("create_plan", self._create_plan)
        builder.add_node("execute_plan", self._execute_plan)
        builder.add_node("evaluate_plan", self._evaluate_plan)
        builder.add_node("record_iteration", self._record_iteration)

        builder.add_edge(START, "create_plan")
        builder.add_edge("create_plan", "execute_plan")
        builder.add_edge("execute_plan", "evaluate_plan")
        builder.add_edge("evaluate_plan", "record_iteration")
        builder.add_conditional_edges(
            "record_iteration",
            self._route_after_evaluation,
            {
                "create_plan": "create_plan",
                "end": END,
            },
        )

        return builder.compile()

    def _create_plan(self, state: ReplanningState) -> dict[str, object]:
        """Create the initial or feedback-informed replacement plan."""
        plan = self._planner.create_plan(
            state["intent"],
            feedback=state["feedback"],
        )

        return {
            "current_plan": plan,
            "iteration_number": state["iteration_number"] + 1,
        }

    def _execute_plan(self, state: ReplanningState) -> dict[str, object]:
        """Execute the current plan and normalize executor failures."""
        try:
            execution_result = self._executor.execute(
                intent=state["intent"],
                task_plan=state["current_plan"],
                evidence=state["evidence"],
            )
        except AgentExecutionError as error:
            execution_result = PlanExecutionResult(
                plan_id=state["current_plan"].plan_id,
                results=[
                    AgentResult(
                        agent_name="executor",
                        status=AgentStatus.FAILED,
                        error_message=str(error),
                    )
                ],
            )

        return {"current_execution": execution_result}

    def _evaluate_plan(self, state: ReplanningState) -> dict[str, object]:
        """Evaluate the latest execution and optional deterministic budget result."""
        budget_assessment = None

        if self._budget_assessment_builder is not None:
            budget_assessment = self._budget_assessment_builder(
                state["current_execution"]
            )

        evaluation = self._critic.evaluate(
            EvaluationInput(
                intent=state["intent"],
                agent_results=state["current_execution"].results,
                budget_assessment=budget_assessment,
            )
        )

        return {"current_evaluation": evaluation}

    def _record_iteration(self, state: ReplanningState) -> dict[str, object]:
        """Save the completed attempt and derive concise planner feedback."""
        evaluation = state["current_evaluation"]
        feedback = [issue.message for issue in evaluation.issues]

        iteration = ReplanningIteration(
            iteration_number=state["iteration_number"],
            task_plan=state["current_plan"],
            execution_result=state["current_execution"],
            evaluation_report=evaluation,
            feedback_for_next_iteration=feedback,
        )

        return {
            "history": [*state["history"], iteration],
            "feedback": feedback,
        }

    def _route_after_evaluation(
        self,
        state: ReplanningState,
    ) -> Literal["create_plan", "end"]:
        """Retry only when evaluation fails and safe attempts remain."""
        evaluation = state["current_evaluation"]

        if (
            evaluation.requires_replanning
            and state["iteration_number"] < self._config.max_iterations
        ):
            return "create_plan"

        return "end"