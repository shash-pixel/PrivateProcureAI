"""LangGraph orchestration for intent extraction and task planning."""

##It connects my Intent Extraction module (Module 4) and Task Planning module (Module 5) using LangGraph.

from __future__ import annotations

from typing import Literal, cast

from langgraph.graph import END, START, StateGraph

from privateprocureai.intent import IntentExtractor
from privateprocureai.llm import StructuredLLM
from privateprocureai.orchestration.state import PlanningWorkflowState
from privateprocureai.planning import TaskPlanner
from privateprocureai.schemas import (
    PlanningRequest,
    WorkflowEvent,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)


class PlanningWorkflow:
    """Coordinate the completed intent and planning modules through LangGraph."""

    def __init__(self, llm: StructuredLLM) -> None:
        self._intent_extractor = IntentExtractor(llm)
        self._task_planner = TaskPlanner(llm)
        self._graph = self._build_graph()

    def run(self, planning_request: PlanningRequest) -> WorkflowResult:
        """Run intent extraction and, when possible, task planning."""
        final_state = cast(
            PlanningWorkflowState,
            self._graph.invoke(
                {
                    "planning_request": planning_request,
                    "execution_log": [],
                }
            ),
        )

        intent = final_state["extracted_intent"]
        task_plan = final_state.get("task_plan")
        status = (
            WorkflowStatus.NEEDS_CLARIFICATION
            if intent.clarification_needed
            else WorkflowStatus.PLANNED
        )

        return WorkflowResult(
            request_id=planning_request.request_id,
            status=status,
            intent=intent,
            task_plan=task_plan,
            events=final_state.get("execution_log", []),
        )

    def _build_graph(self):
        """Build the deterministic Module 6 workflow graph."""
        builder = StateGraph(PlanningWorkflowState)

        builder.add_node("extract_intent", self._extract_intent)
        builder.add_node("create_plan", self._create_plan)

        builder.add_edge(START, "extract_intent")
        builder.add_conditional_edges(
            "extract_intent",
            self._route_after_intent,
            {
                "create_plan": "create_plan",
                "end": END,
            },
        )
        builder.add_edge("create_plan", END)

        return builder.compile()

    def _extract_intent(
        self,
        state: PlanningWorkflowState,
    ) -> dict[str, object]:
        """Run Module 4 intent extraction and record its completion."""
        intent = self._intent_extractor.extract(state["planning_request"])

        return {
            "extracted_intent": intent,
            "execution_log": [
                WorkflowEvent(
                    step=WorkflowStep.INTENT_EXTRACTION,
                    message="Extracted structured intent from the planning request.",
                )
            ],
        }

    @staticmethod
    def _route_after_intent(   ##This method decides what happens immediately after intent extraction.
        state: PlanningWorkflowState,
    ) -> Literal["create_plan", "end"]:
        """Stop safely when clarification is needed; otherwise continue."""
        if state["extracted_intent"].clarification_needed:
            return "end"

        return "create_plan"

    def _create_plan(
        self,
        state: PlanningWorkflowState,
    ) -> dict[str, object]:
        """Run Module 5 planning after intent has been validated."""
        task_plan = self._task_planner.create_plan(state["extracted_intent"])

        return {
            "task_plan": task_plan,
            "execution_log": [
                WorkflowEvent(
                    step=WorkflowStep.TASK_PLANNING,
                    message="Created a validated dependency-aware task plan.",
                )
            ],
        }