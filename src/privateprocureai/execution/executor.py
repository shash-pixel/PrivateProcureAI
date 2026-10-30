"""Dependency-aware parallel execution of specialized agent tasks."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from privateprocureai.agents.base import TaskAgent
from privateprocureai.rag.retrieval import (
    EvidenceRetriever,
    build_retrieval_query,
)
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    AgentTaskContext,
    Evidence,
    ExtractedIntent,
    PlanExecutionResult,
    TaskPlan,
)


class AgentExecutionError(RuntimeError):
    """Raised when a task cannot be safely assigned or completed."""


class ParallelAgentExecutor:
    """Run independent plan tasks concurrently while respecting dependencies."""

    def __init__(
    self,
    agents: Sequence[TaskAgent],
    *,
    retriever: EvidenceRetriever | None = None,
    max_workers: int = 4,
) -> None:
        if not agents:
            raise ValueError("At least one task agent is required.")

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")

        self._agents = list(agents)
        self._retriever = retriever
        self._max_workers = max_workers

    def execute(
        self,
        *,
        intent: ExtractedIntent,
        task_plan: TaskPlan,
        evidence: Sequence[Evidence],
    ) -> PlanExecutionResult:
        """Execute a plan in dependency-respecting parallel batches."""
        pending_tasks = {task.task_id: task for task in task_plan.tasks}
        completed_results: dict = {}

        while pending_tasks:
            ready_tasks = [
                task
                for task in pending_tasks.values()
                if all(dependency_id in completed_results for dependency_id in task.depends_on)
            ]

            if not ready_tasks:
                raise AgentExecutionError(
                    "No executable task remains. A dependency may have failed."
                )

            with ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(ready_tasks))
            ) as executor:
                futures = {
                    executor.submit(
                        self._run_task,
                        intent=intent,
                        task=task,
                        evidence=list(evidence),
                        dependency_results=[
                            completed_results[dependency_id]
                            for dependency_id in task.depends_on
                        ],
                    ): task.task_id
                    for task in ready_tasks
                }

                for future in as_completed(futures):
                    task_id = futures[future]
                    result = future.result()

                    if result.status is AgentStatus.FAILED:
                        raise AgentExecutionError(
                            f"Task '{task_id}' failed: {result.error_message}"
                        )

                    completed_results[task_id] = result
                    del pending_tasks[task_id]

        ordered_results = [
            completed_results[task.task_id]
            for task in task_plan.tasks
        ]

        return PlanExecutionResult(
            plan_id=task_plan.plan_id,
            results=ordered_results,
        )

    def _run_task(
        self,
        *,
        intent: ExtractedIntent,
        task,
        evidence: list[Evidence],
        dependency_results: list[AgentResult],
    ) -> AgentResult:
        """Find a supporting agent and run one task."""
        agent = next(
            (candidate for candidate in self._agents if candidate.supports(task)),
            None,
        )

        if agent is None:
            raise AgentExecutionError(
                f"No registered agent supports task kind '{task.kind}'."
            )
        task_evidence = self._collect_evidence(
            intent=intent,
            task=task,
            supplied_evidence=evidence,
        )


        context = AgentTaskContext(
            intent=intent,
            task=task,
            evidence=task_evidence,
            dependency_results=dependency_results,
        )

        return agent.run(context)

    def _collect_evidence(
        self,
        *,
        intent: ExtractedIntent,
        task,
        supplied_evidence: list[Evidence],
) ->    list[Evidence]:
        """Combine caller-supplied and retrieved evidence without duplicates."""
        combined_evidence = list(supplied_evidence)

        if task.requires_retrieval:
            if self._retriever is None:
                raise AgentExecutionError(
                    "This task requires retrieval, but no evidence retriever is configured."
                )

            query = build_retrieval_query(intent, task)
            retrieval_result = self._retriever.retrieve(query)
            combined_evidence.extend(retrieval_result.evidence)

        unique_evidence: dict[str, Evidence] = {}

        for item in combined_evidence:
            unique_evidence.setdefault(item.source_id, item)

        return list(unique_evidence.values())    