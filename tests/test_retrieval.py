"""Tests for vector retrieval and agent-executor integration."""

from uuid import uuid4

from privateprocureai.execution import ParallelAgentExecutor
from privateprocureai.rag.retrieval import RagRetriever
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    ExtractedIntent,
    PlanTask,
    RetrievalConfig,
    TaskKind,
    TaskPlan,
    VectorSearchMatch,
)


class FakeEmbeddingProvider:
    """Deterministic embedding provider with no Gemini API dependency."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    """Fixed vector matches for retrieval tests."""

    def search(self, query_embedding, *, limit: int):
        del query_embedding, limit

        return [
            VectorSearchMatch(
                chunk_id=uuid4(),
                document_id=uuid4(),
                text="Highly relevant source text.",
                distance=0.2,
                source_name="source.txt",
            ),
            VectorSearchMatch(
                chunk_id=uuid4(),
                document_id=uuid4(),
                text="Weakly related source text.",
                distance=1.5,
                source_name="source.txt",
            ),
        ]


class CapturingAgent:
    """Test agent that records evidence passed by the executor."""

    agent_name = "capturing_agent"

    def __init__(self) -> None:
        self.received_evidence_count = 0

    def supports(self, task: PlanTask) -> bool:
        return task.kind is TaskKind.RESEARCH

    def run(self, context) -> AgentResult:
        self.received_evidence_count = len(context.evidence)

        return AgentResult(
            task_id=context.task.task_id,
            agent_name=self.agent_name,
            status=AgentStatus.COMPLETED,
            summary="Completed research task.",
            evidence=context.evidence,
        )


def test_retriever_filters_weak_vector_matches() -> None:
    """Only matches above the relevance threshold become Evidence."""
    retriever = RagRetriever(
        FakeEmbeddingProvider(),
        FakeVectorStore(),
        RetrievalConfig(candidate_limit=5, minimum_relevance_score=0.6),
    )

    result = retriever.retrieve("Find relevant information.")

    assert len(result.evidence) == 1
    assert result.evidence[0].excerpt == "Highly relevant source text."
    assert result.discarded_match_count == 1


def test_executor_retrieves_evidence_for_retrieval_task() -> None:
    """A retrieval-enabled task passes vector evidence to its assigned agent."""
    task = PlanTask(
        title="Research supporting information",
        instructions="Analyze information relevant to the objective.",
        kind=TaskKind.RESEARCH,
        requires_retrieval=True,
    )

    plan = TaskPlan(
        request_id=uuid4(),
        planning_summary="Retrieve information before research synthesis.",
        tasks=[task],
    )

    retriever = RagRetriever(
        FakeEmbeddingProvider(),
        FakeVectorStore(),
        RetrievalConfig(minimum_relevance_score=0.6),
    )

    agent = CapturingAgent()

    result = ParallelAgentExecutor(
        [agent],
        retriever=retriever,
    ).execute(
        intent=ExtractedIntent(
            request_id=plan.request_id,
            objective="Analyze supplied information.",
            constraints=[],
            preferences=[],
            assumptions=[],
            clarification_needed=False,
            clarification_questions=[],
            confidence_score=0.9,
        ),
        task_plan=plan,
        evidence=[],
    )

    assert len(result.results) == 1
    assert agent.received_evidence_count == 1