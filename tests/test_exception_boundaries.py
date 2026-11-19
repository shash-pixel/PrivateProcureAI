"""Tests for exception boundaries across provider, retrieval, agent, and database layers."""

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

from privateprocureai.database.exceptions import DatabaseUnavailableError
from privateprocureai.database.session import initialize_database
from privateprocureai.execution.executor import AgentExecutionError, ParallelAgentExecutor
from privateprocureai.integrity.signing import (
    Ed25519ResultSigner,
    SigningError,
    generate_ed25519_keypair,
)
from privateprocureai.llm.base import StructuredGenerationRequest
from privateprocureai.llm.exceptions import LLMProviderError, StructuredOutputError
from privateprocureai.llm.mock import MockLLMProvider
from privateprocureai.rag.retrieval import RagRetriever, RetrievalError
from privateprocureai.schemas import (
    ExtractedIntent,
    FinalRecommendation,
    FinalRecommendationStatus,
    PlanTask,
    TaskKind,
    TaskPlan,
    TaskPriority,
)


class DummyModel(BaseModel):
    value: str


def test_provider_failure_boundary() -> None:
    """When an LLM provider fails, LLMProviderError is raised without leaking secrets."""
    provider = MockLLMProvider(responses=[])
    request = StructuredGenerationRequest(
        system_instruction="sys",
        user_prompt="prompt with key AIzaSyD-1234567890abcdefghijklmnopqrst",
    )

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate_structured(request, DummyModel)

    assert "no queued response" in str(exc_info.value).lower()


def test_malformed_llm_output_boundary() -> None:
    """When LLM returns malformed JSON, StructuredOutputError is raised."""
    provider = MockLLMProvider(responses=[{"wrong_key": 123}])
    request = StructuredGenerationRequest(system_instruction="sys", user_prompt="test")

    with pytest.raises(StructuredOutputError) as exc_info:
        provider.generate_structured(request, DummyModel)

    assert "DummyModel" in str(exc_info.value)


def test_retrieval_failure_boundary() -> None:
    """When the vector store or embedding fails during retrieval, RetrievalError is raised."""

    class FailingEmbeddingProvider:
        def embed_query(self, query: str):
            raise RuntimeError("Underlying embedding connection refused")

    class DummyVectorStore:
        def search(self, *args, **kwargs):
            return []

    retriever = RagRetriever(
        embedding_provider=FailingEmbeddingProvider(),
        vector_store=DummyVectorStore(),
    )

    with pytest.raises(RetrievalError) as exc_info:
        retriever.retrieve("find equipment")

    assert "Evidence retrieval failed" in str(exc_info.value)
    assert "connection refused" in str(exc_info.value)


def test_agent_failure_boundary() -> None:
    """When an agent fails or is unsupported, AgentExecutionError is raised with context."""
    intent = ExtractedIntent(
        request_id=uuid4(),
        objective="Test",
        constraints=[],
        preferences=[],
        assumptions=[],
        clarification_needed=False,
        clarification_questions=[],
        confidence_score=0.9,
    )
    plan = TaskPlan(
        request_id=intent.request_id,
        plan_id=uuid4(),
        planning_summary="Test plan",
        tasks=[
            PlanTask(
                task_id=uuid4(),
                title="Unsupported",
                instructions="Do something",
                kind=TaskKind.BUDGET_ANALYSIS,
                priority=TaskPriority.NORMAL,
                depends_on=[],
                requires_retrieval=False,
            )
        ],
    )

    class DummyAgent:
        agent_name = "dummy"

        def supports(self, task):
            return False

        def run(self, context):
            raise NotImplementedError

    executor = ParallelAgentExecutor(agents=[DummyAgent()])

    with pytest.raises(AgentExecutionError) as exc_info:
        executor.execute(intent=intent, task_plan=plan, evidence=[])

    assert "No registered agent supports task kind" in str(exc_info.value)


def test_database_unavailable_boundary() -> None:
    """When database operations fail, DatabaseUnavailableError is raised with clean message."""
    from unittest.mock import MagicMock

    from sqlalchemy.exc import OperationalError

    mock_engine = MagicMock()
    # Mocking metadata create_all to simulate unreachable postgresql
    with pytest.MonkeyPatch.context() as mp:

        def failing_create_all(*args, **kwargs):
            raise OperationalError(
                "Connection refused",
                None,
                Exception("psycopg error with password mypassword"),
            )

        from privateprocureai.database.models import Base

        mp.setattr(Base.metadata, "create_all", failing_create_all)

        with pytest.raises(DatabaseUnavailableError) as exc_info:
            initialize_database(mock_engine)

        # Verify password is not in the error message
        assert "mypassword" not in str(exc_info.value)
        assert "Database initialization failed" in str(exc_info.value)


def test_signing_unavailable_boundary(tmp_path: Path) -> None:
    """When signing key files are missing or mismatched, SigningError is raised."""
    missing_priv = tmp_path / "missing-priv.pem"
    missing_pub = tmp_path / "missing-pub.pem"

    with pytest.raises(SigningError) as exc_info:
        Ed25519ResultSigner.from_pem_files(
            private_key_path=missing_priv,
            public_key_path=missing_pub,
        )

    assert "Could not load signing key files" in str(exc_info.value)


def test_signing_unready_recommendation_boundary(tmp_path: Path) -> None:
    """Signing a BLOCKED recommendation raises SigningError."""
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_ed25519_keypair(private_key_path=priv, public_key_path=pub, passphrase=None)

    signer = Ed25519ResultSigner.from_pem_files(
        private_key_path=priv,
        public_key_path=pub,
    )

    blocked_rec = FinalRecommendation(
        status=FinalRecommendationStatus.BLOCKED,
        summary="Blocked recommendation",
        selected_recommendations=[],
        requires_replanning=True,
    )

    with pytest.raises(SigningError) as exc_info:
        signer.sign(blocked_rec)

    assert "Only a ready final recommendation may receive a signature" in str(exc_info.value)
