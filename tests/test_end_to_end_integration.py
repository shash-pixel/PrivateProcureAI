"""End-to-end integration tests connecting intent, planning, RAG, and signing."""

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from privateprocureai.api import create_api_app
from privateprocureai.database.models import Base
from privateprocureai.database.repositories import PostgresSignedRecommendationRepository
from privateprocureai.integrity.signing import (
    Ed25519ResultSigner,
    generate_ed25519_keypair,
)
from privateprocureai.llm.mock import MockLLMProvider
from privateprocureai.orchestration import ApplicationWorkflow
from privateprocureai.schemas import (
    ApplicationWorkflowResult,
    ApplicationWorkflowStatus,
    BudgetConstraint,
    BudgetStatus,
    Evidence,
    FinalRecommendationStatus,
    Money,
    PlanningRequest,
    RetrievalResult,
)


class MockRetriever:
    """Mock retriever that supplies deterministic evidence chunks without remote calls."""

    def __init__(self, evidence_list: list[Evidence]) -> None:
        self._evidence_list = evidence_list

    def retrieve(self, query: str) -> RetrievalResult:
        del query
        return RetrievalResult(
            query="test query",
            evidence=self._evidence_list,
            discarded_match_count=0,
        )


def _setup_mock_signer(tmp_path: Path) -> Ed25519ResultSigner:
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    generate_ed25519_keypair(private_key_path=priv, public_key_path=pub, passphrase=None)
    return Ed25519ResultSigner.from_pem_files(private_key_path=priv, public_key_path=pub)


def test_end_to_end_workflow_happy_path(tmp_path: Path) -> None:
    """A generic goal passes through all components to produce a signed result."""
    evidence_id = uuid4()
    rec_id = uuid4()

    evidence_item = Evidence(
        evidence_id=evidence_id,
        source_id="doc_chunk_1",
        excerpt="Enterprise package tier provides comprehensive coverage at 500 USD.",
        relevance_score=0.92,
    )

    llm_responses = [
        # 1. Intent Extraction
        {
            "objective": "Procure a comprehensive operational equipment package.",
            "constraints": [
                {
                    "description": "Must not exceed 1000 USD total cost.",
                    "strength": "hard",
                    "source": "explicit",
                }
            ],
            "preferences": ["Prefer enterprise support tier."],
            "assumptions": [],
            "clarification_needed": False,
            "clarification_questions": [],
            "confidence_score": 0.95,
        },
        # 2. Task Planning
        {
            "planning_summary": "Research available options and generate candidate recommendation.",
            "tasks": [
                {
                    "task_key": "research_task",
                    "title": "Research Options",
                    "instructions": "Gather available package tiers and pricing.",
                    "kind": "research",
                    "priority": "high",
                    "depends_on": [],
                    "requires_retrieval": True,
                },
                {
                    "task_key": "recommendation_task",
                    "title": "Recommend Solution",
                    "instructions": "Select candidate package satisfying constraints.",
                    "kind": "recommendation",
                    "priority": "normal",
                    "depends_on": ["research_task"],
                    "requires_retrieval": False,
                },
            ],
        },
        # 3. Research Agent Output
        {
            "summary": "Identified available enterprise package option.",
            "findings": [
                {
                    "statement": "Enterprise package meets stated requirements.",
                    "evidence_ids": [str(evidence_id)],
                    "confidence_score": 0.95,
                }
            ],
            "evidence_gaps": [],
        },
        # 4. Recommendation Agent Output
        {
            "summary": "Recommended enterprise tier package.",
            "recommendations": [
                {
                    "recommendation_id": str(rec_id),
                    "title": "Enterprise Solution Package",
                    "rationale": "Directly supported by verified evidence.",
                    "estimated_cost": {"amount": "500.00", "currency": "USD"},
                    "evidence_ids": [str(evidence_id)],
                    "confidence_score": 0.9,
                }
            ],
            "evidence_gaps": [],
        },
        # 5. Critic Agent Evaluation
        {
            "summary": "Plan and execution satisfy constraints and evidence requirements.",
            "criteria": [
                {
                    "name": "evidence_grounding",
                    "passed": True,
                    "severity": "info",
                    "rationale": "Findings and recommendations cite valid evidence.",
                    "related_task_ids": [],
                }
            ],
        },
        # 6. Final Recommendation Optimizer Decision
        {
            "summary": "Selected Enterprise Solution Package within approved budget.",
            "selected_recommendation_ids": [str(rec_id)],
        },
    ]

    mock_llm = MockLLMProvider(responses=llm_responses)
    mock_retriever = MockRetriever([evidence_item])
    signer = _setup_mock_signer(tmp_path)

    # In-memory database for repository
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    repository = PostgresSignedRecommendationRepository(session_factory)

    workflow = ApplicationWorkflow(
        llm=mock_llm,
        retriever=mock_retriever,
        signer=signer,
        repository=repository,
    )

    request = PlanningRequest(
        goal="Procure operational package satisfying constraints.",
        budget=BudgetConstraint(
            maximum=Money(amount=Decimal("1000.00"), currency="USD"),
            is_hard_limit=True,
        ),
    )

    result = workflow.run(request)

    # Acceptance criteria assertions
    assert result.status is ApplicationWorkflowStatus.COMPLETED
    assert result.intent.objective == "Procure a comprehensive operational equipment package."
    assert result.task_plan is not None
    assert len(result.task_plan.tasks) == 2
    assert len(result.agent_results) == 2
    assert len(result.evidence) == 1
    assert result.budget_assessment is not None
    assert result.budget_assessment.status == BudgetStatus.COMPLIANT
    assert result.evaluation_report is not None
    assert result.evaluation_report.status.value == "pass"

    # Final recommendation
    assert result.final_recommendation is not None
    assert result.final_recommendation.status is FinalRecommendationStatus.READY
    assert len(result.final_recommendation.selected_recommendations) == 1
    assert result.final_recommendation.selected_recommendations[0].recommendation_id == rec_id

    # Cryptographic signature
    assert result.signed_recommendation is not None
    assert result.signed_recommendation.integrity.algorithm.value == "ed25519"
    assert len(result.signed_recommendation.integrity.payload_sha256) == 64
    assert result.verification is not None
    assert result.verification.is_valid is True

    # Database persistence
    assert result.persisted_result_id is not None
    loaded_record = repository.get(result.persisted_result_id)
    assert loaded_record is not None
    loaded_hash = loaded_record.integrity.payload_sha256
    assert loaded_hash == result.signed_recommendation.integrity.payload_sha256

    # Observable events with durations
    assert len(result.events) >= 5
    step_names = [e.step.value for e in result.events]
    assert "intent_extraction" in step_names
    assert "task_planning" in step_names
    assert "agent_execution" in step_names
    assert "budget_validation" in step_names
    assert "evaluation" in step_names
    assert "optimization" in step_names
    assert "signing" in step_names
    assert "persistence" in step_names


def test_end_to_end_workflow_clarification_exit() -> None:
    """A vague goal stops early after intent extraction, producing no tasks or signatures."""
    llm_responses = [
        {
            "objective": "Plan an unspecified activity.",
            "constraints": [],
            "preferences": [],
            "assumptions": [],
            "clarification_needed": True,
            "clarification_questions": ["What is the desired deliverable and budget?"],
            "confidence_score": 0.3,
        }
    ]

    workflow = ApplicationWorkflow(llm=MockLLMProvider(responses=llm_responses))
    request = PlanningRequest(goal="Help me buy something.")
    result = workflow.run(request)

    assert result.status is ApplicationWorkflowStatus.NEEDS_CLARIFICATION
    assert result.task_plan is None
    assert result.final_recommendation is None
    assert result.signed_recommendation is None
    assert len(result.intent.clarification_questions) == 1


def test_fastapi_workflow_endpoint() -> None:
    """FastAPI exposes the full end-to-end workflow at /v1/workflow/run."""

    class StubWorkflowRunner:
        def run(self, request: PlanningRequest) -> ApplicationWorkflowResult:
            from privateprocureai.schemas import ExtractedIntent

            return ApplicationWorkflowResult(
                request_id=request.request_id,
                status=ApplicationWorkflowStatus.COMPLETED,
                intent=ExtractedIntent(
                    request_id=request.request_id,
                    objective="Complete procurement successfully.",
                    constraints=[],
                    preferences=[],
                    assumptions=[],
                    clarification_needed=False,
                    clarification_questions=[],
                    confidence_score=0.9,
                ),
            )

    class StubPlanningRunner:
        def run(self, planning_request):
            raise NotImplementedError

    client = TestClient(
        create_api_app(
            planning_runner=StubPlanningRunner(),
            application_runner=StubWorkflowRunner(),
        )
    )

    req_id = uuid4()
    response = client.post(
        "/v1/workflow/run",
        json={
            "request_id": str(req_id),
            "goal": "Procure test equipment.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == str(req_id)
    assert data["status"] == "completed"
    assert data["intent"]["objective"] == "Complete procurement successfully."


def test_end_to_end_workflow_hard_budget_violation(tmp_path: Path) -> None:
    """When candidates exceed a hard budget, the workflow returns BLOCKED without signing."""
    rec_id = uuid4()
    evidence_id = uuid4()

    llm_responses = [
        # Intent
        {
            "objective": "Procure luxury supplies.",
            "constraints": [
                {
                    "description": "Hard budget limit of 100 USD.",
                    "strength": "hard",
                    "source": "explicit",
                }
            ],
            "preferences": [],
            "assumptions": [],
            "clarification_needed": False,
            "clarification_questions": [],
            "confidence_score": 0.9,
        },
        # Plan
        {
            "planning_summary": "Identify candidates.",
            "tasks": [
                {
                    "task_key": "rec_task",
                    "title": "Recommend Solution",
                    "instructions": "Propose solution.",
                    "kind": "recommendation",
                    "priority": "normal",
                    "depends_on": [],
                    "requires_retrieval": True,
                }
            ],
        },
        # Recommendation Agent Output (cost 500 exceeds 100 limit)
        {
            "summary": "Proposed high-end option.",
            "recommendations": [
                {
                    "recommendation_id": str(rec_id),
                    "title": "High-End Package",
                    "rationale": "High quality option.",
                    "estimated_cost": {"amount": "500.00", "currency": "USD"},
                    "evidence_ids": [str(evidence_id)],
                    "confidence_score": 0.8,
                }
            ],
            "evidence_gaps": [],
        },
        # Critic Agent (will fail due to hard budget violation in deterministic check)
        {
            "summary": "Plan exceeds budget.",
            "criteria": [
                {
                    "name": "budget_compliance",
                    "passed": False,
                    "severity": "critical",
                    "rationale": "Exceeds budget limit.",
                    "related_task_ids": [],
                }
            ],
        },
        # Re-planning coordinator attempt 1: plan
        {
            "planning_summary": "Revised plan within budget.",
            "tasks": [
                {
                    "task_key": "rec_task_2",
                    "title": "Recommend Solution",
                    "instructions": "Propose solution.",
                    "kind": "recommendation",
                    "priority": "normal",
                    "depends_on": [],
                    "requires_retrieval": True,
                }
            ],
        },
        # Re-planning coordinator attempt 1: agent
        {
            "summary": "Still high-end option.",
            "recommendations": [
                {
                    "recommendation_id": str(rec_id),
                    "title": "High-End Package",
                    "rationale": "High quality option.",
                    "estimated_cost": {"amount": "500.00", "currency": "USD"},
                    "evidence_ids": [str(evidence_id)],
                    "confidence_score": 0.8,
                }
            ],
            "evidence_gaps": [],
        },
        # Re-planning coordinator attempt 1: critic
        {
            "summary": "Still exceeds budget.",
            "criteria": [
                {
                    "name": "budget_compliance",
                    "passed": False,
                    "severity": "critical",
                    "rationale": "Still exceeds limit.",
                    "related_task_ids": [],
                }
            ],
        },
    ]

    mock_llm = MockLLMProvider(responses=llm_responses)
    signer = _setup_mock_signer(tmp_path)
    evidence_item = Evidence(
        evidence_id=evidence_id,
        source_id="doc1",
        excerpt="Test excerpt",
        relevance_score=0.9,
    )

    workflow = ApplicationWorkflow(
        llm=mock_llm,
        retriever=MockRetriever([evidence_item]),
        signer=signer,
        max_replanning_iterations=1,
    )

    request = PlanningRequest(
        goal="Procure supplies under budget.",
        budget=BudgetConstraint(
            maximum=Money(amount=Decimal("100.00"), currency="USD"),
            is_hard_limit=True,
        ),
    )

    result = workflow.run(request)

    assert result.status is ApplicationWorkflowStatus.REPLANNING_EXHAUSTED
    assert result.signed_recommendation is None
    assert result.verification is None
    assert len(result.replanning_iterations) == 1
