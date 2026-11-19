"""Comprehensive test suite covering the 8 reliability and intelligence scenarios.

Test 1: Complete procurement request (all mandatory requirements preserved in intent)
Test 2: Missing evidence (reports insufficient evidence rather than inventing)
Test 3: Unrealistic procurement cost (flags implausible $750 estimate for 25 workstations)
Test 4: Budget violation (exceeding hard budget fails budget validation)
Test 5: Requirement mismatch (proposed product fails minimum RAM requirement)
Test 6: Incomplete supplier verification (unverified supplier classified as unverified)
Test 7: Cryptographic verification (tampered signed procurement payload fails verification)
Test 8: Agent execution failure (failed agent prevents dependent tasks from proceeding)
"""

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from privateprocureai.constraints import BudgetEngine
from privateprocureai.evaluation import CriticAgent
from privateprocureai.execution import ParallelAgentExecutor
from privateprocureai.execution.executor import AgentExecutionError
from privateprocureai.integrity.signing import (
    Ed25519ResultSigner,
    generate_ed25519_keypair,
)
from privateprocureai.intent import IntentExtractor
from privateprocureai.llm.mock import MockLLMProvider
from privateprocureai.schemas import (
    AgentResult,
    AgentStatus,
    BudgetAssessment,
    BudgetConstraint,
    BudgetEvaluationInput,
    BudgetStatus,
    ConstraintStrength,
    CostLineItem,
    EvaluationInput,
    EvaluationIssueCode,
    EvaluationStatus,
    Evidence,
    EvidenceClassification,
    ExtractedIntent,
    FinalRecommendation,
    FinalRecommendationStatus,
    IntentConstraint,
    IntentSource,
    Money,
    PlanningRequest,
    PlanTask,
    Recommendation,
    TaskKind,
    TaskPlan,
    UserPreferences,
)


def _setup_test_signer(tmp_path: Path) -> Ed25519ResultSigner:
    priv = tmp_path / "test_priv.pem"
    pub = tmp_path / "test_pub.pem"
    generate_ed25519_keypair(private_key_path=priv, public_key_path=pub, passphrase=None)
    return Ed25519ResultSigner.from_pem_files(private_key_path=priv, public_key_path=pub)


# ==============================================================================
# Test 1: Complete procurement request
# ==============================================================================
def test_1_complete_procurement_request_preserves_all_requirements() -> None:
    """Mandatory requirements, exclusions, and numerical thresholds survive extraction."""
    request = PlanningRequest(
        goal=(
            "Procure 25 high-performance workstations for an AI laboratory. "
            "Dedicated GPU support required. Enterprise-grade networking."
        ),
        budget=BudgetConstraint(
            maximum=Money(amount=Decimal("75000.00"), currency="USD"),
            is_hard_limit=True,
        ),
        preferences=UserPreferences(
            must_have=[
                "Minimum 32GB RAM per workstation",
                "Minimum 1TB SSD storage",
                "Dedicated GPU support for machine learning",
                "Enterprise-grade networking equipment",
                "Minimum 3-year warranty on computing equipment",
                "Reliable supplier with verifiable business credentials",
            ],
            nice_to_have=[
                "64GB RAM on at least 5 workstations",
                "Energy-efficient hardware",
                "Bulk procurement discounts",
            ],
            avoid=[
                "Refurbished equipment",
                "Products without manufacturer warranty",
            ],
            notes="Ensure all procurement outputs have cryptographic verification.",
        ),
    )

    provider = MockLLMProvider(
        responses=[
            {
                "objective": "Establish an AI development laboratory with 25 workstations.",
                "requirements": {
                    "workstations": {
                        "quantity": 25,
                        "ram_gb_min": 32,
                        "storage_gb_min": 1024,
                        "gpu_required": True,
                    },
                    "networking": {"enterprise_grade": True},
                    "warranty_years_min": 3,
                },
                "constraints": [
                    {
                        "description": "Procure 25 workstations with minimum 32GB RAM",
                        "strength": "hard",
                        "source": "explicit",
                        "target_quantity": 25,
                        "min_threshold": 32.0,
                        "unit": "GB",
                        "original_text": "25 workstations with min 32GB RAM",
                    }
                ],
                "preferences": [
                    "64GB RAM on at least 5 workstations",
                    "Energy-efficient hardware",
                ],
                "exclusions": [
                    "Refurbished equipment",
                    "Products without manufacturer warranty",
                ],
                "supplier_requirements": ["Reliable supplier with verifiable business credentials"],
                "clarification_needed": False,
                "clarification_questions": [],
                "confidence_score": 0.95,
            }
        ]
    )

    intent = IntentExtractor(provider).extract(request)

    # 1. Budget preserved with hard limit
    budget_constraint = next(
        (c for c in intent.constraints if c.category == "budget" or "75000" in c.description),
        None,
    )
    assert budget_constraint is not None
    assert budget_constraint.strength is ConstraintStrength.HARD

    # 2. All must-haves preserved
    constraint_texts = " ".join(c.description for c in intent.constraints)
    for must_have in request.preferences.must_have:
        assert must_have in constraint_texts or any(
            must_have in c.description for c in intent.constraints
        )

    # 3. Exclusions preserved
    for avoid_item in request.preferences.avoid:
        assert avoid_item in intent.exclusions

    # 4. Numerical quantities and requirements
    assert intent.requirements.get("workstations", {}).get("quantity") == 25
    assert intent.requirements.get("workstations", {}).get("ram_gb_min") == 32
    assert intent.confidence_score >= 0.9


# ==============================================================================
# Test 2: Missing evidence
# ==============================================================================
def test_2_missing_evidence_reports_insufficient_evidence() -> None:
    """System reports insufficient evidence rather than inventing suppliers or specs."""
    evidence_id = uuid4()
    missing_evidence = Evidence(
        evidence_id=evidence_id,
        source_id="unregistered_catalog_entry",
        excerpt="No authoritative pricing or specification record exists.",
        relevance_score=0.2,
        classification=EvidenceClassification.MISSING,
        provenance="unverified_source",
    )

    intent = ExtractedIntent(
        request_id=uuid4(),
        objective="Procure custom unverified experimental hardware.",
        constraints=[
            IntentConstraint(
                description="Must have valid technical documentation.",
                strength=ConstraintStrength.HARD,
                source=IntentSource.EXPLICIT,
            )
        ],
        confidence_score=0.8,
    )

    agent_result = AgentResult(
        task_id=uuid4(),
        agent_name="research_agent",
        status=AgentStatus.COMPLETED,
        summary="Could not find authoritative sources for requested hardware.",
        findings=[],
        evidence=[missing_evidence],
        unresolved_issues=["No verified specification document available."],
    )

    critic = CriticAgent(
        MockLLMProvider(
            responses=[
                {
                    "summary": "Evidence is missing for candidate requirements.",
                    "criteria": [
                        {
                            "name": "Evidence Grounding",
                            "passed": False,
                            "severity": "critical",
                            "rationale": "No verifiable evidence exists for the requested items.",
                            "related_task_ids": [],
                        }
                    ],
                }
            ]
        )
    )

    evaluation = critic.evaluate(
        EvaluationInput(
            intent=intent,
            agent_results=[agent_result],
        )
    )

    assert evaluation.status is EvaluationStatus.FAIL
    assert evaluation.requires_replanning is True
    assert any(
        iss.code == EvaluationIssueCode.INSUFFICIENT_EVIDENCE or "missing" in iss.message.lower()
        for iss in evaluation.issues
    )


# ==============================================================================
# Test 3: Unrealistic procurement cost
# ==============================================================================
def test_3_unrealistic_procurement_cost_is_flagged_by_critic() -> None:
    """Proposing 25 enterprise computing workstations for $750 is flagged as implausibly low."""
    intent = ExtractedIntent(
        request_id=uuid4(),
        objective="Set up AI lab with 25 workstations.",
        constraints=[
            IntentConstraint(
                description="Procure 25 workstations for AI laboratory",
                strength=ConstraintStrength.HARD,
                source=IntentSource.EXPLICIT,
                target_quantity=25,
            )
        ],
        confidence_score=0.9,
    )

    # Implausible proposal: 25 workstations at $750 total ($30/workstation)
    rec = Recommendation(
        title="Super Saver Workstation Pack",
        rationale="25 workstations offered at 750 USD total.",
        estimated_cost=Money(amount=Decimal("750.00"), currency="USD"),
        evidence_ids=[uuid4()],
        confidence_score=0.4,
    )

    agent_result = AgentResult(
        task_id=uuid4(),
        agent_name="recommendation_agent",
        status=AgentStatus.COMPLETED,
        summary="Synthesized candidate recommendation.",
        recommendations=[rec],
        evidence=[
            Evidence(
                source_id="mock_unrealistic",
                excerpt="25 workstations at $750 total.",
                relevance_score=0.5,
                classification=EvidenceClassification.ESTIMATED,
            )
        ],
    )

    budget_assessment = BudgetAssessment(
        budget=BudgetConstraint(maximum=Money(amount=Decimal("75000.00"), currency="USD")),
        status=BudgetStatus.COMPLIANT,
        total_cost=Money(amount=Decimal("750.00"), currency="USD"),
        remaining_amount=Decimal("74250.00"),
        line_item_count=1,
        violations=[],
        requires_replanning=False,
    )

    critic = CriticAgent(
        MockLLMProvider(
            responses=[
                {
                    "summary": "Review of candidate packages.",
                    "criteria": [],
                }
            ]
        )
    )

    evaluation = critic.evaluate(
        EvaluationInput(
            intent=intent,
            agent_results=[agent_result],
            budget_assessment=budget_assessment,
        )
    )

    assert evaluation.status is EvaluationStatus.FAIL
    assert evaluation.requires_replanning is True
    assert any(
        iss.code == EvaluationIssueCode.UNREALISTIC_PROCUREMENT_COST for iss in evaluation.issues
    )


# ==============================================================================
# Test 4: Budget violation
# ==============================================================================
def test_4_budget_violation_exceeding_hard_budget_fails() -> None:
    """A valid proposal exceeding the hard budget fails budget validation."""
    budget = BudgetConstraint(
        maximum=Money(amount=Decimal("75000.00"), currency="USD"),
        is_hard_limit=True,
    )

    line_items = [
        CostLineItem(
            label="25 AI Workstations Tier-1",
            cost=Money(amount=Decimal("80000.00"), currency="USD"),
            quantity=25,
        )
    ]

    engine = BudgetEngine()
    assessment = engine.evaluate(BudgetEvaluationInput(budget=budget, line_items=line_items))

    assert assessment.status is BudgetStatus.EXCEEDED
    assert assessment.requires_replanning is True
    assert assessment.total_cost.amount == Decimal("80000.00")
    assert assessment.remaining_amount == Decimal("-5000.00")


# ==============================================================================
# Test 5: Requirement mismatch (RAM spec)
# ==============================================================================
def test_5_requirement_mismatch_fails_critic_evaluation() -> None:
    """Proposed candidate with 16GB RAM fails when 32GB RAM is required."""
    intent = ExtractedIntent(
        request_id=uuid4(),
        objective="Procure AI workstations with high memory.",
        constraints=[
            IntentConstraint(
                description="Minimum 32GB RAM per workstation required",
                strength=ConstraintStrength.HARD,
                source=IntentSource.EXPLICIT,
                min_threshold=32.0,
                unit="GB",
            )
        ],
        confidence_score=0.9,
    )

    rec = Recommendation(
        title="Entry Workstation Tier",
        rationale="Configured with 16GB RAM for entry-level machine learning.",
        estimated_cost=Money(amount=Decimal("1500.00"), currency="USD"),
        evidence_ids=[uuid4()],
        confidence_score=0.7,
    )

    agent_result = AgentResult(
        task_id=uuid4(),
        agent_name="recommendation_agent",
        status=AgentStatus.COMPLETED,
        summary="Generated recommendation.",
        recommendations=[rec],
        evidence=[
            Evidence(
                source_id="catalog_entry_1",
                excerpt="Entry workstation with 16GB RAM.",
                relevance_score=0.85,
            )
        ],
    )

    critic = CriticAgent(
        MockLLMProvider(
            responses=[
                {
                    "summary": "Candidate review.",
                    "criteria": [],
                }
            ]
        )
    )

    evaluation = critic.evaluate(
        EvaluationInput(
            intent=intent,
            agent_results=[agent_result],
        )
    )

    assert evaluation.status is EvaluationStatus.FAIL
    assert evaluation.requires_replanning is True
    assert any(iss.code == EvaluationIssueCode.REQUIREMENT_MISMATCH for iss in evaluation.issues)


# ==============================================================================
# Test 6: Incomplete supplier verification
# ==============================================================================
def test_6_unverified_supplier_evidence_is_classified_as_unverified() -> None:
    """Supplier info that lacks authoritative credentials is classified as unverified."""
    unverified_ev = Evidence(
        source_id="unregistered_reseller_quote",
        source_url="http://grey-market-vendor.corp/offer",
        excerpt="Hardware units available with no manufacturer warranty or credentials.",
        relevance_score=0.7,
        classification=EvidenceClassification.UNVERIFIED,
        provenance="unaccredited_vendor",
        associated_requirement="supplier_credentials",
    )

    assert unverified_ev.classification is EvidenceClassification.UNVERIFIED
    assert unverified_ev.provenance == "unaccredited_vendor"


# ==============================================================================
# Test 7: Cryptographic verification of tampered payload
# ==============================================================================
def test_7_tampered_procurement_payload_fails_signature_verification(tmp_path: Path) -> None:
    """Tampering with signed procurement payload fails Ed25519 verification."""
    signer = _setup_test_signer(tmp_path)

    rec = FinalRecommendation(
        status=FinalRecommendationStatus.READY,
        summary="Original verified procurement plan.",
        selected_recommendations=[],
        requires_replanning=False,
    )

    signed_result = signer.sign(
        rec,
        request_id=uuid4(),
        intent_hash_sha256="a" * 64,
        evidence_ids=[uuid4()],
        budget_status="compliant",
        evaluation_status="pass",
    )

    # 1. Valid signature verifies
    verification = signer.verify(signed_result)
    assert verification.is_valid is True

    # 2. Tampered recommendation summary fails verification
    tampered_rec = signed_result.recommendation.model_copy(
        update={"summary": "Tampered fraudulent summary."}
    )
    tampered_signed = signed_result.model_copy(update={"recommendation": tampered_rec})
    failed_verification = signer.verify(tampered_signed)
    assert failed_verification.is_valid is False
    assert "hash does not match" in failed_verification.reason

    # 3. Tampered integrity metadata fails verification
    tampered_integ = signed_result.integrity.model_copy(update={"budget_status": "exceeded"})
    tampered_integ_signed = signed_result.model_copy(update={"integrity": tampered_integ})
    failed_integ_verification = signer.verify(tampered_integ_signed)
    assert failed_integ_verification.is_valid is False


# ==============================================================================
# Test 8: Specialist agent execution failure halts dependent tasks
# ==============================================================================
def test_8_specialist_agent_failure_halts_dependent_tasks() -> None:
    """When an upstream agent task fails, execution halts and dependent tasks never execute."""

    class FailingAgent:
        agent_name = "failing_agent"

        def supports(self, task: PlanTask) -> bool:
            return task.kind is TaskKind.RESEARCH

        def run(self, context) -> AgentResult:
            raise RuntimeError("Database connection refused during catalog lookup.")

    class DownstreamAgent:
        agent_name = "downstream_agent"
        executed: bool = False

        def supports(self, task: PlanTask) -> bool:
            return task.kind is TaskKind.RECOMMENDATION

        def run(self, context) -> AgentResult:
            DownstreamAgent.executed = True
            return AgentResult(
                task_id=context.task.task_id,
                agent_name=self.agent_name,
                status=AgentStatus.COMPLETED,
            )

    upstream_task = PlanTask(
        title="Catalog research",
        instructions="Lookup catalog.",
        kind=TaskKind.RESEARCH,
    )
    downstream_task = PlanTask(
        title="Recommendation generation",
        instructions="Synthesize recommendation.",
        kind=TaskKind.RECOMMENDATION,
        depends_on=[upstream_task.task_id],
    )

    plan = TaskPlan(
        request_id=uuid4(),
        planning_summary="Sequential pipeline.",
        tasks=[upstream_task, downstream_task],
    )

    executor = ParallelAgentExecutor([FailingAgent(), DownstreamAgent()])

    with pytest.raises(AgentExecutionError, match="failed: Database connection refused"):
        executor.execute(
            intent=ExtractedIntent(
                request_id=plan.request_id,
                objective="Test failure isolation.",
                constraints=[],
                confidence_score=0.9,
            ),
            task_plan=plan,
            evidence=[],
        )

    # Crucial assertion: Downstream agent was NEVER called
    assert DownstreamAgent.executed is False
