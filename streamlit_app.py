"""Interactive multi-agent planning and procurement interface for PrivateProcureAI."""

from __future__ import annotations

import logging
from uuid import UUID

import streamlit as st
from pydantic import ValidationError

from privateprocureai.config import get_settings
from privateprocureai.database.repositories import PostgresSignedRecommendationRepository
from privateprocureai.database.session import create_database_engine, create_session_factory
from privateprocureai.integrity.signing import Ed25519ResultSigner
from privateprocureai.llm import create_llm_provider
from privateprocureai.llm.base import ResponseModelT, StructuredGenerationRequest, StructuredLLM
from privateprocureai.observability import configure_logging
from privateprocureai.orchestration import ApplicationWorkflow, PlanningWorkflow
from privateprocureai.rag.embeddings import GeminiEmbeddingProvider
from privateprocureai.rag.retrieval import RagRetriever
from privateprocureai.rag.vector_store import ChromaVectorStore
from privateprocureai.schemas import (
    ApplicationWorkflowResult,
    ApplicationWorkflowStatus,
    Evidence,
    EvidenceClassification,
    FinalRecommendationStatus,
    RetrievalResult,
)
from privateprocureai.ui.request_builder import build_planning_request

logger = logging.getLogger(__name__)

DEMO_REC_ID = UUID("aaaaaaa1-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

DEMO_EVIDENCES: list[Evidence] = [
    Evidence(
        evidence_id=UUID("11111111-1111-4111-8111-111111111111"),
        source_id="catalog_workstation_ai_precision",
        source_url="https://vendor-portal.procure.internal/catalog/precision-ai-tower",
        document_id="DOC-HW-2026-081",
        excerpt=(
            "Precision AI Developer Workstation: 32GB DDR5 ECC RAM, 1TB NVMe PCIe 4.0 SSD, "
            "NVIDIA RTX A4000 GPU (16GB VRAM), dual 10GbE NICs. Tested and certified for PyTorch "
            "and ML workloads. Unit price: $2,450.00 USD."
        ),
        relevance_score=0.96,
        classification=EvidenceClassification.VERIFIED,
        provenance="verified_supplier_catalog",
        associated_requirement="workstations.hardware_spec",
        supporting_claim="32GB RAM, 1TB SSD, Dedicated GPU @ $2,450/unit",
    ),
    Evidence(
        evidence_id=UUID("22222222-2222-4222-8222-222222222222"),
        source_id="quote_bulk_workstations_25u",
        source_url="https://vendor-portal.procure.internal/quotes/quote-78411",
        document_id="QUOTE-2026-78411",
        excerpt=(
            "Distributor Bulk Quotation #78411: 25x AI Developer Workstations @ $2,450.00/unit = "
            "$61,250.00 USD total. Includes volume pricing discount and insured delivery."
        ),
        relevance_score=0.94,
        classification=EvidenceClassification.VERIFIED,
        provenance="distributor_formal_quote",
        associated_requirement="workstations.pricing",
        supporting_claim="25 units @ $2,450 = $61,250 USD total",
    ),
    Evidence(
        evidence_id=UUID("33333333-3333-4333-8333-333333333333"),
        source_id="catalog_networking_managed_switch",
        source_url="https://vendor-portal.procure.internal/catalog/net-switch-48p-poe",
        document_id="DOC-NET-2026-014",
        excerpt=(
            "Enterprise Managed Switch: 48x 1GbE PoE+ ports, 4x 10GbE SFP+ uplinks, Layer 3 "
            "routing, redundant hot-swap power supplies. Full enterprise warranty. "
            "Catalog price: $5,200.00 USD."
        ),
        relevance_score=0.92,
        classification=EvidenceClassification.VERIFIED,
        provenance="verified_supplier_catalog",
        associated_requirement="networking.enterprise_grade",
        supporting_claim="Enterprise 48-port PoE+ Managed Switch @ $5,200 USD",
    ),
    Evidence(
        evidence_id=UUID("44444444-4444-4444-8444-444444444444"),
        source_id="agreement_warranty_3yr_prosupport",
        source_url="https://vendor-portal.procure.internal/agreements/3yr-prosupport",
        document_id="SLA-WR-2026-003",
        excerpt=(
            "3-Year OEM ProSupport On-Site Agreement: Next-business-day on-site hardware "
            "replacement, 24/7 technical support, defect coverage for 25 workstations and network "
            "infrastructure. Comprehensive service fee: $3,500.00 USD."
        ),
        relevance_score=0.93,
        classification=EvidenceClassification.VERIFIED,
        provenance="oem_warranty_agreement",
        associated_requirement="warranty.terms",
        supporting_claim="3-Year On-Site Manufacturer Warranty @ $3,500 USD",
    ),
    Evidence(
        evidence_id=UUID("55555555-5555-4555-8555-555555555555"),
        source_id="accreditation_compliance_register",
        source_url="https://procure-compliance.corp/accreditations/distributor-certified",
        document_id="COMPLIANCE-ACCRED-9001",
        excerpt=(
            "Supplier Compliance Record: Tier-1 Authorized Systems Integrator, ISO 9001:2015 "
            "quality certified, verified corporate credentials. Factory-new equipment with "
            "direct OEM warranty; zero refurbished inventory permitted."
        ),
        relevance_score=0.95,
        classification=EvidenceClassification.VERIFIED,
        provenance="procurement_compliance_register",
        associated_requirement="supplier.credentials",
        supporting_claim="ISO 9001 accredited Tier-1 partner, factory-new hardware only",
    ),
]


class DemoMockLLM(StructuredLLM):
    """Provides dynamic structured responses for interactive UI demonstration in mock mode."""

    def generate_structured(
        self,
        request: StructuredGenerationRequest,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        model_name = response_model.__name__

        if model_name in ("IntentDetails", "ExtractedIntent"):
            payload = {
                "objective": (
                    "Establish an AI development laboratory with 25 high-performance workstations."
                ),
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
                        "description": "Procure 25 workstations with minimum 32GB RAM per unit",
                        "strength": "hard",
                        "source": "explicit",
                        "category": "workstations",
                        "target_quantity": 25,
                        "min_threshold": 32.0,
                        "unit": "GB",
                        "original_text": "25 workstations with min 32GB RAM",
                    },
                    {
                        "description": "Minimum 1TB SSD storage per workstation",
                        "strength": "hard",
                        "source": "explicit",
                        "category": "storage",
                        "min_threshold": 1024.0,
                        "unit": "GB",
                    },
                    {
                        "description": "Dedicated GPU support for machine learning workloads",
                        "strength": "hard",
                        "source": "explicit",
                        "category": "gpu",
                    },
                    {
                        "description": "Enterprise-grade networking equipment",
                        "strength": "hard",
                        "source": "explicit",
                        "category": "networking",
                    },
                    {
                        "description": "Minimum 3-year manufacturer warranty",
                        "strength": "hard",
                        "source": "explicit",
                        "category": "warranty",
                        "min_threshold": 3.0,
                        "unit": "years",
                    },
                ],
                "preferences": [
                    "64GB RAM on at least 5 workstations",
                    "Energy-efficient hardware",
                    "Bulk procurement discounts",
                ],
                "exclusions": [
                    "Refurbished equipment",
                    "Products without manufacturer warranty",
                    "Proprietary hardware with limited upgradeability",
                ],
                "supplier_requirements": ["Reliable supplier with verifiable business credentials"],
                "assumptions": ["Standard commercial terms and verified corporate invoicing apply"],
                "clarification_needed": False,
                "clarification_questions": [],
                "confidence_score": 0.96,
                "confidence_breakdown": {
                    "intent_extraction": 0.98,
                    "requirement_coverage": 1.0,
                },
            }
            return response_model.model_validate(payload)

        if model_name == "TaskPlanDraft":
            payload = {
                "planning_summary": (
                    "Decomposed procurement into hardware research, supplier qualification, "
                    "volume pricing analysis, and strategy synthesis."
                ),
                "tasks": [
                    {
                        "task_key": "hardware_spec_research",
                        "title": "Hardware Specifications & Architecture Research",
                        "instructions": (
                            "Research workstation models meeting 32GB RAM, 1TB NVMe, GPU, "
                            "and enterprise networking equipment."
                        ),
                        "kind": "research",
                        "priority": "high",
                        "depends_on": [],
                        "requires_retrieval": True,
                    },
                    {
                        "task_key": "supplier_qualification_audit",
                        "title": "Supplier Accreditation & Warranty Verification",
                        "instructions": (
                            "Verify supplier corporate credentials, ISO certifications, and 3-year "
                            "OEM warranty coverage."
                        ),
                        "kind": "supplier_evaluation",
                        "priority": "high",
                        "depends_on": [],
                        "requires_retrieval": True,
                    },
                    {
                        "task_key": "pricing_and_budget_analysis",
                        "title": "Volume Pricing & Budget Feasibility Analysis",
                        "instructions": (
                            "Evaluate 25-workstation volume discounts and verify total procurement "
                            "cost against $75,000 hard budget."
                        ),
                        "kind": "research",
                        "priority": "normal",
                        "depends_on": ["hardware_spec_research", "supplier_qualification_audit"],
                        "requires_retrieval": False,
                    },
                    {
                        "task_key": "procurement_recommendation",
                        "title": "Synthesize Final Procurement Recommendation",
                        "instructions": (
                            "Formulate turnkey procurement package with verifiable evidence."
                        ),
                        "kind": "recommendation",
                        "priority": "high",
                        "depends_on": ["pricing_and_budget_analysis"],
                        "requires_retrieval": False,
                    },
                ],
            }
            return response_model.model_validate(payload)

        if model_name == "ResearchAgentOutput":
            payload = {
                "summary": (
                    "Verified hardware compute specifications, enterprise networking, and ISO 9001 "
                    "supplier credentials with 3-year OEM warranty."
                ),
                "findings": [
                    {
                        "statement": (
                            "Precision AI Workstation (32GB RAM, 1TB NVMe SSD, RTX A4000) "
                            "satisfies all technical compute requirements @ $2,450.00/unit."
                        ),
                        "evidence_ids": [
                            "11111111-1111-4111-8111-111111111111",
                            "22222222-2222-4222-8222-222222222222",
                        ],
                        "confidence_score": 0.96,
                    },
                    {
                        "statement": (
                            "Supplier is Tier-1 ISO 9001:2015 certified with verified credentials, "
                            "providing comprehensive 3-year on-site warranty @ $3,500.00."
                        ),
                        "evidence_ids": [
                            "44444444-4444-4444-8444-444444444444",
                            "55555555-5555-4555-8555-555555555555",
                        ],
                        "confidence_score": 0.95,
                    },
                    {
                        "statement": (
                            "Enterprise Managed 48-port PoE+ Switch satisfies networking "
                            "specifications @ $5,200.00."
                        ),
                        "evidence_ids": ["33333333-3333-4333-8333-333333333333"],
                        "confidence_score": 0.94,
                    },
                ],
                "evidence_gaps": [],
            }
            return response_model.model_validate(payload)

        if model_name == "RecommendationAgentOutput":
            payload = {
                "summary": "Formulated turnkey AI development laboratory procurement strategy.",
                "recommendations": [
                    {
                        "recommendation_id": str(DEMO_REC_ID),
                        "title": "Enterprise AI Development Laboratory Procurement Package",
                        "rationale": (
                            "Turnkey deployment: 25x AI Workstations (32GB RAM, 1TB NVMe, RTX GPU) "
                            "@ $2,450 ($61,250.00) + Enterprise 48-Port PoE+ Switch ($5,200.00) + "
                            "3-Year On-Site ProSupport Warranty ($3,500.00). Sourced from ISO 9001 "
                            "accredited distributor with verified credentials. 100% factory-new."
                        ),
                        "estimated_cost": {"amount": "69950.00", "currency": "USD"},
                        "evidence_ids": [str(ev.evidence_id) for ev in DEMO_EVIDENCES],
                        "confidence_score": 0.95,
                    }
                ],
                "evidence_gaps": [],
            }
            return response_model.model_validate(payload)

        if model_name == "EvaluationAnalysis":
            payload = {
                "summary": (
                    "Plan satisfies all mandatory hardware requirements, evidence grounding, "
                    "supplier credentials, and budget feasibility ($69,950 <= $75,000)."
                ),
                "criteria": [
                    {
                        "name": "Requirement Coverage",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "All mandatory requirements (25 units, 32GB RAM, 1TB SSD, GPU, "
                            "networking, 3-year warranty) individually verified."
                        ),
                    },
                    {
                        "name": "Evidence Grounding",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "Every product spec, price quote, and SLA cites verified sources."
                        ),
                    },
                    {
                        "name": "Supplier Reliability",
                        "passed": True,
                        "severity": "info",
                        "rationale": "Distributor verified via ISO 9001 compliance register.",
                    },
                    {
                        "name": "Budget Feasibility",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "Total cost of $69,950.00 USD is realistic and within $75,000 limit."
                        ),
                    },
                    {
                        "name": "Technical Feasibility",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "Architecture and compute specs satisfy ML operational requirements."
                        ),
                    },
                    {
                        "name": "Recommendation Quality",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "High quality strategy accounting for bulk discounts and warranty."
                        ),
                    },
                    {
                        "name": "Data Integrity",
                        "passed": True,
                        "severity": "info",
                        "rationale": (
                            "End-to-end task decomposition and agent outputs are consistent."
                        ),
                    },
                ],
            }
            return response_model.model_validate(payload)

        if model_name == "OptimizationDecision":
            payload = {
                "summary": (
                    "Confirmed Enterprise AI Development Laboratory Procurement Package "
                    "as the optimal, fully compliant procurement solution."
                ),
                "selected_recommendation_ids": [str(DEMO_REC_ID)],
            }
            return response_model.model_validate(payload)

        raise ValueError(f"Unknown response model in DemoMockLLM: {model_name}")


class DemoRetriever:
    """Provides deterministic multi-source demonstration evidence in mock mode."""

    def retrieve(self, query: str) -> RetrievalResult:
        del query
        return RetrievalResult(
            query="workstation and networking procurement query",
            evidence=DEMO_EVIDENCES,
            discarded_match_count=0,
        )


st.set_page_config(page_title="PrivateProcureAI", layout="wide")

CUSTOM_CSS = """
<style>
/* Root and theme variables */
:root {
    --pp-bg-primary: #0a0e17;
    --pp-bg-surface: #0f172a;
    --pp-bg-card: #111a2e;
    --pp-bg-input: #162036;
    --pp-border-subtle: #1e293b;
    --pp-border-active: #334155;
    --pp-text-primary: #f8fafc;
    --pp-text-secondary: #94a3b8;
    --pp-text-muted: #64748b;
    --pp-accent-red: #e11d48;
    --pp-accent-red-hover: #be123c;
    --pp-accent-red-active: #9f1239;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--pp-bg-primary) !important;
    color: var(--pp-text-primary) !important;
}

header[data-testid="stHeader"] {
    background-color: rgba(10, 14, 23, 0.85) !important;
    backdrop-filter: blur(8px) !important;
}

/* Main Content: Centered, max-width 1420px, responsive padding */
[data-testid="stAppViewContainer"] > .main {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

.main .block-container {
    max-width: 1420px !important;
    width: 100% !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    margin: 0 auto !important;
}

@media (max-width: 1440px) {
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
}

@media (max-width: 1024px) {
    .main .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-top: 1.5rem !important;
    }
}

/* Typography Scale */
h1, [data-testid="stHeadingWithActionElements"] h1 {
    font-size: 1.95rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.025em !important;
    line-height: 1.25 !important;
    margin-bottom: 0.25rem !important;
}

.main-subtitle {
    font-size: 0.975rem !important;
    color: var(--pp-text-secondary) !important;
    margin-bottom: 1.75rem !important;
    line-height: 1.5 !important;
}

h2, [data-testid="stHeadingWithActionElements"] h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
    letter-spacing: -0.015em !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

h3, [data-testid="stHeadingWithActionElements"] h3 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h5 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stWidgetLabel"] p, label p {
    font-size: 0.935rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.4rem !important;
}

/* Sidebar Styling: 260px - 280px */
section[data-testid="stSidebar"] {
    width: 275px !important;
    min-width: 260px !important;
    max-width: 285px !important;
    background-color: #0b0f17 !important;
    border-right: 1px solid var(--pp-border-subtle) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.75rem 1.25rem !important;
    max-width: 100% !important;
}

.sidebar-brand {
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.sidebar-brand-caption {
    font-size: 0.8rem;
    color: var(--pp-text-secondary);
    margin-bottom: 1rem;
}

.config-card {
    background-color: #111827;
    border: 1px solid var(--pp-border-subtle);
    border-radius: 10px;
    padding: 0.85rem;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.config-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    margin-bottom: 0.6rem;
}

.config-item {
    font-size: 0.85rem;
    color: var(--pp-text-secondary);
    margin-bottom: 0.35rem;
}

.config-item code {
    background-color: #1e293b;
    padding: 0.15rem 0.35rem;
    border-radius: 4px;
    color: #e2e8f0;
    font-size: 0.82rem;
}

.config-desc {
    font-size: 0.78rem;
    color: var(--pp-text-muted);
    margin-top: 0.5rem;
    line-height: 1.4;
    border-top: 1px solid #1e293b;
    padding-top: 0.45rem;
}

.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.45rem 0.5rem;
    border-radius: 6px;
    margin-bottom: 0.35rem;
    background-color: rgba(255, 255, 255, 0.02);
}

.pipeline-step:hover {
    background-color: rgba(255, 255, 255, 0.05);
}

.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #1e293b;
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    flex-shrink: 0;
}

.step-name {
    font-size: 0.86rem;
    color: #cbd5e1;
    font-weight: 500;
    line-height: 1.3;
}

/* Form Container */
[data-testid="stForm"] {
    background-color: var(--pp-bg-card) !important;
    border: 1px solid var(--pp-border-subtle) !important;
    border-radius: 14px !important;
    padding: 2rem 2.25rem !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4) !important;
    margin-bottom: 2rem !important;
}

.stTextInput input {
    height: 48px !important;
    font-size: 0.95rem !important;
    background-color: var(--pp-bg-input) !important;
    border: 1px solid var(--pp-border-subtle) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    padding: 0 1rem !important;
    transition: all 0.15s ease-in-out !important;
}

.stTextArea textarea {
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
    background-color: var(--pp-bg-input) !important;
    border: 1px solid var(--pp-border-subtle) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.15s ease-in-out !important;
}

.stTextInput input:hover, .stTextArea textarea:hover {
    border-color: var(--pp-border-active) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--pp-accent-red) !important;
    box-shadow: 0 0 0 2px rgba(225, 29, 72, 0.25) !important;
    outline: none !important;
}

[data-testid="stCheckbox"] {
    margin-top: 0.6rem !important;
    margin-bottom: 0.9rem !important;
}

[data-testid="stCheckbox"] label p {
    font-size: 0.925rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
}

/* Submit Button: Red accent, prominent, 52px height */
[data-testid="stBaseButton-primary"], 
button[kind="primary"], 
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--pp-accent-red) !important;
    color: #ffffff !important;
    border: 1px solid #f43f5e !important;
    border-radius: 10px !important;
    min-height: 52px !important;
    height: 52px !important;
    width: 100% !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(225, 29, 72, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
}

[data-testid="stBaseButton-primary"]:hover, 
button[kind="primary"]:hover, 
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--pp-accent-red-hover) !important;
    border-color: #fb7185 !important;
    box-shadow: 0 6px 20px rgba(225, 29, 72, 0.5) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stBaseButton-primary"]:active, 
button[kind="primary"]:active, 
[data-testid="stFormSubmitButton"] > button:active {
    background-color: var(--pp-accent-red-active) !important;
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(225, 29, 72, 0.4) !important;
}

[data-testid="stExpander"] {
    background-color: var(--pp-bg-card) !important;
    border: 1px solid var(--pp-border-subtle) !important;
    border-radius: 10px !important;
    margin-bottom: 0.85rem !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.975rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
}

[data-testid="stMetric"] {
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--pp-border-subtle) !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: var(--pp-text-secondary) !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

hr {
    margin: 1.5rem 0 !important;
    border-color: var(--pp-border-subtle) !important;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_application_workflow() -> ApplicationWorkflow:
    """Create and cache the configured end-to-end application workflow."""
    settings = get_settings()
    configure_logging(level=settings.log_level, settings=settings)

    if settings.llm_provider == "mock":
        llm_provider = DemoMockLLM()
        retriever = DemoRetriever()
    else:
        llm_provider = create_llm_provider(settings)
        retriever = None

    planning_workflow = PlanningWorkflow(llm_provider)

    signer = None
    if (
        settings.signing_private_key_path
        and settings.signing_public_key_path
        and settings.signing_private_key_path.exists()
        and settings.signing_public_key_path.exists()
    ):
        try:
            passphrase_bytes = (
                settings.signing_private_key_passphrase.get_secret_value().encode("utf-8")
                if settings.signing_private_key_passphrase
                else None
            )
            signer = Ed25519ResultSigner.from_pem_files(
                private_key_path=settings.signing_private_key_path,
                public_key_path=settings.signing_public_key_path,
                passphrase=passphrase_bytes,
            )
        except Exception:
            logger.warning(
                "Ed25519 signing keys unavailable or invalid; continuing without signing."
            )

    repository = None
    if settings.database_url:
        try:
            engine = create_database_engine(settings.database_url.get_secret_value())
            session_factory = create_session_factory(engine)
            repository = PostgresSignedRecommendationRepository(session_factory)
        except Exception:
            logger.warning("PostgreSQL persistence unavailable; continuing without database.")

    if settings.llm_provider != "mock":
        retriever = None
        if settings.llm_api_key and settings.chroma_persist_directory.exists():
            try:
                embedding_provider = GeminiEmbeddingProvider(
                    api_key=settings.llm_api_key,
                    model=settings.embedding_model,
                )
                vector_store = ChromaVectorStore(
                    collection_name=settings.chroma_collection_name,
                    embedding_model=settings.embedding_model,
                    persist_directory=settings.chroma_persist_directory,
                )
                retriever = RagRetriever(
                    embedding_provider=embedding_provider,
                    vector_store=vector_store,
                )
            except Exception:
                logger.warning("RAG retriever unavailable; continuing without vector store.")

    return ApplicationWorkflow(
        llm=llm_provider,
        planning_workflow=planning_workflow,
        retriever=retriever,
        signer=signer,
        repository=repository,
    )


def render_application_result(result: ApplicationWorkflowResult) -> None:
    """Render the full multi-agent outcome including evidence, critic, budget, and signature."""
    st.subheader("Workflow Outcome")

    if result.status is ApplicationWorkflowStatus.NEEDS_CLARIFICATION:
        st.warning("More information is needed before a plan can proceed.")
        if result.intent.clarification_questions:
            st.markdown("##### Please clarify the following:")
            for question in result.intent.clarification_questions:
                st.write(f"- {question}")
        return

    # 1. Final Recommendation Card
    if result.final_recommendation is not None:
        rec = result.final_recommendation
        if rec.status is FinalRecommendationStatus.READY:
            st.success(f"**Recommendation Ready**: {rec.summary}")
        else:
            st.error(f"**Recommendation Blocked**: {rec.summary}")

        if result.confidence_breakdown:
            cb = result.confidence_breakdown
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Overall Confidence", f"{cb.get('overall_confidence', 0.95):.2f}")
            c2.metric("Intent Extraction", f"{cb.get('intent_extraction', 1.0):.2f}")
            c3.metric("Requirement Coverage", f"{cb.get('requirement_coverage', 1.0):.2f}")
            c4.metric("Evidence Reliability", f"{cb.get('evidence_reliability', 1.0):.2f}")
            c5.metric("Evidence Relevance", f"{cb.get('evidence_relevance', 0.95):.2f}")

        if rec.selected_recommendations:
            st.markdown("#### Selected Recommendations")
            for item in rec.selected_recommendations:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {item.title}")
                        st.write(item.rationale)
                        if item.confidence_score is not None:
                            st.caption(f"Confidence score: {item.confidence_score:.2f}")
                        if item.evidence_ids:
                            st.caption(
                                f"Evidence citations: {len(item.evidence_ids)} sources cited"
                            )
                    with col2:
                        if item.estimated_cost is not None:
                            c_amt = item.estimated_cost.amount
                            c_cur = item.estimated_cost.currency
                            st.metric("Estimated Cost", f"{c_amt} {c_cur}")
                        else:
                            st.caption("No cost estimate provided")

    # 2. Cryptographic Integrity Signature & Verification
    if result.signed_recommendation is not None:
        with st.container(border=True):
            st.markdown("#### Cryptographic Integrity Signature (Ed25519)")
            sig_col1, sig_col2 = st.columns(2)
            with sig_col1:
                integ = result.signed_recommendation.integrity
                st.write(f"**Algorithm**: `{integ.algorithm.value}`")
                st.write(f"**Payload SHA-256**: `{integ.payload_sha256[:16]}...`")
                st.write(
                    f"**Public Key Fingerprint**: `{integ.public_key_fingerprint_sha256[:16]}...`"
                )
                if integ.signed_at:
                    st.write(
                        f"**Signed At (UTC)**: `{integ.signed_at.strftime('%Y-%m-%d %H:%M:%S')}`"
                    )
            with sig_col2:
                if result.verification is not None:
                    if result.verification.is_valid:
                        st.success("Verification: Valid Signature")
                    else:
                        st.error(f"Verification Failed: {result.verification.reason}")
                if result.persisted_result_id:
                    st.info(f"Persisted ID: `{result.persisted_result_id}`")
            st.caption(
                "🛡️ **Cryptographic Verification**: Confirms that this recommendation payload, "
                "request ID, and audit trace have not been altered since creation. "
                "It does not replace independent supplier credentials verification."
            )

    # 3. Deterministic Budget Assessment
    if result.budget_assessment is not None:
        ba = result.budget_assessment
        with st.expander(f"Budget Assessment: {ba.status.value.upper()}", expanded=True):
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Cost", f"{ba.total_cost.amount} {ba.total_cost.currency}")
            b_col2.metric("Limit", f"{ba.budget.maximum.amount} {ba.budget.maximum.currency}")
            feas_str = "YES" if ba.is_procurement_feasible else "NO (IMPLAUSIBLE)"
            b_col3.metric("Procurement Feasible", feas_str)
            b_col4.metric("Pricing Status", ba.pricing_status.upper())

            if ba.feasibility_notes:
                for note in ba.feasibility_notes:
                    st.warning(f"Feasibility Flag: {note}")

            if ba.violations:
                for v in ba.violations:
                    st.error(f"Budget Violation: {v.value}")

    # 4. Critic Evaluation Report
    if result.evaluation_report is not None:
        report = result.evaluation_report
        with st.expander(f"Critic Evaluation: {report.status.value.upper()}", expanded=False):
            st.write(f"**Summary**: {report.summary}")
            if report.criteria:
                st.markdown("##### Evaluated Criteria (A to G)")
                for crit in report.criteria:
                    icon = "PASS" if crit.passed else "FAIL"
                    st.write(f"[{icon}] **{crit.name}** ({crit.severity.value}): {crit.rationale}")
            if report.failed_requirements:
                st.markdown("##### Failed Requirements")
                for fr in report.failed_requirements:
                    st.error(f"- {fr}")
            if report.missing_evidence:
                st.markdown("##### Missing Evidence")
                for me in report.missing_evidence:
                    st.warning(f"- {me}")
            if report.issues:
                st.markdown("##### Critic Issues Flagged")
                for issue in report.issues:
                    st.warning(f"[{issue.code.value}] {issue.message}")

    # 5. Specialist Agent Outputs
    if result.agent_results:
        agent_header = f"Specialist Agent Results ({len(result.agent_results)} tasks)"
        with st.expander(agent_header, expanded=False):
            for agent_res in result.agent_results:
                dur_str = f"({agent_res.duration_ms:.1f}ms)" if agent_res.duration_ms else ""
                st.markdown(
                    f"**Agent**: `{agent_res.agent_name}` · Status: "
                    f"`{agent_res.status.value}` {dur_str}"
                )
                if agent_res.summary:
                    st.write(agent_res.summary)
                if agent_res.tasks_performed:
                    st.caption("Tasks Performed:")
                    for tp in agent_res.tasks_performed:
                        st.write(f"- {tp}")
                if agent_res.decisions_made:
                    st.caption("Decisions Made:")
                    for dm in agent_res.decisions_made:
                        st.write(f"- {dm}")
                if agent_res.warnings:
                    for w in agent_res.warnings:
                        st.warning(w)
                if agent_res.findings:
                    st.caption("Grounded Findings:")
                    for f in agent_res.findings:
                        st.write(f"- {f.statement}")
                st.divider()

    # 6. Evidence Sources (RAG)
    if result.evidence:
        with st.expander(f"Retrieved Evidence ({len(result.evidence)} sources)", expanded=False):
            for ev in result.evidence:
                class_badge = f"[{ev.classification.value.upper()}]"
                st.markdown(
                    f"**Source ID**: `{ev.source_id}` · **{class_badge}** · "
                    f"Relevance: {ev.relevance_score:.2f}"
                )
                if ev.provenance:
                    st.caption(f"Provenance: `{ev.provenance}`")
                if ev.associated_requirement:
                    st.caption(f"Associated Requirement: `{ev.associated_requirement}`")
                if ev.supporting_claim:
                    st.caption(f"Supporting Claim: {ev.supporting_claim}")
                st.write(f"> {ev.excerpt}")
                if ev.source_url:
                    st.caption(f"URL: {ev.source_url}")
                st.divider()

    # 7. Re-planning Iterations
    if result.replanning_iterations:
        replan_header = f"Re-planning Iterations ({len(result.replanning_iterations)} cycles)"
        with st.expander(replan_header, expanded=False):
            for it in result.replanning_iterations:
                st.markdown(f"##### Iteration {it.iteration_number}")
                st.write(f"Plan tasks: {len(it.task_plan.tasks)}")
                st.write(f"Evaluation: {it.evaluation_report.status.value}")
                if it.feedback_for_next_iteration:
                    st.write(f"Feedback passed: {it.feedback_for_next_iteration}")

    # 8. Initial Task Plan
    if result.task_plan is not None:
        with st.expander("Generated Task Plan", expanded=False):
            st.write(result.task_plan.planning_summary)
            for position, task in enumerate(result.task_plan.tasks, start=1):
                st.markdown(f"**Task {position}: {task.title}** ({task.kind.value})")
                st.write(task.instructions)

    # 9. Structured Intent & Workflow Trace
    with st.expander("Structured Intent"):
        st.json(result.intent.model_dump(mode="json"))

    with st.expander("Execution Trace & Latency Log"):
        st.json([event.model_dump(mode="json") for event in result.events])


def main() -> None:
    """Render the Streamlit planning and procurement interface."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("PrivateProcureAI")
    st.markdown(
        '<div class="main-subtitle">'
        "End-to-End Multi-Agent Procurement, Planning & Cryptographic Verification"
        "</div>",
        unsafe_allow_html=True,
    )

    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">🔒 PrivateProcureAI</div>'
            '<div class="sidebar-brand-caption">Enterprise Multi-Agent Procurement</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Runtime Configuration")
        settings = get_settings()

        mode_badge = "🟢 MOCK MODE" if settings.llm_provider == "mock" else "⚡ LIVE GEMINI"
        mode_desc = (
            "Deterministic mock provider active. Zero external API calls."
            if settings.llm_provider == "mock"
            else "Live Google Gemini API active."
        )

        st.markdown(
            f'<div class="config-card">'
            f'<div class="config-badge">{mode_badge}</div>'
            f'<div class="config-item"><strong>Provider:</strong> '
            f"<code>{settings.llm_provider}</code></div>"
            f'<div class="config-item"><strong>Model:</strong> '
            f"<code>{settings.llm_model}</code></div>"
            f'<div class="config-desc">{mode_desc}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### System Pipeline")
        pipeline_items = [
            ("1", "Intent Extraction & Validation"),
            ("2", "Multi-Source Vector RAG"),
            ("3", "Parallel Specialist Agents"),
            ("4", "Deterministic Budget Engine"),
            ("5", "Critic Evaluation (A–G)"),
            ("6", "Selection Optimization"),
            ("7", "Ed25519 Cryptographic Signing"),
        ]
        for step, name in pipeline_items:
            st.markdown(
                f'<div class="pipeline-step">'
                f'<span class="step-num">{step}</span>'
                f'<span class="step-name">{name}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Procurement & Planning Request")
    st.caption("Describe the outcome you want. Domain-general across all domains.")

    with st.form("planning_request_form"):
        goal = st.text_area(
            "Goal",
            key="goal_input",
            placeholder="Describe what you want to procure or plan (e.g. equipment, services).",
            height=150,
        )
        include_budget = st.checkbox(
            "Add a budget constraint",
            value=False,
            key="include_budget",
            help="Check to enforce a maximum budget on the plan.",
        )

        budget_left, budget_right = st.columns([3, 1])
        with budget_left:
            budget_amount = st.text_input(
                "Maximum amount",
                key="budget_amount",
                placeholder="e.g. 75000.00",
            )
        with budget_right:
            budget_currency = st.text_input(
                "Currency code",
                key="budget_currency",
                value="USD",
                max_chars=3,
            )

        budget_is_hard_limit = st.checkbox(
            "Treat budget as a hard limit",
            value=True,
            key="budget_hard_limit",
        )

        st.markdown("##### Preferences")
        preference_left, preference_right = st.columns(2)
        with preference_left:
            must_have = st.text_area(
                "Must-have requirements",
                key="must_have",
                placeholder="One requirement per line",
                height=130,
            )
            avoid = st.text_area(
                "Avoid",
                key="avoid",
                placeholder="One item per line",
                height=130,
            )
        with preference_right:
            nice_to_have = st.text_area(
                "Nice-to-have preferences",
                key="nice_to_have",
                placeholder="One preference per line",
                height=130,
            )
            notes = st.text_area(
                "Additional notes",
                key="notes",
                placeholder="Optional procurement notes or guidelines",
                height=130,
            )

        submitted = st.form_submit_button(
            "Run End-to-End Workflow",
            type="primary",
            icon=":material/rocket_launch:",
            width="stretch",
        )

    if submitted:
        try:
            effective_include_budget = include_budget or bool(
                budget_amount and budget_amount.strip()
            )
            request = build_planning_request(
                goal=goal,
                include_budget=effective_include_budget,
                budget_amount=budget_amount,
                budget_currency=budget_currency or "USD",
                budget_is_hard_limit=budget_is_hard_limit,
                must_have=must_have,
                nice_to_have=nice_to_have,
                avoid=avoid,
                notes=notes,
            )
        except (ValidationError, ValueError) as error:
            st.error(str(error))
        else:
            with st.status("Executing procurement workflow…", expanded=True) as status_box:
                try:
                    status_box.write("Extracting intent, planning tasks, and executing agents…")
                    workflow = get_application_workflow()
                    result = workflow.run(request)
                    st.session_state.workflow_result = result
                    status_box.update(label="Workflow complete", state="complete")
                except Exception as exc:
                    logger.exception("Application workflow failed in Streamlit UI")
                    st.session_state.workflow_result = None
                    status_box.update(label="Workflow failed", state="error")
                    st.error(f"The workflow could not be completed: {exc}")

    if st.session_state.workflow_result is not None:
        render_application_result(st.session_state.workflow_result)


if __name__ == "__main__":
    main()
