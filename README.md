# PrivateProcureAI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Architecture: LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Security: Ed25519 Signed](https://img.shields.io/badge/integrity-Ed25519-green.svg)](https://cryptography.io/)

**PrivateProcureAI** is an enterprise-grade, domain-general multi-agent AI procurement and planning system. It translates high-level natural language goals and user preferences into evidence-grounded, budget-compliant, and cryptographically signed recommendations.

---

## Architecture Overview

PrivateProcureAI separates non-deterministic AI generation from deterministic policy enforcement:

```text
User / API Client
       │
       ▼
[Planning Request]
       │
       ├─► 1. Intent Extraction (Structured JSON via Gemini/Mock)
       │      └─► [Needs Clarification?] ──► Safe Early Exit
       │
       ├─► 2. Task Planning (DAG Decomposition & Dependency Validation)
       │
       ├─► 3. Parallel Specialist Execution (ThreadPoolExecutor)
       │      ├─► RAG Evidence Retrieval (ChromaDB Vector Store)
       │      ├─► Research Agent (Evidence synthesis & citation validation)
       │      └─► Recommendation Agent (Candidate option generation)
       │
       ├─► 4. Deterministic Budget Enforcement (Decimal precision rules)
       │
       ├─► 5. Critic Agent Review & Bounded Re-Planning
       │      └─► [Issues Flagged?] ──► LangGraph Re-Planning Loop (Max 3 attempts)
       │
       ├─► 6. Final Candidate Selection & Cost Validation
       │
       ├─► 7. Cryptographic Signing & Verification (Ed25519 detached signatures)
       │
       └─► 8. Optional Persistence (PostgreSQL immutable audit records)
```

---

## Request Lifecycle

```text
FastAPI / Streamlit UI
       │
       ▼
ApplicationWorkflow.run()
  ├── 1. Request Context initialized with request_id
  ├── 2. PlanningWorkflow extracts intent; pauses if clarification is required
  ├── 3. PlanTask dependencies resolved topologically; independent tasks run in parallel
  ├── 4. Vector similarity search executes only for tasks requiring external retrieval
  ├── 5. Evidence IDs are validated across agent findings to eliminate hallucinated sources
  ├── 6. BudgetEngine evaluates costs using Decimal math (hard limits vs soft guidelines)
  ├── 7. CriticAgent combines qualitative analysis with deterministic failure rules
  ├── 8. ReplanningCoordinator loops with feedback if quality or budget issues are detected
  ├── 9. FinalRecommendationOptimizer selects verified candidate IDs
  ├── 10. Ed25519ResultSigner canonicalizes the payload, computes SHA-256, and signs it
  └── 11. PostgresSignedRecommendationRepository stores the immutable audit record
```

---

## Key AI Engineering Decisions

| Requirement | Conventional AI Pattern | PrivateProcureAI Architecture |
| :--- | :--- | :--- |
| **Budget Enforcement** | Prompting LLM: *"Please stay under \$1,000"* | **Deterministic Python `BudgetEngine`** with `Decimal` arithmetic. Violations trigger automated plan rejection or re-planning. |
| **Source Grounding** | Allowing LLMs to cite general web knowledge | **Strict RAG citation verification**. Agents must cite explicit chunk UUIDs from ChromaDB; uncited claims are rejected. |
| **Output Consistency** | Raw text markdown parsing | **Pydantic schema enforcement** at provider boundary (`response_mime_type="application/json"`). |
| **Iterative Revision** | Unbounded model loops | **Bounded LangGraph `StateGraph`** with explicit iteration counters (capped at 3). |
| **Result Tampering** | Trusting plain database records | **Ed25519 digital signatures** with detached SHA-256 canonical payload hashing. |
| **Observability & Secrets** | Unfiltered stdout logging | **Contextual logging with `SecretMaskingFilter`**. Automatic scrubbing of API keys, passphrases, and connection strings. |

---

## Getting Started

### 1. Prerequisites
- **Python >= 3.11**
- Virtual environment tool (`venv`, `conda`, or `uv`)

### 2. Installation

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/your-username/PrivateProcureAI.git
cd PrivateProcureAI

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

### 3. Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

Edit `.env` to configure your runtime:

```ini
# Application environment
PP_AI_ENVIRONMENT=development
PP_AI_LOG_LEVEL=INFO

# Provider selection: gemini, mock, or openai
PP_AI_LLM_PROVIDER=gemini
PP_AI_LLM_MODEL=gemini-2.5-flash

# Gemini API Key (Required for remote LLM and embeddings)
PP_AI_LLM_API_KEY=your_gemini_api_key_here

# Request timeout in seconds
PP_AI_REQUEST_TIMEOUT_SECONDS=30

# Vector database settings
PP_AI_EMBEDDING_MODEL=gemini-embedding-2
PP_AI_CHROMA_PERSIST_DIRECTORY=data/chroma
PP_AI_CHROMA_COLLECTION_NAME=privateprocureai_chunks

# Optional Ed25519 signing keys
PP_AI_SIGNING_PRIVATE_KEY_PATH=keys/privateprocureai-private.pem
PP_AI_SIGNING_PUBLIC_KEY_PATH=keys/privateprocureai-public.pem
# PP_AI_SIGNING_PRIVATE_KEY_PASSPHRASE=your_passphrase_here

# Optional PostgreSQL database
# PP_AI_DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/privateprocureai
```

> **Note on Testing & Mock Mode**: The entire test suite and application can run in **Mock Mode** (`PP_AI_LLM_PROVIDER=mock`) without an API key or network access.

---

## Generating Cryptographic Signing Keys

To enable Ed25519 recommendation signing, generate key files:

```python
from pathlib import Path
from privateprocureai.integrity.signing import generate_ed25519_keypair

generate_ed25519_keypair(
    private_key_path=Path("keys/privateprocureai-private.pem"),
    public_key_path=Path("keys/privateprocureai-public.pem"),
    passphrase=None,  # Or pass b"your-passphrase"
)
```

Set `PP_AI_SIGNING_PRIVATE_KEY_PATH` and `PP_AI_SIGNING_PUBLIC_KEY_PATH` in `.env`.

---

## Running the Application

### Interactive Streamlit Interface

Launch the interactive UI to submit procurement goals and inspect task plans, agent findings, budget evaluations, and signature verifications:

```bash
streamlit run streamlit_app.py
```

### FastAPI Production Server

Start the REST API with Uvicorn:

```bash
uvicorn privateprocureai.api.main:create_application --factory --host 0.0.0.0 --port 8000
```

---

## API Usage Examples

### 1. Health Liveness Probe
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### 2. Full End-to-End Workflow Execution
```bash
curl -X POST http://localhost:8000/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Procure a redundant power backup solution for a server cluster.",
    "budget": {
      "maximum": {"amount": "1500.00", "currency": "USD"},
      "is_hard_limit": true
    },
    "preferences": {
      "must_have": ["Rack mountable", "SNMP network monitoring"],
      "nice_to_have": ["Hot-swappable batteries"]
    }
  }'
```

**Response Summary:**
```json
{
  "request_id": "78792606-613d-4fac-80d7-63a43f9c320c",
  "status": "completed",
  "intent": {
    "objective": "Procure redundant power backup solution for server cluster.",
    "confidence_score": 0.95
  },
  "final_recommendation": {
    "status": "ready",
    "summary": "Selected redundant dual-UPS package meeting all power and network requirements.",
    "selected_recommendations": [
      {
        "title": "Dual Enterprise Online UPS System",
        "rationale": "Includes SNMP management card and supports rack deployment within budget.",
        "estimated_cost": {"amount": "1420.00", "currency": "USD"}
      }
    ]
  },
  "budget_assessment": {
    "status": "compliant",
    "total_cost": {"amount": "1420.00", "currency": "USD"},
    "remaining_amount": "80.00"
  },
  "signed_recommendation": {
    "integrity": {
      "algorithm": "ed25519",
      "payload_sha256": "8f4a7c1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a",
      "signature_base64": "U3VwZXJTZWNyZXRTaWduYXR1cmVCYXNlNjRFZDI1NTE5..."
    }
  },
  "verification": {
    "is_valid": true
  }
}
```

---

## Testing & Code Quality

Run tests using Pytest:

```bash
# Run complete test suite (unit, integration, and security tests)
pytest

# Run with verbose output and coverage
pytest -v
```

Run code formatting and lint checks:

```bash
# Run Ruff linting
ruff check .

# Run Ruff formatting check
ruff format --check .
```

---

## Generic Demo Walkthrough

1. **Submit Request**: A user inputs: *"Procure replacement office network switches with PoE+ support for 24 workstations. Total expenditure cannot exceed \$2,000 USD."*
2. **Intent Extraction**: The system creates an `ExtractedIntent` object identifying PoE+ as a mandatory hard constraint and extracts a \$2,000 USD hard budget limit.
3. **Planning**: `TaskPlanner` breaks the goal into:
   - Task 1: Research available 24-port PoE+ managed switches (`requires_retrieval=True`).
   - Task 2: Recommend best options (`depends_on=[Task 1]`).
4. **Execution**:
   - `ParallelAgentExecutor` invokes `RagRetriever` to pull relevant product specs from the local Chroma store.
   - `ResearchAgent` synthesizes findings citing specific retrieved excerpt UUIDs.
   - `RecommendationAgent` evaluates the research findings and generates candidates.
5. **Deterministic Budget Check**: `BudgetEngine` tallies costs (\$1,750 USD) against the \$2,000 USD limit using decimal arithmetic, verifying compliance.
6. **Critic Evaluation**: `CriticAgent` confirms evidence citation coverage and verifies budget feasibility.
7. **Optimization & Signing**: `FinalRecommendationOptimizer` selects the top candidate. `Ed25519ResultSigner` hashes and signs the canonical payload.
8. **Presentation & Verification**: The Streamlit UI displays the recommendation card with verification badge and detailed step latency logs.

---

## What This Demonstrates (Portfolio & Interview Readiness)

- **AI Engineering Maturity**: Does not rely on prompting alone for mission-critical business rules; separates stochastic generation from deterministic arithmetic and cryptography.
- **Resilient Distributed Workflows**: Composes DAG-based task planners with dependency-aware parallel thread executors and bounded iterative loops.
- **Enterprise Security Practices**: Detached Ed25519 asymmetric signatures prevent payload tampering; automated log sanitization eliminates secret leakage (`AIza...`, Bearer tokens, passwords).
- **Production Clean Architecture**: Strict decoupling of protocol interfaces (`StructuredLLM`, `VectorStore`, `EvidenceRetriever`) allows 100% of the test suite to execute in milliseconds without external cloud dependencies.

---

## Known Limitations

- **Provider Rate Limits**: Gemini API tier limits apply when running against remote models. High-volume concurrent runs should configure retries and backoff.
- **ChromaDB Local Storage**: Default configuration uses an embedded local Chroma instance. For high-scale distributed deployments, an external vector cluster (e.g. pgvector or Pinecone) is recommended.
- **Single-Node Thread Execution**: Specialist tasks run in parallel via Python `ThreadPoolExecutor`. For cluster-scale agent workloads, Celery or Temporal can wrap the same `TaskAgent` interface.
- **Optional Database Persistence**: PostgreSQL persistence is purely optional; the core pipeline functions stand-alone with local in-memory or filesystem state.
