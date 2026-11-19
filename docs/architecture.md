# PrivateProcureAI System Architecture

PrivateProcureAI is a domain-general, multi-agent AI planning and procurement system engineered with deterministic safeguards, retrieval-augmented generation (RAG), bounded re-planning loops, and cryptographic integrity verification.

---

## 1. High-Level Architecture

The system coordinates specialized agents and deterministic verification layers through structured schemas:

```
                            ┌──────────────────────────────────────────────┐
                            │            Client Interface                  │
                            │      (Streamlit Web UI / FastAPI)            │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │              Planning Workflow               │
                            │   • Intent Extraction (LLM + Pydantic)       │
                            │   • Task Planning (DAG Dependency Graph)     │
                            └──────────────────────┬───────────────────────┘
                                                   │
                       [Needs Clarification?] ─────┼─────► [Early Safe Exit]
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │        Parallel Specialist Execution         │
                            │   • RAG Retrieval (ChromaDB + Embeddings)    │
                            │   • Research Agent (Evidence Grounding)      │
                            │   • Recommendation Agent (Candidate Options) │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │      Deterministic Budget Validation         │
                            │   • Decimal Precision Cost Aggregation       │
                            │   • Hard Constraint vs Soft Limit Checks     │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │            Critic Agent Review               │
                            │   • Evidence Coverage & Soundness            │
                            │   • Deterministic Violation Enforcement      │
                            └──────────────────────┬───────────────────────┘
                                                   │
                       [Requires Re-planning?] ────┴─────► ┌───────────────────────────┐
                                      │                    │ Bounded Re-Planning Loop  │
                                      │                    │  • Max attempts (e.g. 3)  │
                                      │                    │  • Feeds critic feedback  │
                                      │                    └─────────────┬─────────────┘
                                      ▼                                  │
                            ┌──────────────────────────────────────────┐ │
                            │     Final Candidate Optimization         │◄┘
                            │   • Domain-General Candidate Ranking     │
                            │   • Deterministic Cost Re-Validation     │
                            └──────────────────────┬───────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │     Ed25519 Cryptographic Signing            │
                            │   • SHA-256 Canonical JSON Payload Hash      │
                            │   • Detached Signature & Verification        │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │       Optional PostgreSQL Persistence        │
                            │   • Immutable Signed Recommendation Record   │
                            │   • User Preference Memory Revisions         │
                            └──────────────────────────────────────────────┘
```

---

## 2. Request Lifecycle & Sequence Flow

The following sequence details how a request moves across components:

```text
User / API Client
       │
       │ 1. POST /v1/workflow/run (PlanningRequest)
       ▼
ApplicationWorkflow
       │
       │ 2. IntentExtractor.extract()
       ├─────────────────────────────────────────► Gemini / Mock LLM
       │◄───────────────────────────────────────── ExtractedIntent
       │
       │ [If clarification_needed: return early]
       │
       │ 3. TaskPlanner.create_plan()
       ├─────────────────────────────────────────► Gemini / Mock LLM
       │◄───────────────────────────────────────── TaskPlan (DAG validated)
       │
       │ 4. ParallelAgentExecutor.execute()
       │   ├── Task requires retrieval:
       │   │   RagRetriever.retrieve() ──────────► ChromaDB Vector Search
       │   ├── ResearchAgent.run() ──────────────► Gemini / Mock LLM
       │   └── RecommendationAgent.run() ────────► Gemini / Mock LLM
       │◄───────────────────────────────────────── PlanExecutionResult
       │
       │ 5. BudgetEngine.evaluate() (Deterministic Decimal rules)
       │◄───────────────────────────────────────── BudgetAssessment
       │
       │ 6. CriticAgent.evaluate()
       ├─────────────────────────────────────────► Gemini / Mock LLM
       │◄───────────────────────────────────────── EvaluationReport
       │
       │ [If requires_replanning: ReplanningCoordinator loops up to 3x]
       │
       │ 7. FinalRecommendationOptimizer.optimize()
       ├─────────────────────────────────────────► Gemini / Mock LLM
       │◄───────────────────────────────────────── FinalRecommendation (READY/BLOCKED)
       │
       │ 8. Ed25519ResultSigner.sign() & verify()
       │   ├── Canonicalize JSON payload
       │   ├── SHA-256 payload hash
       │   └── Ed25519 private key signature
       │◄───────────────────────────────────────── SignedFinalRecommendation
       │
       │ 9. [Optional] PostgresSignedRecommendationRepository.save()
       │◄───────────────────────────────────────── Record UUID
       │
       ▼
ApplicationWorkflowResult (Returned to Client / Streamlit UI)
```

---

## 3. Core Architectural Pillars

### 3.1. Provider-Neutral LLM Abstraction
All model interactions pass through the `StructuredLLM` protocol:
- **`GeminiProvider`**: Uses the official Google GenAI SDK (`google-genai`) with Pydantic JSON schema constraints, enforcing strict output schemas at the model boundary.
- **`MockLLMProvider`**: Queues deterministic fixtures for instant, zero-cost unit and integration tests without network access or live API credentials.

### 3.2. Evidence Grounding & Vector Retrieval (RAG)
- Unstructured documentation and catalogs are chunked by `DocumentIngestor` with configurable token overlap and typed metadata.
- Embeddings are generated via `GeminiEmbeddingProvider` (`gemini-embedding-2`).
- Storage and similarity search are managed by `ChromaVectorStore` with cosine distance.
- `RagRetriever` converts matches into strongly-typed `Evidence` records and enforces a minimum relevance threshold.
- Specialist agents (`ResearchAgent`, `RecommendationAgent`) validate evidence IDs and reject hallucinated citations.

### 3.3. Deterministic Budget Enforcement
- Spending constraints are treated as hard non-negotiable boundaries, not model suggestions.
- All monetary math uses Python's `Decimal` via the `Money` schema to avoid binary floating-point rounding errors.
- `BudgetEngine` verifies currency matching, computes total cost, remaining allowances, and emits structured `BudgetViolation` codes.

### 3.4. Qualitative Review and Bounded Re-Planning
- `CriticAgent` evaluates candidate outputs against intent, evidence grounding, and deterministic budget assessments.
- When critical issues or hard budget violations arise, the `ReplanningCoordinator` initiates a bounded revision loop using a LangGraph `StateGraph`.
- Re-planning attempts are strictly capped (default: 3 iterations) to eliminate infinite loops and control token spend.

### 3.5. Cryptographic Integrity (Ed25519)
- Only final recommendations validated as `READY` can receive a signature.
- Recommendation payloads are canonicalized (deterministic sorted JSON) and hashed with SHA-256.
- The payload is signed with an Ed25519 private key, producing an `IntegrityRecord` containing:
  - Algorithm (`ed25519`)
  - Payload SHA-256 hash
  - Base64-encoded signature
  - Public-key SHA-256 fingerprint
- `verify()` independently recalculates the hash and verifies the signature using the corresponding public key.

### 3.6. Security, Redaction & Observability
- `SecretMaskingFilter` automatically scrubs Gemini API keys (`AIza...`), OpenAI keys (`sk-...`), Bearer tokens, private key blocks, and database passwords from all logs.
- Sensitive user inputs are truncated and sanitized.
- Every workflow step is instrumented with `timed_step`, recording wall-clock latency (`duration_ms`), agent name, request ID, and status on `WorkflowEvent` objects.
