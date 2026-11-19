"""Production composition root for the FastAPI application."""

from fastapi import FastAPI

from privateprocureai.api.app import create_api_app
from privateprocureai.config import get_settings
from privateprocureai.database.repositories import PostgresSignedRecommendationRepository
from privateprocureai.database.session import create_database_engine, create_session_factory
from privateprocureai.integrity.signing import Ed25519ResultSigner
from privateprocureai.llm import create_llm_provider
from privateprocureai.observability import configure_logging
from privateprocureai.orchestration import ApplicationWorkflow, PlanningWorkflow
from privateprocureai.rag.embeddings import GeminiEmbeddingProvider
from privateprocureai.rag.retrieval import RagRetriever
from privateprocureai.rag.vector_store import ChromaVectorStore


def create_application() -> FastAPI:
    """Create the configured API application for Uvicorn with full workflow support."""
    settings = get_settings()
    configure_logging(level=settings.log_level, settings=settings)

    llm_provider = create_llm_provider(settings)
    planning_workflow = PlanningWorkflow(llm_provider)

    # Optional signer
    signer = None
    if (
        settings.signing_private_key_path
        and settings.signing_public_key_path
        and settings.signing_private_key_path.exists()
        and settings.signing_public_key_path.exists()
    ):
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

    # Optional database persistence
    repository = None
    if settings.database_url:
        engine = create_database_engine(settings.database_url.get_secret_value())
        session_factory = create_session_factory(engine)
        repository = PostgresSignedRecommendationRepository(session_factory)

    # Optional RAG retriever
    retriever = None
    if settings.llm_api_key and settings.chroma_persist_directory.exists():
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

    application_workflow = ApplicationWorkflow(
        llm=llm_provider,
        planning_workflow=planning_workflow,
        retriever=retriever,
        signer=signer,
        repository=repository,
    )

    return create_api_app(
        planning_runner=planning_workflow,
        application_runner=application_workflow,
    )
