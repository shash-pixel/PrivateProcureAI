"""FastAPI application factory for the planning workflow."""

from __future__ import annotations

import logging
from typing import Protocol

from fastapi import APIRouter, FastAPI, HTTPException, status

from privateprocureai.api.models import HealthResponse
from privateprocureai.schemas import (
    ApplicationWorkflowResult,
    PlanningRequest,
    WorkflowResult,
)

logger = logging.getLogger(__name__)


class PlanningRunner(Protocol):
    """Minimal contract required by the HTTP planning endpoint."""

    def run(self, planning_request: PlanningRequest) -> WorkflowResult:
        """Run planning for one validated request."""


class ApplicationRunner(Protocol):
    """Contract required by the full end-to-end workflow endpoint."""

    def run(self, request: PlanningRequest) -> ApplicationWorkflowResult:
        """Run the complete multi-agent workflow for one request."""


def create_api_app(
    planning_runner: PlanningRunner,
    application_runner: ApplicationRunner | None = None,
) -> FastAPI:
    """Create a FastAPI application around a supplied planning workflow."""
    app = FastAPI(
        title="PrivateProcureAI API",
        version="0.1.0",
        description=("A domain-general multi-agent AI planning and procurement API."),
    )

    router = APIRouter(prefix="/v1")

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health_check() -> HealthResponse:
        """Return a minimal liveness response."""
        return HealthResponse(status="ok")

    @router.post(
        "/plans",
        response_model=WorkflowResult,
        status_code=status.HTTP_200_OK,
        tags=["planning"],
    )
    def create_plan(planning_request: PlanningRequest) -> WorkflowResult:
        """Extract intent and create a validated task plan."""
        try:
            return planning_runner.run(planning_request)
        except Exception as error:
            logger.exception(
                "Planning workflow failed for request_id=%s",
                planning_request.request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Planning service is temporarily unavailable.",
            ) from error

    if application_runner is not None:

        @router.post(
            "/workflow/run",
            response_model=ApplicationWorkflowResult,
            status_code=status.HTTP_200_OK,
            tags=["workflow"],
        )
        def run_full_workflow(
            request: PlanningRequest,
        ) -> ApplicationWorkflowResult:
            """Run the complete multi-agent procurement workflow."""
            try:
                return application_runner.run(request)
            except Exception as error:
                logger.exception(
                    "Application workflow failed for request_id=%s",
                    request.request_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Application workflow service is temporarily unavailable.",
                ) from error

    app.include_router(router)

    return app
