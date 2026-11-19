"""Tests for FastAPI planning routes."""

from uuid import uuid4

from fastapi.testclient import TestClient

from privateprocureai.api import create_api_app
from privateprocureai.schemas import (
    ExtractedIntent,
    PlanningRequest,
    WorkflowResult,
    WorkflowStatus,
)


class StubPlanningWorkflow:
    """Deterministic workflow substitute used by API tests."""

    def run(self, planning_request: PlanningRequest) -> WorkflowResult:
        return WorkflowResult(
            request_id=planning_request.request_id,
            status=WorkflowStatus.NEEDS_CLARIFICATION,
            intent=ExtractedIntent(
                request_id=planning_request.request_id,
                objective="Clarify the requested outcome.",
                constraints=[],
                preferences=[],
                assumptions=[],
                clarification_needed=True,
                clarification_questions=["What outcome should the plan optimize for?"],
                confidence_score=0.4,
            ),
            task_plan=None,
            events=[],
        )


class FailingPlanningWorkflow:
    """Workflow substitute that demonstrates safe API failure handling."""

    def run(self, planning_request: PlanningRequest) -> WorkflowResult:
        del planning_request
        raise RuntimeError("Simulated internal failure.")


def test_health_endpoint_returns_ok() -> None:
    """The API exposes a lightweight liveness endpoint."""
    client = TestClient(create_api_app(StubPlanningWorkflow()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_plan_endpoint_returns_workflow_result() -> None:
    """A valid request is passed to the workflow and serialized as JSON."""
    client = TestClient(create_api_app(StubPlanningWorkflow()))
    request = PlanningRequest(
        request_id=uuid4(),
        goal="Create a suitable plan for the stated objective.",
    )

    response = client.post(
        "/v1/plans",
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["request_id"] == str(request.request_id)


def test_plan_endpoint_hides_internal_failures() -> None:
    """Internal exception details are logged but not disclosed to clients."""
    client = TestClient(create_api_app(FailingPlanningWorkflow()))

    response = client.post(
        "/v1/plans",
        json={
            "goal": "Create a supported plan.",
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Planning service is temporarily unavailable."}
