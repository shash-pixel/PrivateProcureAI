"""Small API-specific response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health-check response without exposing configuration or secrets."""

    status: str
