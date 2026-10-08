"""Provider-independent contracts for structured LLM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class StructuredGenerationRequest(BaseModel):
    """Provider-neutral input for one structured-generation request."""

    model_config=ConfigDict(extra="forbid",str_strip_whitespace=True)

    system_instruction: str=Field(min_length=1,max_length=10_000)
    user_prompt:str=Field(min_length=1,max_length=20_000)
    temperature:float | None=Field(default=None,ge=0.0,le=2.0)

class StructuredLLM(ABC):
    """Abstract interface implemented by every LLM provider adapter."""

    @abstractmethod
    def generate_structured(
        self,
        request: StructuredGenerationRequest,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Generate data that validates against ``response_model``."""