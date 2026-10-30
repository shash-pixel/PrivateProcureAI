"""Schemas for qualitative and deterministic plan evaluation."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import Field

from privateprocureai.schemas.budget import BudgetAssessment
from privateprocureai.schemas.intent import ExtractedIntent
from privateprocureai.schemas.models import AgentResult, DomainModel


class EvaluationStatus(StrEnum):
    """Terminal quality outcome for one candidate plan."""

    PASS = "pass"
    FAIL = "fail"


class EvaluationSeverity(StrEnum):
    """Impact of an evaluation concern."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class EvaluationIssueCode(StrEnum):
    """Known reasons a candidate plan can fail evaluation."""

    AGENT_FAILURE = "agent_failure"
    HARD_BUDGET_VIOLATION = "hard_budget_violation"
    BUDGET_UNEVALUABLE = "budget_unevaluable"
    CRITICAL_QUALITY_CONCERN = "critical_quality_concern"


class EvaluationCriterion(DomainModel):
    """One qualitative criterion assessed by the critic agent."""

    name: str = Field(min_length=3, max_length=200)
    passed: bool
    severity: EvaluationSeverity
    rationale: str = Field(min_length=3, max_length=2_000)
    related_task_ids: list[UUID] = Field(default_factory=list)


class EvaluationAnalysis(DomainModel):
    """Structured Gemini output before deterministic rules are applied."""

    summary: str = Field(min_length=3, max_length=2_000)
    criteria: list[EvaluationCriterion] = Field(default_factory=list)


class EvaluationIssue(DomainModel):
    """A deterministic or critic-reported reason for concern."""

    code: EvaluationIssueCode
    severity: EvaluationSeverity
    message: str = Field(min_length=3, max_length=2_000)


class EvaluationInput(DomainModel):
    """All validated data needed to evaluate one candidate plan."""

    intent: ExtractedIntent
    agent_results: list[AgentResult] = Field(min_length=1)
    budget_assessment: BudgetAssessment | None = None


class EvaluationReport(DomainModel):
    """Final evaluation outcome after deterministic safeguards."""

    status: EvaluationStatus
    summary: str = Field(min_length=3, max_length=2_000)
    criteria: list[EvaluationCriterion] = Field(default_factory=list)
    issues: list[EvaluationIssue] = Field(default_factory=list)
    requires_replanning: bool