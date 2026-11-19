"""Tests for logging, secret masking, context propagation, and timing instrumentation."""

import logging
from uuid import uuid4

import pytest

from privateprocureai.observability import (
    RequestContext,
    SecretMaskingFilter,
    clear_registered_secrets,
    get_agent_name,
    get_request_id,
    get_workflow_step,
    redact_secrets,
    register_secret,
    sanitize_user_input,
    timed_step,
)
from privateprocureai.schemas import WorkflowEvent, WorkflowStep


def test_redact_secrets_gemini_api_key() -> None:
    """Gemini API keys matching standard pattern are masked."""
    key = "AIzaSyD-1234567890abcdefghijklmnopqrst"
    text = f"Connecting with key {key} to Gemini"
    redacted = redact_secrets(text)
    assert key not in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_redact_secrets_openai_key() -> None:
    """OpenAI style API keys are masked."""
    key = "sk-proj-1234567890abcdefghijklmnopqr"
    text = f"Using {key} for external call"
    redacted = redact_secrets(text)
    assert key not in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_redact_secrets_database_url_password() -> None:
    """Database URLs have passwords stripped while preserving username and host."""
    url = "postgresql+psycopg://dbuser:supersecretpass123@localhost:5432/privateprocureai"
    redacted = redact_secrets(f"Connection failed to {url}")
    assert "supersecretpass123" not in redacted
    assert "[REDACTED_PASSWORD]" in redacted
    assert "dbuser" in redacted


def test_redact_registered_custom_secret() -> None:
    """Explicitly registered secrets like passphrases are masked."""
    passphrase = "my-ultra-confidential-passphrase"
    register_secret(passphrase)
    try:
        text = f"Decryption failed with passphrase {passphrase} for key"
        redacted = redact_secrets(text)
        assert passphrase not in redacted
        assert "[REDACTED]" in redacted
    finally:
        clear_registered_secrets()


def test_sanitize_user_input_truncates_and_redacts() -> None:
    """Long user inputs are safely truncated and cleaned of newlines and secrets."""
    long_input = "Buy laptops " + ("word " * 50) + "AIzaSyD-1234567890abcdefghijklmnopqrst"
    sanitized = sanitize_user_input(long_input, max_length=50)
    assert len(sanitized) < len(long_input)
    assert "truncated" in sanitized
    assert "AIza" not in sanitized


def test_secret_masking_filter_on_log_record() -> None:
    """SecretMaskingFilter scrubs record.msg and args."""
    filter_ = SecretMaskingFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Found key: %s",
        args=("AIzaSyD-1234567890abcdefghijklmnopqrst",),
        exc_info=None,
    )
    filter_.filter(record)
    assert "AIza" not in record.args[0]
    assert "[REDACTED_API_KEY]" in record.args[0]


def test_request_context_scopes_variables() -> None:
    """RequestContext sets and resets context variables cleanly."""
    req_id = uuid4()
    assert get_request_id() is None
    assert get_workflow_step() is None
    assert get_agent_name() is None

    with RequestContext(request_id=req_id, step="intent_extraction", agent_name="test_agent"):
        assert get_request_id() == str(req_id)
        assert get_workflow_step() == "intent_extraction"
        assert get_agent_name() == "test_agent"

    assert get_request_id() is None
    assert get_workflow_step() is None
    assert get_agent_name() is None


def test_timed_step_measures_success() -> None:
    """timed_step logs success and captures duration."""
    with timed_step("test_step") as timing:
        # small workload
        _ = sum(range(1000))

    assert timing.status == "success"
    assert timing.duration_ms >= 0.0
    assert timing.error is None


def test_timed_step_captures_failure_and_redacts_error() -> None:
    """timed_step marks failure and redacts error message upon exception."""
    secret = "secret-passphrase-to-hide"
    register_secret(secret)
    try:
        with pytest.raises(RuntimeError):
            with timed_step("failing_step") as timing:
                raise RuntimeError(f"Error with {secret}")

        assert timing.status == "failed"
        assert timing.duration_ms >= 0.0
        assert secret not in timing.safe_error_message
        assert "[REDACTED]" in timing.safe_error_message
    finally:
        clear_registered_secrets()


def test_workflow_event_structured_fields() -> None:
    """WorkflowEvent schema supports all observability fields."""
    event = WorkflowEvent(
        step=WorkflowStep.BUDGET_VALIDATION,
        message="Validated deterministic budget constraints.",
        status="success",
        duration_ms=12.4,
        agent_name="budget_engine",
    )
    assert event.step == WorkflowStep.BUDGET_VALIDATION
    assert event.duration_ms == 12.4
    assert event.status == "success"
    assert event.agent_name == "budget_engine"
