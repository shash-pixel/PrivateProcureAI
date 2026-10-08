"""Exceptions raised by LLM provider adapters."""


class LLMProviderError(Exception):
    """Base exception for provider communication or configuration failures."""


class StructuredOutputError(LLMProviderError):
    """Raised when a provider response cannot satisfy the requested schema."""


class ProviderNotImplementedError(LLMProviderError):
    """Raised when configuration selects an adapter not implemented yet."""