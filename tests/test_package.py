"""Smoke tests for the Module 0 package skeleton."""

from privateprocureai import __version__


def test_package_exposes_a_version() -> None:
    """The importable package provides a version for diagnostics and packaging."""
    assert __version__ == "0.1.0"