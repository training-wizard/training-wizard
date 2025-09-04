"""Pytest configuration file."""

from typing import Any


def pytest_addoption(parser: Any):
    """Add custom command line options to pytest."""
    parser.addoption("--ci", action="store_true", default=False, help="run tests in CI environment")
