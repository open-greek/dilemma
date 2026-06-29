"""Pytest configuration shared across the dilemma test suite.

Defines the ``--run-slow`` flag (and the auto-skip of ``@pytest.mark.slow``
tests without it) at the suite root so it works for ``pytest tests/`` as well
as ``pytest tests/tagger/``.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests that require model weights",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Needs --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
