"""Pytest fixtures and markers for famail_temporal tests."""

import pytest

from famail_temporal.utils.seeding import set_all_seeds


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run tests marked @pytest.mark.slow")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselected by default)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def seeded():
    """Autouse fixture — set all seeds to a known value before each test."""
    set_all_seeds(42)
