"""Slow tests (real datasets, many seeds) run only with --runslow."""

import pytest


# Pattern from the pytest documentation ('Control skipping of tests according to command line option').
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run @pytest.mark.slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs real data or minutes of CPU; --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
