"""Slow tests (real datasets, many seeds) run only with --runslow.

Also holds `shimmed_fedartml`, the shared-RNG-stream harness used to show our
partitioners replicate FedArtML exactly.
"""

import types

import numpy as np
import pytest


@pytest.fixture
def shimmed_fedartml(monkeypatch):
    """Factory: FedArtML's SplitAsFederatedData with np.random routed to `rng`.

    np.random.seed becomes a no-op; shuffle, dirichlet and permutation draw from
    `rng`. FedArtML makes those draws in the same order as our partitioners, so
    on a shared stream both implementations see the same numbers and must agree
    exactly.
    """
    module = pytest.importorskip("fedartml.fl_split_as_federated_data")

    def make(rng):
        fake_random = types.SimpleNamespace(
            seed=lambda *_: None,
            shuffle=rng.shuffle,
            dirichlet=rng.dirichlet,
            permutation=rng.permutation,
        )

        class NumpyShim:
            random = fake_random

            def __getattr__(self, name):
                return getattr(np, name)

        monkeypatch.setattr(module, "np", NumpyShim())
        return module.SplitAsFederatedData

    return make


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
