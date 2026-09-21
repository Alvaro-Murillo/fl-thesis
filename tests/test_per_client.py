"""Tests for `natural_per_client`, mirroring the natural_per_class ones."""

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import partition_id
from experiments.seeding import seeds_for
from partitioning.dirichlet import natural_per_client
from partitioning.metrics import partition_metrics

LABELS = np.repeat(np.arange(10), 600)                     # balanced, N = 6000
IMBALANCED = np.repeat(np.arange(4), [3000, 900, 90, 10])  # Covtype-like skew


@pytest.mark.parametrize("labels", [LABELS, IMBALANCED])
@pytest.mark.parametrize("alpha", [0.03, 1.0, 100.0])
def test_every_sample_assigned_exactly_once(labels, alpha):
    parts = natural_per_client(labels, 30, alpha, np.random.default_rng(3))
    assert len(parts) == 30
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
    assert all(p.dtype == np.int32 for p in parts)


def test_deterministic_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = natural_per_client(LABELS, 30, 0.3, np.random.default_rng(5))
    b = natural_per_client(LABELS, 30, 0.3, np.random.default_rng(5))
    c = natural_per_client(LABELS, 30, 0.3, np.random.default_rng(6))
    assert np.array_equal(np.random.get_state()[1], before)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_min_size_is_honoured_and_exhaustion_raises():
    parts = natural_per_client(LABELS, 30, 0.1, np.random.default_rng(0), min_size=50)
    assert min(map(len, parts)) >= 50
    with pytest.raises(RuntimeError, match="min_size"):
        natural_per_client(LABELS, 30, 0.1, np.random.default_rng(0),
                           min_size=len(LABELS), max_redraws=3)


@pytest.mark.parametrize("labels", [LABELS, IMBALANCED])
def test_large_alpha_limit_is_iid_against_empirical_prior(labels):
    """Column normalisation sends alpha -> inf to Q_global, even when imbalanced."""
    parts = natural_per_client(labels, 10, 1e6, np.random.default_rng(0))
    m = partition_metrics(labels, parts, len(np.unique(labels)))
    assert m["realized_label_hd"] < 0.02
    assert m["realized_quantity_hd"] < 0.01


def test_quantity_byproduct_grows_as_alpha_falls():
    def mean_quantity_hd(alpha):
        return np.mean([partition_metrics(
            LABELS, natural_per_client(LABELS, 30, alpha, np.random.default_rng(s)),
            10)["realized_quantity_hd"] for s in range(5)])
    low, mid, high = mean_quantity_hd(0.1), mean_quantity_hd(1.0), mean_quantity_hd(100.0)
    assert low > mid > high
    assert low > 0.02


def smoke_with(**partition):
    data = to_dict(load_config("config/smoke.yaml"))
    data["partition"].update(partition)
    return from_dict(data)


def test_mode_is_registered_in_the_runner(monkeypatch):
    import experiments.run_partition as runner

    monkeypatch.setattr(runner, "load_labels", lambda dataset: LABELS)
    config = smoke_with(mode="natural_per_client")
    _, parts = runner.build_partition(config, np.random.default_rng(0))
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(LABELS)))
    per_class = smoke_with()
    assert partition_id(config, seeds_for(config).partition) != \
           partition_id(per_class, seeds_for(per_class).partition)
