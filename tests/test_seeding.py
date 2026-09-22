"""Tests for the A4.2 seeding module."""

import numpy as np
import pytest

from experiments.config import load_config, to_dict, from_dict
from experiments.seeding import Seeds, derive_seeds, generator, seeds_for


def fake_partition(seed, K=10, num_classes=10):
    """Stand-in for a Dirichlet partitioner until B2 lands."""
    return generator(seed).dirichlet(np.full(num_classes, 0.5), size=K)


def test_seed_reproducibility():
    assert derive_seeds(0) == derive_seeds(0)
    assert np.array_equal(
        fake_partition(derive_seeds(0).partition), fake_partition(derive_seeds(0).partition)
    )


def test_different_trial_differs():
    a, b = derive_seeds(0), derive_seeds(1)
    assert {a.partition, a.init, a.train}.isdisjoint({b.partition, b.init, b.train})
    assert not np.array_equal(fake_partition(a.partition), fake_partition(b.partition))


def test_streams_are_independent():
    seeds = derive_seeds(0)
    assert len({seeds.partition, seeds.init, seeds.train}) == 3


def test_seeds_are_uint32():
    for trial in range(3):
        seeds = derive_seeds(trial)
        assert all(0 <= s < 2**32 for s in (seeds.partition, seeds.init, seeds.train))


def test_overrides_are_honoured():
    assert derive_seeds(0, partition=7).partition == 7
    derived = derive_seeds(0)
    overridden = derive_seeds(0, init=7)
    assert overridden.init == 7
    assert overridden.partition == derived.partition


def test_seeds_for_config(tmp_path):
    config = load_config("config/base.yaml")
    assert seeds_for(config) == derive_seeds(config.runtime.trial)

    data = to_dict(config)
    data["runtime"]["partition_seed"] = 123
    assert seeds_for(from_dict(data)).partition == 123


def test_partition_seed_ignores_training_fields():
    """Training knobs must not move the partition stream (DoD item 3)."""
    data = to_dict(load_config("config/base.yaml"))
    data["federated"]["rounds"] += 1
    assert (
        seeds_for(from_dict(data)).partition == seeds_for(load_config("config/base.yaml")).partition
    )


def test_global_numpy_state_untouched():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    fake_partition(derive_seeds(0).partition)
    assert np.array_equal(np.random.get_state()[1], before)


def test_rejects_out_of_range_override():
    with pytest.raises(ValueError, match="uint32"):
        derive_seeds(0, partition=2**32)
