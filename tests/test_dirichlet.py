"""Tests for `natural_per_class`, including the FedArtML-faithful size cap."""

import types

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import partition_id
from experiments.seeding import seeds_for
from partitioning.dirichlet import natural_per_class

LABELS = np.repeat(np.arange(10), 600)  # N = 6000, balanced


def shimmed_fedartml(monkeypatch, rng):
    """FedArtML's module with np.random routed to `rng`, seed() a no-op.

    FedArtML draws shuffle-then-dirichlet per class, as we do, so on a shared
    stream both implementations see the same numbers and must agree exactly.
    """
    module = pytest.importorskip("fedartml.fl_split_as_federated_data")
    fake_random = types.SimpleNamespace(
        seed=lambda *_: None, shuffle=rng.shuffle, dirichlet=rng.dirichlet)

    class NumpyShim:
        random = fake_random

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr(module, "np", NumpyShim())
    return module.SplitAsFederatedData


@pytest.mark.parametrize("alpha", [0.1, 0.5, 5.0])
def test_niid_bench_cap_replicates_fedartml_exactly(monkeypatch, alpha):
    ours = natural_per_class(LABELS, 20, alpha, np.random.default_rng(7),
                             min_size=10, size_cap="niid_bench")
    splitter = shimmed_fedartml(monkeypatch, np.random.default_rng(7))
    _, _, theirs, _ = splitter.dirichlet_method(LABELS, 20, alpha=alpha, random_state=0)
    assert all(np.array_equal(a, np.sort(b)) for a, b in zip(ours, theirs))


def test_uncapped_differs_from_fedartml(monkeypatch):
    ours = natural_per_class(LABELS, 20, 0.1, np.random.default_rng(7),
                             min_size=10, size_cap="none")
    splitter = shimmed_fedartml(monkeypatch, np.random.default_rng(7))
    _, _, theirs, _ = splitter.dirichlet_method(LABELS, 20, alpha=0.1, random_state=0)
    assert not all(np.array_equal(a, np.sort(b)) for a, b in zip(ours, theirs))


@pytest.mark.parametrize("size_cap", ["none", "niid_bench"])
def test_every_sample_assigned_exactly_once(size_cap):
    parts = natural_per_class(LABELS, 30, 0.1, np.random.default_rng(1), size_cap=size_cap)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(LABELS)))


@pytest.mark.parametrize("size_cap", ["none", "niid_bench"])
def test_deterministic_and_global_rng_untouched(size_cap):
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = natural_per_class(LABELS, 30, 0.3, np.random.default_rng(5), size_cap=size_cap)
    b = natural_per_class(LABELS, 30, 0.3, np.random.default_rng(5), size_cap=size_cap)
    assert np.array_equal(np.random.get_state()[1], before)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_cap_compresses_sizes_but_allows_within_class_overshoot():
    """The cap is checked before each class split, so N/K can be exceeded."""
    K, cap = 30, len(LABELS) / 30
    capped_max, uncapped_max, overshoot = [], [], False
    for seed in range(20):
        capped = natural_per_class(LABELS, K, 0.1, np.random.default_rng(seed),
                                   size_cap="niid_bench")
        uncapped = natural_per_class(LABELS, K, 0.1, np.random.default_rng(seed))
        capped_max.append(max(map(len, capped)))
        uncapped_max.append(max(map(len, uncapped)))
        overshoot |= capped_max[-1] > cap
    assert np.mean(capped_max) < np.mean(uncapped_max)
    assert overshoot
    # Bounded overshoot: at most one class's stock beyond the cap.
    assert max(capped_max) < cap + 600


def test_unknown_size_cap_rejected():
    with pytest.raises(ValueError, match="size_cap"):
        natural_per_class(LABELS, 10, 1.0, np.random.default_rng(0), size_cap="fedartml")


def smoke_with(**partition):
    data = to_dict(load_config("config/smoke.yaml"))
    data["partition"].update(partition)
    return from_dict(data)


def test_size_cap_enters_partition_id_only_when_non_default():
    default, explicit = smoke_with(), smoke_with(size_cap="none")
    capped = smoke_with(size_cap="niid_bench")
    pid = lambda c: partition_id(c, seeds_for(c).partition)
    assert pid(default) == pid(explicit) == "30fbc685a3"
    assert pid(capped) != pid(default)


def test_size_cap_validated_in_config():
    with pytest.raises(ValueError, match="size_cap"):
        smoke_with(size_cap="bogus")
    with pytest.raises(ValueError, match="natural_per_class only"):
        smoke_with(mode="combined", beta=1.0, size_cap="niid_bench")


@pytest.mark.parametrize("mode", ["natural_per_class", "natural_per_client"])
def test_redraw_count_is_reported_and_exhaustion_is_typed(mode):
    from partitioning import dirichlet

    partitioner = getattr(dirichlet, mode)
    stats = {}
    partitioner(LABELS, 30, 5.0, np.random.default_rng(0), min_size=10, stats=stats)
    assert stats["n_redraws"] == 0
    stats = {}
    with pytest.raises(dirichlet.MinSizeInfeasible) as err:
        partitioner(LABELS, 30, 0.1, np.random.default_rng(0),
                    min_size=len(LABELS), max_redraws=7, stats=stats)
    assert stats["n_redraws"] == 7 and err.value.max_redraws == 7


def test_max_redraws_does_not_enter_partition_id():
    """A budget only decides whether a draw is found, never which one."""
    pid = lambda c: partition_id(c, seeds_for(c).partition)
    assert pid(smoke_with(max_redraws=5)) == pid(smoke_with()) == "30fbc685a3"
