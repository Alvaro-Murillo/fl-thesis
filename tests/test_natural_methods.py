"""Tests for `natural_per_class`, including the FedArtML-faithful size cap.

`shimmed_fedartml` (conftest.py) routes FedArtML's np.random to a Generator:
FedArtML draws shuffle-then-dirichlet per class, as we do.
"""

import hashlib

import numpy as np
import pytest

from experiments.config import config_hash, from_dict, load_config, to_dict
from experiments.recording import partition_id
from experiments.seeding import seeds_for
from partitioning.natural_methods import natural_per_class

LABELS = np.repeat(np.arange(10), 600)  # N = 6000, balanced


@pytest.mark.parametrize("alpha", [0.1, 0.5, 5.0])
def test_niid_bench_cap_replicates_fedartml_exactly(shimmed_fedartml, alpha):
    ours = natural_per_class(
        LABELS, 20, alpha, np.random.default_rng(7), min_size=10, size_cap="niid_bench"
    )
    splitter = shimmed_fedartml(np.random.default_rng(7))
    _, _, theirs, _ = splitter.dirichlet_method(LABELS, 20, alpha=alpha, random_state=0)
    assert all(np.array_equal(a, np.sort(b)) for a, b in zip(ours, theirs))


def test_uncapped_differs_from_fedartml(shimmed_fedartml):
    ours = natural_per_class(
        LABELS, 20, 0.1, np.random.default_rng(7), min_size=10, size_cap="none"
    )
    splitter = shimmed_fedartml(np.random.default_rng(7))
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
        capped = natural_per_class(
            LABELS, K, 0.1, np.random.default_rng(seed), size_cap="niid_bench"
        )
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
    from partitioning import natural_methods

    partitioner = getattr(natural_methods, mode)
    stats = {}
    partitioner(LABELS, 30, 5.0, np.random.default_rng(0), min_size=10, stats=stats)
    assert stats["n_redraws"] == 0
    stats = {}
    with pytest.raises(natural_methods.MinSizeInfeasible) as err:
        partitioner(
            LABELS,
            30,
            0.1,
            np.random.default_rng(0),
            min_size=len(LABELS),
            max_redraws=7,
            stats=stats,
        )
    assert stats["n_redraws"] == 7 and err.value.max_redraws == 7


def test_max_redraws_does_not_enter_partition_id():
    """A budget only decides whether a draw is found, never which one."""
    pid = lambda c: partition_id(c, seeds_for(c).partition)
    assert pid(smoke_with(max_redraws=5)) == pid(smoke_with()) == "30fbc685a3"


def digest(parts):
    flat = np.concatenate([np.r_[len(p), p] for p in parts]).astype(np.int64)
    return hashlib.sha256(flat.tobytes()).hexdigest()[:16]


def count_matrix(parts, labels=LABELS):
    return np.stack([np.bincount(labels[p], minlength=10) for p in parts])


@pytest.mark.parametrize(
    "size_cap, pinned", [("none", "36cc2501c64ba492"), ("niid_bench", "c5adab67fd604467")]
)
@pytest.mark.parametrize("order", [{}, {"class_order": "index"}])
def test_index_order_partitions_are_unchanged(size_cap, pinned, order):
    """Digests pinned before class_order existed."""
    parts = natural_per_class(
        LABELS, 30, 0.3, np.random.default_rng(11), min_size=10, size_cap=size_cap, **order
    )
    assert digest(parts) == pinned


@pytest.mark.parametrize("size_cap", ["none", "niid_bench"])
@pytest.mark.parametrize("alpha, min_size", [(0.3, 10), (0.3, 50)])
def test_reversed_order_reverses_the_count_matrix_columns_on_balanced_labels(
    size_cap, alpha, min_size
):
    """The k-th class split draws the same shuffle and Dirichlet vector in either order.

    min_size 50 at alpha 0.3 makes some seeds redraw; the redraws match too.
    """
    redraws = []
    for seed in range(8):
        index_stats, reversed_stats = {}, {}
        index = natural_per_class(
            LABELS,
            30,
            alpha,
            np.random.default_rng(seed),
            min_size=min_size,
            size_cap=size_cap,
            stats=index_stats,
        )
        reversed_ = natural_per_class(
            LABELS,
            30,
            alpha,
            np.random.default_rng(seed),
            min_size=min_size,
            size_cap=size_cap,
            stats=reversed_stats,
            class_order="reversed",
        )
        assert np.array_equal(count_matrix(reversed_), count_matrix(index)[:, ::-1])
        assert reversed_stats == index_stats
        assert np.array_equal(np.sort(np.concatenate(reversed_)), np.arange(len(LABELS)))
        redraws.append(index_stats["n_redraws"])
    if min_size == 50:
        assert max(redraws) > 0


def test_reversed_order_changes_which_classes_the_cap_concentrates():
    """Under the cap the last classes split find the most clients masked."""
    index = natural_per_class(
        LABELS, 30, 0.3, np.random.default_rng(3), size_cap="niid_bench", class_order="index"
    )
    reversed_ = natural_per_class(
        LABELS, 30, 0.3, np.random.default_rng(3), size_cap="niid_bench", class_order="reversed"
    )
    assert not np.array_equal(count_matrix(reversed_), count_matrix(index))


def test_unknown_class_order_rejected():
    with pytest.raises(ValueError, match="class_order"):
        natural_per_class(LABELS, 10, 1.0, np.random.default_rng(0), class_order="shuffled")
    with pytest.raises(ValueError, match="class_order"):
        smoke_with(class_order="shuffled")


def test_class_order_enters_partition_id_and_config_hash_only_when_non_default():
    default, explicit = smoke_with(), smoke_with(class_order="index")
    reversed_ = smoke_with(class_order="reversed")
    pid = lambda c: partition_id(c, seeds_for(c).partition)
    assert pid(default) == pid(explicit) == "30fbc685a3"
    assert config_hash(default) == config_hash(explicit) == "acdc3d91"
    assert pid(reversed_) != pid(default)
    assert config_hash(reversed_) != config_hash(default)


@pytest.mark.parametrize(
    "config, change",
    [
        ("smoke.yaml", {"mode": "natural_per_client"}),
        ("smoke_pure_label.yaml", {}),
        ("smoke_quantity_only.yaml", {}),
        ("smoke_combined.yaml", {}),
    ],
)
def test_class_order_is_refused_outside_natural_per_class(config, change):
    data = to_dict(load_config(f"config/{config}"))
    data["partition"].update(change)
    assert from_dict(data).partition.class_order == "index"
    data["partition"]["class_order"] = "reversed"
    with pytest.raises(ValueError, match="class_order applies to natural_per_class only"):
        from_dict(data)


def test_the_partition_runner_passes_class_order_and_records_it(tmp_path):
    import experiments.run_partition as runner

    index_config = smoke_with(size_cap="niid_bench")
    reversed_config = smoke_with(size_cap="niid_bench", class_order="reversed")
    seed = seeds_for(index_config).partition
    assert seeds_for(reversed_config).partition == seed
    _, index = runner.build_partition(index_config, np.random.default_rng(seed), LABELS, {})
    _, reversed_ = runner.build_partition(reversed_config, np.random.default_rng(seed), LABELS, {})
    assert np.array_equal(count_matrix(reversed_), count_matrix(index)[:, ::-1])

    data = to_dict(reversed_config)
    data["runtime"]["out_dir"] = str(tmp_path)
    row = runner.generate_partition(from_dict(data), labels=LABELS)
    assert row["class_order"] == "reversed"
    assert "order=reversed" in runner.describe(row)
