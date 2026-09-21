"""Tests for the partition skew metrics."""

import numpy as np
import pytest

from partitioning.metrics import (
    class_counts, hellinger, pairwise_hellinger, partition_metrics,
    quantity_hd_from_pairwise,
)


def test_hellinger_identical_is_zero():
    p = [0.1, 0.2, 0.7]
    assert hellinger(p, p) == 0.0


def test_hellinger_disjoint_is_one():
    assert hellinger([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_hellinger_is_symmetric():
    p, q = [0.3, 0.5, 0.2], [0.6, 0.1, 0.3]
    assert hellinger(p, q) == pytest.approx(hellinger(q, p))


def test_pairwise_matches_fedartml():
    """Our pairwise scalar must equal FedArtML's, or HD values aren't comparable."""
    fedartml = pytest.importorskip("fedartml.function_base")
    rng = np.random.default_rng(0)
    dists = rng.dirichlet(np.full(10, 0.5), size=8)
    assert pairwise_hellinger(dists) == pytest.approx(
        fedartml.hellinger_distance(dists))


def test_iid_partition_has_near_zero_skew():
    labels = np.tile(np.arange(10), 500)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(labels))
    client_indices = np.array_split(shuffled, 10)

    m = partition_metrics(labels, client_indices, num_classes=10)
    assert m["realized_label_hd"] < 0.1
    assert m["realized_quantity_hd"] == pytest.approx(0.0, abs=1e-9)
    assert m["n_empty_class_slots"] == 0


def test_one_class_per_client_is_maximal_label_skew():
    labels = np.repeat(np.arange(10), 100)
    client_indices = [np.where(labels == c)[0] for c in range(10)]

    m = partition_metrics(labels, client_indices, num_classes=10)
    assert m["realized_label_hd"] == pytest.approx(m["max_per_client_label_deviation"])
    assert m["realized_label_hd_pairwise"] == pytest.approx(1.0)
    assert m["realized_quantity_hd"] == pytest.approx(0.0, abs=1e-9)  # sizes equal
    assert m["n_empty_class_slots"] == 90


def test_quantity_skew_is_independent_of_label_skew():
    """Equal label mix, unequal sizes: label-HD ~0 while quantity-HD is not."""
    labels = np.tile(np.arange(10), 500)
    by_class = [np.where(labels == c)[0] for c in range(10)]
    shares = [0.5, 0.3, 0.1, 0.05, 0.05]
    client_indices, cursor = [], 0
    for share in shares:
        take = int(share * 500)
        client_indices.append(np.concatenate([idx[cursor:cursor + take] for idx in by_class]))
        cursor += take

    m = partition_metrics(labels, client_indices, num_classes=10)
    assert m["realized_label_hd"] == pytest.approx(0.0, abs=1e-9)
    assert m["realized_quantity_hd"] > 0.2


def test_class_counts_shape():
    labels = np.tile(np.arange(4), 25)
    counts = class_counts(labels, np.array_split(np.arange(100), 5), num_classes=4)
    assert counts.shape == (5, 4)
    assert counts.sum() == 100


def test_label_hd_ceiling_for_one_class_clients():
    """Balanced C=10, every client one class: HD = sqrt(1 - 1/sqrt(10)) ~ 0.827."""
    labels = np.repeat(np.arange(10), 100)
    client_indices = [np.where(labels == c)[0] for c in range(10)]
    m = partition_metrics(labels, client_indices, num_classes=10)
    ceiling = np.sqrt(1 - 1 / np.sqrt(10))
    assert ceiling == pytest.approx(0.827, abs=5e-4)
    assert m["realized_label_hd"] == pytest.approx(ceiling)


def test_quantity_hd_ceiling_for_one_full_client():
    """K=30, one client holds everything: HD = sqrt(1 - 1/sqrt(30)) ~ 0.904."""
    labels = np.tile(np.arange(10), 30)
    empty = np.array([], dtype=np.int64)
    client_indices = [np.arange(len(labels))] + [empty] * 29
    m = partition_metrics(labels, client_indices, num_classes=10)
    ceiling = np.sqrt(1 - 1 / np.sqrt(30))
    assert ceiling == pytest.approx(0.904, abs=5e-4)
    assert m["realized_quantity_hd"] == pytest.approx(ceiling)
    assert m["n_empty_clients"] == 29


@pytest.mark.parametrize("K", [2, 10, 30, 100])
def test_quantity_conversion_roundtrips_against_fedartml(K):
    """HD_ours = sqrt(1 - sqrt(1 - (K-1) HD_lib^2)), HD_lib from FedArtML itself."""
    fedartml = pytest.importorskip("fedartml.function_base")
    rng = np.random.default_rng(K)
    for concentration in (0.05, 0.5, 5.0, 500.0):
        sizes = rng.multinomial(50_000, rng.dirichlet(np.full(K, concentration)))
        shares = sizes / sizes.sum()
        lib = fedartml.hellinger_distance([[s] for s in shares])  # create_clients() form
        ours = hellinger(shares, np.full(K, 1 / K))
        assert quantity_hd_from_pairwise(lib, K) == pytest.approx(ours, abs=1e-9)


def test_quantity_pairwise_matches_fedartml_on_a_partition():
    fedartml = pytest.importorskip("fedartml.function_base")
    labels = np.tile(np.arange(10), 500)
    client_indices = np.split(np.arange(5000), [100, 700, 2000, 4000])
    m = partition_metrics(labels, client_indices, num_classes=10)
    shares = [[len(c) / 5000] for c in client_indices]
    assert m["realized_quantity_hd_pairwise"] == pytest.approx(
        fedartml.hellinger_distance(shares))


def test_empty_clients_are_excluded_from_label_hd_but_counted():
    labels = np.repeat(np.arange(10), 100)
    one_class = [np.where(labels == c)[0] for c in range(10)]
    empty = np.array([], dtype=np.int64)
    without = partition_metrics(labels, one_class, num_classes=10)
    with_empty = partition_metrics(labels, one_class + [empty, empty], num_classes=10)

    assert with_empty["n_empty_clients"] == 2
    assert without["n_empty_clients"] == 0
    for key in ("realized_label_hd", "realized_label_hd_pairwise",
                "max_per_client_label_deviation"):
        assert with_empty[key] == pytest.approx(without[key])
    # ...but they do count as quantity skew
    assert with_empty["realized_quantity_hd"] > without["realized_quantity_hd"]
    assert with_empty["min_client_size"] == 0
