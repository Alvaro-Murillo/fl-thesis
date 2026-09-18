"""Tests for the partition skew metrics."""

import numpy as np
import pytest

from partitioning.metrics import (
    class_counts, hellinger, pairwise_hellinger, partition_metrics,
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
