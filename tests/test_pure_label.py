"""Tests for pure_label: equal client sizes, Dirichlet label mixes, matrix-scaled counts."""

import itertools

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import load_partition, partition_id
from experiments.seeding import seeds_for
from partitioning.allocations import (
    allocate_to_budgets,
    equal_budgets,
    label_drift,
    pure_label,
    round_to_margins,
    sinkhorn_scale,
)
from partitioning.calibration import evaluate
from partitioning.metrics import partition_metrics

LABELS = np.repeat(np.arange(10), 600)  # balanced, N = 6000
IMBALANCED = np.repeat(np.arange(4), [3000, 900, 90, 10])  # Covtype-like skew


def plain_sinkhorn(W, rows, cols, iters=5000):
    X = np.asarray(W, dtype=float).copy()
    for _ in range(iters):
        X *= (rows / X.sum(axis=1))[:, None]
        X *= (cols / X.sum(axis=0))[None, :]
    return X


# --- partition --------------------------------------------------------------


@pytest.mark.parametrize("labels", [LABELS, IMBALANCED])
@pytest.mark.parametrize("K", [10, 30, 100])
@pytest.mark.parametrize("alpha", [0.003, 0.1, 1.0, 1000.0])
def test_every_sample_once_and_sizes_equal(labels, K, alpha):
    parts = pure_label(labels, K, alpha, np.random.default_rng(3))
    assert len(parts) == K
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
    assert all(p.dtype == np.int32 and np.all(np.diff(p) > 0) for p in parts)
    sizes = np.array([len(p) for p in parts])
    assert set(sizes) <= {len(labels) // K, len(labels) // K + 1}


def test_deterministic_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = pure_label(LABELS, 30, 0.3, np.random.default_rng(5))
    b = pure_label(LABELS, 30, 0.3, np.random.default_rng(5))
    c = pure_label(LABELS, 30, 0.3, np.random.default_rng(6))
    assert np.array_equal(np.random.get_state()[1], before)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


@pytest.mark.parametrize("labels", [LABELS, IMBALANCED])
def test_no_quantity_skew_and_iid_limit(labels):
    C = len(np.unique(labels))
    m = partition_metrics(labels, pure_label(labels, 10, 1e6, np.random.default_rng(0)), C)
    assert m["realized_label_hd"] < 0.02
    assert m["realized_quantity_hd"] < 1e-3


def test_label_hd_falls_as_alpha_rises():
    def mean_label_hd(alpha):
        return np.mean(
            [
                partition_metrics(
                    LABELS, pure_label(LABELS, 30, alpha, np.random.default_rng(s)), 10
                )["realized_label_hd"]
                for s in range(3)
            ]
        )

    low, mid, high = mean_label_hd(0.03), mean_label_hd(1.0), mean_label_hd(100.0)
    assert low > mid > high


def test_stats_record_draw_counts_and_drift():
    stats = {}
    parts = pure_label(IMBALANCED, 20, 0.5, np.random.default_rng(1), stats=stats)
    counts = stats["realized_counts"]
    assert stats["requested_proportions"].shape == counts.shape == (20, 4)
    assert np.array_equal(counts.sum(axis=1), [len(p) for p in parts])
    assert np.array_equal(counts.sum(axis=0), np.bincount(IMBALANCED))
    for k, idx in enumerate(parts):
        assert np.array_equal(np.bincount(IMBALANCED[idx], minlength=4), counts[k])
    assert np.allclose(
        stats["label_drift"], label_drift(stats["requested_proportions"], counts), equal_nan=True
    )


def test_identical_requests_get_identical_counts():
    """The correction is shared, not left to whichever client is filled last."""
    rng = np.random.default_rng(0)
    one = rng.dirichlet(np.ones(10))
    proportions = np.vstack([np.tile(one, (5, 1)), rng.dirichlet(np.ones(10), size=5)])
    budgets = equal_budgets(len(LABELS), 10, rng)
    _, counts = allocate_to_budgets(LABELS, budgets, proportions, rng)
    assert np.abs(counts[:5] - counts[0]).max() <= 1


def test_feasible_request_is_met_to_within_rounding():
    """If the drawn mixes already fit the stock exactly, no client is corrected."""
    budgets = np.full(10, 600)
    rng = np.random.default_rng(2)
    proportions = plain_sinkhorn(rng.dirichlet(np.ones(10), size=10), budgets, np.full(10, 600))
    proportions /= proportions.sum(axis=1, keepdims=True)
    _, counts = allocate_to_budgets(LABELS, budgets, proportions, rng)
    assert np.abs(counts - budgets[:, None] * proportions).max() < 1


def test_zero_budget_client_gets_nothing():
    rng = np.random.default_rng(0)
    budgets = np.array([3000, 0, 3000])
    parts, counts = allocate_to_budgets(LABELS, budgets, rng.dirichlet(np.ones(10), 3), rng)
    assert len(parts[1]) == 0 and not counts[1].any()


def test_allocate_rejects_bad_input():
    rng = np.random.default_rng(0)
    p = np.full((10, 10), 0.1)
    with pytest.raises(ValueError, match="sum to"):
        allocate_to_budgets(LABELS, np.full(10, 500), p, rng)
    with pytest.raises(ValueError, match="shape"):
        allocate_to_budgets(LABELS, np.full(10, 600), p[:, :9], rng)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_to_budgets(LABELS, np.full(10, 600), -p, rng)
    with pytest.raises(ValueError, match="alpha"):
        pure_label(LABELS, 10, 0.0, rng)
    with pytest.raises(ValueError, match="K <= N"):
        equal_budgets(5, 10, rng)


# --- scaling and rounding ---------------------------------------------------


def test_scaling_matches_plain_sinkhorn():
    rng = np.random.default_rng(4)
    W = rng.dirichlet(np.ones(6), size=8)
    rows, cols = np.full(8, 75.0), rng.multinomial(600, np.full(6, 1 / 6)).astype(float)
    X, steps = sinkhorn_scale(W, rows, cols)
    assert steps > 0
    assert np.allclose(X.sum(axis=1), rows, atol=1e-3) and np.allclose(X.sum(axis=0), cols)
    assert np.abs(X - plain_sinkhorn(W, rows, cols)).max() < 1e-3


def test_scaling_ignores_row_scale():
    rng = np.random.default_rng(5)
    W = rng.dirichlet(np.ones(5), size=6)
    rows, cols = np.full(6, 50.0), np.full(5, 60.0)
    X, _ = sinkhorn_scale(W, rows, cols)
    X_scaled, _ = sinkhorn_scale(W * np.arange(1, 7)[:, None], rows, cols)
    assert np.abs(X - X_scaled).max() < 1e-3


def test_scaling_fills_a_class_nobody_drew():
    """Exact zeros are underflowed draws, not closed cells, so the margins stay reachable."""
    rng = np.random.default_rng(5)
    W = rng.dirichlet(np.ones(5), size=6)
    W[:, 2] = 0.0
    X, _ = sinkhorn_scale(W, np.full(6, 50.0), np.full(5, 60.0))
    assert np.isclose(X[:, 2].sum(), 60.0, atol=1e-3)


@pytest.mark.parametrize("K", [10, 30, 100])
def test_scaling_converges_at_tiny_alpha(K):
    """The regime where the alternating Sinkhorn updates stall."""
    stock = np.full(10, 5000)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        W = rng.dirichlet(np.full(10, 0.003), size=K)
        budgets = equal_budgets(50_000, K, rng)
        X, steps = sinkhorn_scale(W, budgets, stock)
        assert steps < 1000
        assert np.abs(X.sum(axis=1) - budgets).sum() + np.abs(X.sum(axis=0) - stock).sum() <= 1e-3


def test_rounding_is_exact_and_l1_closest():
    rng = np.random.default_rng(6)
    for _ in range(20):
        W = rng.dirichlet(np.ones(3), size=3)
        rows = rng.integers(2, 6, size=3)
        cols = rng.multinomial(rows.sum(), np.full(3, 1 / 3))
        if (cols == 0).any():
            continue
        X, _ = sinkhorn_scale(W, rows, cols, tol=1e-9)
        Y = round_to_margins(X, rows, cols)
        assert np.array_equal(Y.sum(axis=1), rows) and np.array_equal(Y.sum(axis=0), cols)
        assert np.all((Y == np.floor(X)) | (Y == np.floor(X) + 1))
        # Brute force over every floor/ceil choice with the right margins.
        base = np.floor(X).astype(int)
        best = min(
            np.abs(base + np.reshape(up, (3, 3)) - X).sum()
            for up in itertools.product((0, 1), repeat=9)
            if np.array_equal((base + np.reshape(up, (3, 3))).sum(axis=1), rows)
            and np.array_equal((base + np.reshape(up, (3, 3))).sum(axis=0), cols)
        )
        assert np.isclose(np.abs(Y - X).sum(), best)


def test_rounding_an_integer_matrix_returns_it():
    X = np.array([[2.0, 3.0], [1.0, 0.0]])
    assert np.array_equal(round_to_margins(X, [5, 1], [3, 3]), X.astype(int))


# --- config, runner, calibration --------------------------------------------


def pure_config(**sections):
    data = to_dict(load_config("config/smoke_pure_label.yaml"))
    for section, changes in sections.items():
        data[section].update(changes)
    return from_dict(data)


def test_pure_label_requires_alpha_and_rejects_beta():
    with pytest.raises(ValueError, match="requires alpha"):
        pure_config(partition={"alpha": None})
    with pytest.raises(ValueError, match="does not use beta"):
        pure_config(partition={"beta": 0.5})


def test_pure_label_rejects_redraw_settings():
    with pytest.raises(ValueError, match="min_size_policy"):
        pure_config(partition={"min_size_policy": 10})
    with pytest.raises(ValueError, match="max_redraws"):
        pure_config(partition={"max_redraws": 5})


def test_runner_writes_pure_label_rows(tmp_path):
    import experiments.run_partition as runner

    config = pure_config(runtime={"out_dir": str(tmp_path)})
    row = runner.generate_partition(config, labels=LABELS)
    assert row["status"] == "generated" and row["mode"] == "pure_label"
    assert row["n_redraws"] == 0 and row["n_floored_clients"] is None
    assert row["alpha"] == 0.5 and row["beta"] is None
    assert row["max_client_size"] - row["min_client_size"] <= 1

    parts = load_partition(tmp_path, row["partition_id"])
    expected = pure_label(LABELS, 10, 0.5, np.random.default_rng(seeds_for(config).partition))
    assert all(np.array_equal(a, b) for a, b in zip(parts, expected))
    assert runner.generate_partition(config, labels=LABELS) is None

    natural = from_dict(to_dict(load_config("config/smoke.yaml")))
    assert row["partition_id"] != partition_id(natural, seeds_for(natural).partition)


def test_calibration_evaluates_pure_label():
    point = evaluate(LABELS, 10, "pure_label", 1.0, [0, 1])
    assert point["all_feasible"] and point["mean_quantity_hd"] < 1e-3
    assert all(t["n_redraws"] == 0 for t in point["trials"])
    with pytest.raises(ValueError, match="min_size_policy"):
        evaluate(LABELS, 10, "pure_label", 1.0, [0], min_size_policy=10)


@pytest.mark.slow
@pytest.mark.parametrize("K", [10, 30, 100])
def test_real_cifar10_across_alpha(K):
    from experiments.config import DatasetConfig
    from experiments.datasets import load_labels

    labels = load_labels(DatasetConfig("cifar10", 10))
    for alpha in (0.01, 0.1, 1.0, 100.0):
        stats = {}
        parts = pure_label(labels, K, alpha, np.random.default_rng(0), stats=stats)
        assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
        assert np.ptp([len(p) for p in parts]) <= 1
