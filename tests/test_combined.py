"""Tests for combined: quantity_only client sizes, then label skew fitted within them."""

import json

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import load_partition, partition_id, read_jsonl
from experiments.seeding import seeds_for
from partitioning.allocations import allocate_to_budgets, label_drift, sinkhorn_scale
from partitioning.quantity import draw_client_sizes
from partitioning.two_stage_method import allocate_client_weighted, combined, fit_client_mixes

LABELS = np.repeat(np.arange(10), 600)  # balanced, N = 6000
IMBALANCED = np.repeat(np.arange(4), [3000, 900, 90, 10])  # Covtype-like skew
BIG = np.repeat(np.arange(10), 5000)  # N = 50,000, balanced like CIFAR-10
FLOOR = 31


def stage_one_sizes(labels, K, beta, seed, floor=FLOOR):
    """The client sizes combined draws for this seed: its first spawned stream."""
    size_rng, _, _ = np.random.default_rng(seed).spawn(3)
    return draw_client_sizes(len(labels), K, beta, floor, size_rng)


def drawn(labels, K, alpha, beta, seed):
    """combined's stage-1 sizes and stage-2 mixes for this seed, before the fit."""
    stats = {}
    combined(labels, K, alpha, beta, FLOOR, np.random.default_rng(seed), stats=stats)
    return stats["client_budgets"], stats["requested_proportions"]


# --- partition --------------------------------------------------------------


@pytest.mark.parametrize("labels", [LABELS, IMBALANCED])
@pytest.mark.parametrize("K", [10, 30, 100])
@pytest.mark.parametrize("alpha", [0.003, 0.1, 1.0, 1000.0])
@pytest.mark.parametrize("beta", [0.05, 1.0, 100.0])
def test_every_sample_once_and_sizes_are_the_stage_one_draw(labels, K, alpha, beta):
    parts = combined(labels, K, alpha, beta, FLOOR, np.random.default_rng(3))
    assert len(parts) == K
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
    assert all(p.dtype == np.int32 and np.all(np.diff(p) > 0) for p in parts)
    assert np.array_equal([len(p) for p in parts], stage_one_sizes(labels, K, beta, 3))


def test_stats_describe_the_draw_and_the_fit():
    stats = {}
    parts = combined(LABELS, 30, 0.3, 0.5, FLOOR, np.random.default_rng(1), stats=stats)
    budgets, proportions, counts = (
        stats["client_budgets"],
        stats["requested_proportions"],
        stats["realized_counts"],
    )
    assert proportions.shape == counts.shape == (30, 10)
    assert np.array_equal(counts.sum(axis=1), budgets)
    assert np.array_equal(counts.sum(axis=0), np.bincount(LABELS))
    assert all(
        np.array_equal(np.bincount(LABELS[p], minlength=10), c) for p, c in zip(parts, counts)
    )
    assert np.array_equal(stats["label_drift"], label_drift(proportions, counts))
    assert isinstance(stats["n_floored_clients"], int)


def test_deterministic_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = combined(LABELS, 30, 0.3, 0.5, FLOOR, np.random.default_rng(5))
    b = combined(LABELS, 30, 0.3, 0.5, FLOOR, np.random.default_rng(5))
    c = combined(LABELS, 30, 0.3, 0.5, FLOOR, np.random.default_rng(6))
    assert np.array_equal(np.random.get_state()[1], before)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_stages_draw_from_independent_streams():
    """Sizes do not depend on alpha, nor the drawn mixes on beta."""
    sizes_low, mixes_a = drawn(LABELS, 30, 0.01, 0.5, 7)
    sizes_high, _ = drawn(LABELS, 30, 100.0, 0.5, 7)
    _, mixes_b = drawn(LABELS, 30, 0.01, 10.0, 7)
    assert np.array_equal(sizes_low, sizes_high)
    assert np.array_equal(mixes_a, mixes_b)


def test_empty_clients_get_no_samples_and_nan_drift():
    stats = {}
    parts = combined(IMBALANCED, 300, 0.05, 0.05, 1, np.random.default_rng(4), stats=stats)
    empty = stats["client_budgets"] == 0
    assert empty.any()  # floor 1 lets a size round down to 0
    assert all(len(parts[k]) == 0 for k in np.flatnonzero(empty))
    assert np.isnan(stats["label_drift"][empty]).all()
    assert not np.isnan(stats["label_drift"][~empty]).any()


# --- the fit ----------------------------------------------------------------


def test_fit_meets_both_margins_with_a_size_scaled_tilt():
    """The fitted mixes are p_kc * exp(w_k * nu_c), normalised, with w_k = b_k / max b.

    With both margins met, that form is the optimality condition for the mixes
    closest to the draws in summed KL divergence, every client counted once.
    """
    budgets, proportions = drawn(LABELS, 30, 1.0, 0.3, 2)
    stock = np.bincount(LABELS)
    fitted, _ = fit_client_mixes(proportions, budgets, stock)
    assert np.allclose(fitted.sum(axis=1), budgets)
    assert np.abs(fitted.sum(axis=0) - stock).sum() <= 1e-3

    log_ratio = np.log(fitted / budgets[:, None]) - np.log(proportions)
    tilt = (log_ratio - log_ratio[:, -1:]) / (budgets / budgets.max())[:, None]
    assert np.allclose(tilt, tilt[0], atol=1e-6)


def test_equal_budgets_give_the_pure_label_scaling():
    """With every w_k = 1 the fit is the Sinkhorn scaling pure_label uses."""
    proportions = np.random.default_rng(0).dirichlet(np.full(10, 0.3), size=30)
    budgets, stock = np.full(30, 200), np.bincount(LABELS)
    fitted, _ = fit_client_mixes(proportions, budgets, stock)
    scaled, _ = sinkhorn_scale(proportions, budgets, stock)
    assert np.abs(fitted - scaled).max() < 1e-2


def test_small_clients_keep_their_draws_better_than_under_sinkhorn():
    """Under strong quantity skew a size-blind correction distorts the small clients."""
    stock_fit, sinkhorn = [], []
    for seed in range(3):
        budgets, proportions = drawn(BIG, 30, 0.3, 0.1, seed)
        _, counts = allocate_client_weighted(BIG, budgets, proportions, np.random.default_rng(0))
        _, counts_sk = allocate_to_budgets(BIG, budgets, proportions, np.random.default_rng(0))
        stock_fit.append(label_drift(proportions, counts).mean())
        sinkhorn.append(label_drift(proportions, counts_sk).mean())
    assert np.mean(stock_fit) < 0.5 * np.mean(sinkhorn)


@pytest.mark.parametrize("alpha", [0.001, 0.01])
def test_fit_converges_at_tiny_alpha(alpha):
    for seed in range(5):
        parts = combined(BIG, 100, alpha, 0.05, FLOOR, np.random.default_rng(seed))
        assert sum(len(p) for p in parts) == len(BIG)


# --- config and runner ------------------------------------------------------


def combined_config(**sections):
    data = to_dict(load_config("config/smoke_combined.yaml"))
    for section, changes in sections.items():
        data[section].update(changes)
    return from_dict(data)


def test_combined_requires_alpha_and_beta():
    with pytest.raises(ValueError, match="requires alpha"):
        combined_config(partition={"alpha": None})
    with pytest.raises(ValueError, match="requires beta"):
        combined_config(partition={"beta": None})


def test_combined_rejects_redraw_settings():
    with pytest.raises(ValueError, match="min_size_policy"):
        combined_config(partition={"min_size_policy": 10})
    with pytest.raises(ValueError, match="max_redraws"):
        combined_config(partition={"max_redraws": 5})


def test_quantity_floor_applies_and_enters_the_partition_id():
    default, explicit = combined_config(), combined_config(partition={"quantity_floor": 40})
    assert explicit.partition.quantity_floor == 40
    pid = lambda c: partition_id(c, seeds_for(c).partition)
    assert pid(explicit) != pid(default)


def test_runner_writes_combined_rows(tmp_path):
    import experiments.run_partition as runner

    config = combined_config(runtime={"out_dir": str(tmp_path)})
    row = runner.generate_partition(config, labels=LABELS)
    assert row["status"] == "generated" and row["mode"] == "combined"
    assert row["quantity_floor"] == 31 and row["n_redraws"] == 0
    assert row["alpha"] == 0.5 and row["beta"] == 1.0
    assert isinstance(row["n_floored_clients"], int)

    stats = {}
    rng = np.random.default_rng(seeds_for(config).partition)
    expected = combined(LABELS, 10, 0.5, 1.0, 31, rng, stats=stats)
    parts = load_partition(tmp_path, row["partition_id"])
    assert all(np.array_equal(a, b) for a, b in zip(parts, expected))
    assert np.allclose(row["label_drift"], stats["label_drift"])
    assert row["mean_label_drift"] == pytest.approx(stats["label_drift"].mean())
    assert row["max_label_drift"] == pytest.approx(stats["label_drift"].max())
    assert read_jsonl(tmp_path / "partitions.jsonl") == [row]

    line = runner.describe(row)
    assert all(key in line for key in ("alpha=", "beta=", "floored=", "drift="))
    assert runner.generate_partition(config, labels=LABELS) is None


def test_drift_fields_keep_rows_valid_json():
    from experiments.run_partition import label_drift_fields

    fields = label_drift_fields({"label_drift": np.array([0.1, np.nan, 0.3])})
    assert fields == {
        "label_drift": [0.1, None, 0.3],
        "mean_label_drift": 0.2,
        "max_label_drift": 0.3,
    }
    json.dumps(fields, allow_nan=False)
    assert label_drift_fields({}) == dict.fromkeys(fields)


@pytest.mark.slow
@pytest.mark.parametrize("K", [10, 30, 100])
def test_real_cifar10_across_alpha_and_beta(K):
    from experiments.config import DatasetConfig
    from experiments.datasets import load_labels

    labels = load_labels(DatasetConfig("cifar10", 10))
    for alpha in (0.01, 0.1, 1.0, 100.0):
        for beta in (0.05, 1.0, 100.0):
            parts = combined(labels, K, alpha, beta, FLOOR, np.random.default_rng(0))
            assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(labels)))
            assert np.array_equal([len(p) for p in parts], stage_one_sizes(labels, K, beta, 0))
