"""Tests for the alpha -> label-HD calibration curve and its inverse."""

import numpy as np
import pytest

from experiments.seeding import derive_seeds
from partitioning.calibration import (
    alpha_for_target,
    compute_curve,
    evaluate,
    load_calibration,
    monotone_decreasing,
    save_calibration,
)
from partitioning.dirichlet import natural_per_class
from partitioning.metrics import partition_metrics

LABELS = np.repeat(np.arange(10), 300)
ALPHAS = tuple(np.logspace(-1, 2, 8))


@pytest.fixture(
    scope="module",
    params=[
        ("natural_per_class", "none"),
        ("natural_per_class", "niid_bench"),
        ("natural_per_client", "none"),
    ],
)
def curve(request):
    mode, size_cap = request.param
    return compute_curve(
        LABELS, "synthetic", 10, mode, size_cap, min_size_policy=5, alphas=ALPHAS, trials=(0, 1)
    )


def test_monotone_decreasing_pools_violators():
    assert np.allclose(monotone_decreasing([5, 3, 4, 1]), [5, 3.5, 3.5, 1])
    y = np.random.default_rng(0).random(20)
    fit = monotone_decreasing(y)
    assert np.all(np.diff(fit) <= 1e-12)
    assert np.isclose(fit.sum(), y.sum())  # PAVA preserves the mean


def test_curve_records_seeds_provenance_and_decreases(curve):
    assert curve["seeds"] == [derive_seeds(t).partition for t in (0, 1)]
    assert {"git_commit", "calibration_id", "alphas", "curve"} <= set(curve)
    fit = [p["monotone_label_hd"] for p in curve["curve"]]
    assert all(np.diff(fit) <= 1e-12)
    assert fit[0] > fit[-1] + 0.2


def test_curve_points_are_the_grid_partitions(curve):
    """Same (alpha, K, trial) as a grid cell -> the same realized label-HD."""
    if curve["mode"] != "natural_per_class":
        pytest.skip("checked once, on the per-class mode")
    point = curve["curve"][3]
    parts = natural_per_class(
        LABELS,
        10,
        point["alpha"],
        np.random.default_rng(derive_seeds(1).partition),
        min_size=5,
        max_redraws=1000,
        size_cap=curve["size_cap"],
    )
    assert point["trials"][1]["label_hd"] == pytest.approx(
        partition_metrics(LABELS, parts, 10)["realized_label_hd"]
    )


def test_inverse_interpolates_within_range_and_refuses_outside(curve):
    low, high = curve["label_hd_range"]
    target = (low + high) / 2
    alpha = alpha_for_target(curve, target)
    assert curve["alphas"][0] < alpha < curve["alphas"][-1]
    with pytest.raises(ValueError, match="outside"):
        alpha_for_target(curve, high + 0.05)


def test_bisection_refinement_hits_the_target(curve):
    target = sum(curve["label_hd_range"]) / 2
    alpha = alpha_for_target(curve, target, labels=LABELS, tol=0.01)
    realized = evaluate(
        LABELS, 10, curve["mode"], alpha, curve["seeds"], curve["size_cap"], min_size_policy=5
    )["mean_label_hd"]
    assert realized == pytest.approx(target, abs=0.01)


def test_infeasible_alphas_are_kept_but_not_fitted():
    cal = compute_curve(
        LABELS,
        "synthetic",
        10,
        "natural_per_class",
        min_size_policy=150,
        alphas=(0.05, 100.0),
        trials=(0,),
        max_redraws=5,
    )
    small, large = cal["curve"]
    assert not small["all_feasible"] and small["monotone_label_hd"] is None
    assert small["trials"][0]["n_redraws"] == 5
    assert large["all_feasible"] and large["monotone_label_hd"] is not None


def test_save_load_roundtrip_and_id_depends_on_settings(tmp_path, curve):
    path = save_calibration(curve, tmp_path)
    assert path.name == f"{curve['calibration_id']}.json"
    assert load_calibration(curve["calibration_id"], tmp_path) == curve
    other = compute_curve(
        LABELS,
        "synthetic",
        10,
        curve["mode"],
        curve["size_cap"],
        min_size_policy=5,
        alphas=ALPHAS,
        trials=(0, 1, 2),
    )
    assert other["calibration_id"] != curve["calibration_id"]
