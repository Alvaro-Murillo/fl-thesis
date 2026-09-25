"""Tests for the calibration curves (alpha -> label-HD, beta -> quantity-HD) and inverses."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.seeding import derive_seeds
from partitioning.calibration import (
    alpha_for_target,
    beta_for_target,
    calibration_id,
    compute_curve,
    curve_parameter,
    evaluate,
    load_calibration,
    monotone_decreasing,
    save_calibration,
)
from partitioning.dirichlet import natural_per_class
from partitioning.metrics import partition_metrics
from partitioning.quantity import quantity_only

LABELS = np.repeat(np.arange(10), 300)
ALPHAS = tuple(np.logspace(-1, 2, 8))
QUANTITY_LABELS = np.repeat(np.arange(10), 1000)  # N = 10,000; K * 31 <= N at K = 10
BETAS = tuple(float(b) for b in np.logspace(-3, 3, 7))


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


COMMITTED_ALPHA_CURVES = ("96fb06debf", "d85a67a9c5", "34b3a4dc67")


@pytest.mark.parametrize("cid", COMMITTED_ALPHA_CURVES)
def test_committed_alpha_curve_ids_are_unchanged(cid):
    """The pre-B3 files keep their IDs and read as alpha / label-HD curves."""
    cal = load_calibration(cid)
    assert "parameter" not in cal and curve_parameter(cal) == "alpha"
    assert cid == calibration_id(
        cal["dataset"],
        cal["K"],
        cal["mode"],
        cal["size_cap"],
        cal["min_size_policy"],
        cal["max_redraws"],
        cal["alphas"],
        cal["seeds"],
    )
    alpha = alpha_for_target(cal, sum(cal["label_hd_range"]) / 2)
    assert cal["alphas"][0] < alpha < cal["alphas"][-1]


def test_every_committed_curve_id_recomputes():
    for path in sorted(Path("calibration").glob("*.json")):
        cal = json.loads(path.read_text(encoding="utf-8"))
        parameter = curve_parameter(cal)
        assert (
            path.stem
            == cal["calibration_id"]
            == calibration_id(
                cal["dataset"],
                cal["K"],
                cal["mode"],
                cal["size_cap"],
                cal["min_size_policy"],
                cal["max_redraws"],
                cal[f"{parameter}s"],
                cal["seeds"],
                cal.get("quantity_floor"),
            )
        )


@pytest.fixture(scope="module")
def beta_curve():
    return compute_curve(
        QUANTITY_LABELS, "synthetic", 10, "quantity_only", betas=BETAS, trials=(0, 1)
    )


def test_beta_curve_format_and_monotone_fit(beta_curve):
    assert beta_curve["parameter"] == "beta" and beta_curve["fitted_metric"] == "quantity_hd"
    assert beta_curve["quantity_floor"] == 31 and beta_curve["min_size_policy"] == 0
    assert beta_curve["betas"] == sorted(BETAS)
    fit = [p["monotone_quantity_hd"] for p in beta_curve["curve"]]
    assert all(np.diff(fit) <= 1e-12) and fit[0] > fit[-1] + 0.4
    trial = beta_curve["curve"][0]["trials"][0]
    assert trial["n_redraws"] == 0 and trial["n_floored_clients"] > 0
    assert "alphas" not in beta_curve and "monotone_label_hd" not in beta_curve["curve"][0]


def test_beta_curve_points_are_the_grid_partitions(beta_curve):
    point = beta_curve["curve"][2]
    parts = quantity_only(
        QUANTITY_LABELS, 10, point["beta"], 31, np.random.default_rng(derive_seeds(1).partition)
    )
    assert point["trials"][1]["quantity_hd"] == pytest.approx(
        partition_metrics(QUANTITY_LABELS, parts, 10)["realized_quantity_hd"]
    )


def test_beta_inverse_and_parameter_guard(beta_curve):
    low, high = beta_curve["quantity_hd_range"]
    target = (low + high) / 2
    beta = beta_for_target(beta_curve, target)
    assert beta_curve["betas"][0] < beta < beta_curve["betas"][-1]
    refined = beta_for_target(beta_curve, target, labels=QUANTITY_LABELS, tol=0.01)
    realized = evaluate(QUANTITY_LABELS, 10, "quantity_only", refined, beta_curve["seeds"])
    assert realized["mean_quantity_hd"] == pytest.approx(target, abs=0.01)
    with pytest.raises(ValueError, match="outside"):
        beta_for_target(beta_curve, high + 0.05)
    with pytest.raises(ValueError, match="indexed by beta"):
        alpha_for_target(beta_curve, target)
    with pytest.raises(ValueError, match="min_size_policy"):
        evaluate(QUANTITY_LABELS, 10, "quantity_only", 1.0, [0], min_size_policy=10)


def test_beta_curve_id_depends_on_floor_and_differs_from_alpha_payload(beta_curve):
    same = calibration_id(
        "synthetic", 10, "quantity_only", "none", 0, 1000, BETAS, beta_curve["seeds"], 31
    )
    assert same == beta_curve["calibration_id"]
    other_floor = calibration_id(
        "synthetic", 10, "quantity_only", "none", 0, 1000, BETAS, beta_curve["seeds"], 40
    )
    as_alpha = calibration_id(
        "synthetic", 10, "natural_per_class", "none", 0, 1000, BETAS, beta_curve["seeds"]
    )
    assert len({same, other_floor, as_alpha}) == 3


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
