"""Tests for the calibration curves (alpha -> label-HD, beta -> quantity-HD) and inverses."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.seeding import derive_seeds
from partitioning.calibration import (
    NO_SKEW,
    alpha_for_target,
    beta_for_target,
    calibration_id,
    compute_curve,
    curve_parameter,
    evaluate,
    load_calibration,
    monotone_decreasing,
    resolve_targets,
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
    """The committed alpha curves keep their IDs and read as alpha / label-HD curves."""
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


# --- targets -> parameters ----------------------------------------------------


@pytest.fixture(scope="module")
def target_curves(tmp_path_factory):
    """(directory, label curve, quantity curve), both for CIFAR-10-named cells at K = 10."""
    directory = tmp_path_factory.mktemp("calibration")
    label = compute_curve(LABELS, "cifar10", 10, "pure_label", alphas=ALPHAS, trials=(0, 1))
    quantity = compute_curve(
        QUANTITY_LABELS, "cifar10", 10, "quantity_only", betas=BETAS, trials=(0, 1)
    )
    natural = {**label, "mode": "natural_per_class", "calibration_id": "natural000"}
    for curve in (label, quantity, natural):
        save_calibration(curve, directory)
    return directory, label, quantity


def combined_cell(**partition):
    cell = to_dict(load_config("config/smoke_combined.yaml"))  # K = 10
    cell["partition"].update({"alpha": None, "beta": None, **partition})
    return cell


def mid(curve, metric):
    low, high = curve[f"{metric}_range"]
    return (low + high) / 2


def test_targets_resolve_on_both_curves(target_curves):
    directory, label, quantity = target_curves
    t_label, t_quantity = mid(label, "label_hd"), mid(quantity, "quantity_hd")
    cell = combined_cell(
        target_label_hd=t_label,
        target_quantity_hd=t_quantity,
        calibration_id=label["calibration_id"],
        quantity_calibration_id=quantity["calibration_id"],
    )
    p = from_dict(resolve_targets(cell, directory)).partition
    assert p.alpha == alpha_for_target(label, t_label)
    assert p.beta == beta_for_target(quantity, t_quantity)


def test_target_zero_is_no_skew_and_needs_no_curve():
    cell = resolve_targets(combined_cell(target_label_hd=0.0, target_quantity_hd=0.0))
    assert cell["partition"]["alpha"] == cell["partition"]["beta"] == NO_SKEW


def test_explicit_parameters_are_kept():
    cell = combined_cell(alpha=0.5, beta=2.0, target_label_hd=0.3, calibration_id="absent")
    cell["partition"].update(target_quantity_hd=0.3, quantity_calibration_id="absent")
    p = resolve_targets(cell)["partition"]  # never opens calibration/absent.json
    assert (p["alpha"], p["beta"]) == (0.5, 2.0)


def test_unused_axes_are_left_alone(target_curves):
    directory, label, _ = target_curves
    cell = to_dict(load_config("config/smoke_pure_label.yaml"))
    cell["partition"].update(alpha=None, target_label_hd=0.0, target_quantity_hd=0.3)
    p = from_dict(resolve_targets(cell, directory)).partition
    assert p.alpha == NO_SKEW and p.beta is None


def test_resolution_refuses_a_missing_or_mismatched_curve(target_curves):
    directory, label, quantity = target_curves
    t = mid(label, "label_hd")
    with pytest.raises(ValueError, match="needs calibration_id"):
        resolve_targets(combined_cell(target_label_hd=t), directory)
    swapped = combined_cell(target_quantity_hd=t, quantity_calibration_id=label["calibration_id"])
    with pytest.raises(ValueError, match="indexed by alpha"):
        resolve_targets(swapped, directory)
    natural = combined_cell(target_label_hd=t, calibration_id="natural000")
    with pytest.raises(ValueError, match="mode 'natural_per_class', cell needs 'pure_label'"):
        resolve_targets(natural, directory)
    wrong_K = combined_cell(target_label_hd=t, calibration_id=label["calibration_id"], K=30)
    with pytest.raises(ValueError, match="K 10, cell needs 30"):
        resolve_targets(wrong_K, directory)
    too_far = combined_cell(target_label_hd=0.99, calibration_id=label["calibration_id"])
    with pytest.raises(ValueError, match="outside"):
        resolve_targets(too_far, directory)


def test_grid_expansion_resolves_targets(tmp_path, target_curves):
    from experiments.grid import expand

    directory, label, quantity = target_curves
    t_label, t_quantity = mid(label, "label_hd"), mid(quantity, "quantity_hd")
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "base: config/smoke_combined.yaml\n"
        "constant:\n"
        "  partition.alpha: null\n"
        "  partition.beta: null\n"
        f"  partition.calibration_id: {label['calibration_id']}\n"
        f"  partition.quantity_calibration_id: {quantity['calibration_id']}\n"
        "sweep:\n"
        f"  partition.target_label_hd: [0.0, {t_label}]\n"
        f"  partition.target_quantity_hd: [0.0, {t_quantity}]\n",
        encoding="utf-8",
    )
    configs = expand(grid, calibration_dir=directory)
    resolved = {(c.partition.alpha, c.partition.beta) for c in configs}
    alpha, beta = alpha_for_target(label, t_label), beta_for_target(quantity, t_quantity)
    assert resolved == {(NO_SKEW, NO_SKEW), (NO_SKEW, beta), (alpha, NO_SKEW), (alpha, beta)}


@pytest.mark.slow
def test_combined_hits_label_targets_off_the_pure_label_curve(tmp_path):
    """The pure_label alpha curve stands in for combined's across the quantity range."""
    from experiments.config import DatasetConfig
    from experiments.datasets import load_labels
    from experiments.run_partition import build_partition
    from experiments.seeding import generator, seeds_for
    from shutil import copy

    labels = load_labels(DatasetConfig("cifar10", 10))
    label = compute_curve(labels, "cifar10", 30, "pure_label")
    save_calibration(label, tmp_path)
    copy("calibration/8f995f9546.json", tmp_path)  # quantity_only beta curve, K = 30, 20 trials
    for t_label in (0.25, 0.5, 0.75):
        for t_quantity in (0.0, 0.2, 0.6):
            cell = combined_cell(
                K=30,
                target_label_hd=t_label,
                target_quantity_hd=t_quantity,
                calibration_id=label["calibration_id"],
                quantity_calibration_id="8f995f9546",
            )
            realized = []
            for trial in (0, 1, 2):
                cell["runtime"]["trial"] = trial
                config = from_dict(resolve_targets(cell, tmp_path))
                _, parts = build_partition(config, generator(seeds_for(config).partition), labels)
                realized.append(partition_metrics(labels, parts, 10)["realized_label_hd"])
            assert np.mean(realized) == pytest.approx(t_label, abs=0.05)
