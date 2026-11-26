"""Calibration curves: alpha -> realized label-HD and beta -> realized quantity-HD.

    python -m partitioning.calibration --dataset cifar10 --K 30 \\
        --mode natural_per_class --size-cap none
    python -m partitioning.calibration --dataset cifar10 --K 30 --mode pure_label
    python -m partitioning.calibration --dataset cifar10 --K 30 --mode quantity_only

Label-skew modes: for one (dataset, K, mode, size_cap, min_size_policy) this
partitions at a dense log-spaced alpha grid (30 values in [0.01, 1000]) x 3
trials and fits realized label-HD. quantity_only: for one (dataset, K,
quantity_floor) it partitions at 30 log-spaced beta values in [0.001, 1000]
x 3 trials and fits realized quantity-HD. `--trials n` uses trials 0..n-1
instead (a new calibration_id). Either way the file records mean
label-HD and quantity-HD per point, the seeds and the git commit, and is
written to calibration/{calibration_id}.json.

A curve is indexed by `parameter` ("alpha" or "beta") and fitted on
`fitted_metric` ("label_hd" or "quantity_hd"). Alpha curves carry neither key
(their file format predates beta curves and is unchanged, as are their IDs);
beta curves hash a different payload.

The trial seeds are the grid's own (`derive_seeds(trial).partition`), so a
calibration point and the grid partition at the same (parameter, K, trial) are
the same partition. The same seeds at every point (common random numbers) keep
the curve smooth.

Inverse: `alpha_for_target` / `beta_for_target` interpolate log(parameter) on
a monotone (non-increasing, pool-adjacent-violators) fit of the mean curve;
pass `labels` to refine by bisection against fresh partitions. Points where
any trial exhausted its redraw budget are kept in the file but left out of the
fit. Target HD levels are not chosen here.

Grid cells give targets instead of parameters: `resolve_targets` fills a null
alpha from `target_label_hd` and the curve `calibration_id`, and a null beta
from `target_quantity_hd` and the curve `quantity_calibration_id`, at grid
expansion. A target of 0 means no skew on that axis (`NO_SKEW`). combined has
no curve of its own. Its sizes come from the same draw as quantity_only's, so
the quantity_only beta curve holds for it exactly. Its label-HD stayed within
0.03 of pure_label's at the same alpha across the whole quantity range
(CIFAR-10, K = 30, alpha 0.15 and 1, beta down to 0.05), so it reads alpha off
the pure_label curve. Every row's |realized - target| <= 0.05 label check
catches a cell where that stops holding. Quantity targets are checked per
level, over 20 seeds (`experiments.quantity_levels`).

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.config import FLOORED_MODES, PartitionMode
from experiments.recording import atomic_write_text, git_provenance, utc_stamp
from experiments.seeding import derive_seeds
from partitioning.pure_label_method import pure_label
from partitioning.natural_methods import MinSizeInfeasible, natural_per_class, natural_per_client
from partitioning.metrics import partition_metrics
from partitioning.quantity_only_method import default_quantity_floor, quantity_only

CALIBRATION_DIR = Path("calibration")
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-2, 3, 30))
DEFAULT_BETAS = tuple(float(b) for b in np.logspace(-3, 3, 30))
DEFAULT_TRIALS = (0, 1, 2)
MODES = ("natural_per_class", "natural_per_client", "pure_label", "quantity_only")
PARAMETER = {
    "natural_per_class": "alpha",
    "natural_per_client": "alpha",
    "pure_label": "alpha",
    "quantity_only": "beta",
}
NEVER_REDRAW = ("pure_label", "quantity_only")
FITTED_METRIC = {"alpha": "label_hd", "beta": "quantity_hd"}
# A target of exactly 0 resolves to this concentration, not to a value read off a
# curve: realized HD is never 0, and the fitted ranges start near 0.01 at K = 30.
# 1000 is the source paper's IID alpha. On CIFAR-10 at K = 30 (20 seeds) it gives
# label-HD 0.010 and quantity-HD 0.011. It is a float on purpose: partition_id
# hashes alpha as JSON, and the RQ1 grid recorded its alpha = 1000 cells as 1000.0,
# so a natural cell at target 0 reuses those partitions.
NO_SKEW = 1000.0
# The mode whose alpha curve a mode's label-HD targets are read from, where not its own.
LABEL_CURVE_MODE = {"combined": "pure_label"}
# (parameter, target field, curve field) for the two axes.
_AXES = (
    ("alpha", "target_label_hd", "calibration_id"),
    ("beta", "target_quantity_hd", "quantity_calibration_id"),
)


def _mode_parameter(mode: str) -> str:
    if mode not in PARAMETER:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return PARAMETER[mode]


def _min_size_policy(mode: str, min_size_policy: int | None) -> int:
    """10 for the natural modes (FedArtML's value); pure_label and quantity_only never redraw."""
    _mode_parameter(mode)
    if mode in NEVER_REDRAW:
        if min_size_policy not in (None, 0):
            raise ValueError(f"{mode} never redraws; min_size_policy must be 0")
        return 0
    return 10 if min_size_policy is None else min_size_policy


def _partition(mode, size_cap, labels, K, value, rng, min_size, max_redraws, stats, floor=None):
    if mode == "natural_per_class":
        return natural_per_class(
            labels,
            K,
            value,
            rng,
            min_size=min_size,
            max_redraws=max_redraws,
            size_cap=size_cap,
            stats=stats,
        )
    if size_cap != "none":
        raise ValueError("size_cap applies to natural_per_class only")
    if mode == "natural_per_client":
        return natural_per_client(
            labels, K, value, rng, min_size=min_size, max_redraws=max_redraws, stats=stats
        )
    if mode == "pure_label":
        stats["n_redraws"] = 0
        return pure_label(labels, K, value, rng, stats=stats)
    if mode == "quantity_only":
        stats["n_redraws"] = 0
        return quantity_only(labels, K, value, floor, rng, stats=stats)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def calibration_id(
    dataset, K, mode, size_cap, min_size_policy, max_redraws, values, seeds, quantity_floor=None
) -> str:
    if _mode_parameter(mode) == "beta":
        payload = ["beta", dataset, K, mode, quantity_floor, [round(b, 12) for b in values]]
    else:
        payload = [
            dataset,
            K,
            mode,
            size_cap,
            min_size_policy,
            max_redraws,
            [round(a, 12) for a in values],
        ]
    payload = json.dumps([*payload, list(seeds)], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def measure_realized_skew(
    labels,
    K,
    mode,
    value,
    seeds,
    size_cap="none",
    min_size_policy=None,
    max_redraws=1000,
    quantity_floor=None,
) -> dict:
    """Realized skew at one alpha (or beta, for quantity_only) across the given seeds."""
    parameter = _mode_parameter(mode)
    min_size_policy = _min_size_policy(mode, min_size_policy)
    num_classes = len(np.unique(labels))
    if parameter == "beta" and quantity_floor is None:
        quantity_floor = default_quantity_floor(num_classes)
    trials = []
    for seed in seeds:
        stats = {}
        try:
            parts = _partition(
                mode,
                size_cap,
                labels,
                K,
                value,
                np.random.default_rng(seed),
                min_size_policy,
                max_redraws,
                stats,
                quantity_floor,
            )
        except MinSizeInfeasible:
            trials.append({"seed": seed, "feasible": False, "n_redraws": stats["n_redraws"]})
            continue
        m = partition_metrics(labels, parts, num_classes)
        trial = {
            "seed": seed,
            "feasible": True,
            "n_redraws": stats["n_redraws"],
            "label_hd": m["realized_label_hd"],
            "quantity_hd": m["realized_quantity_hd"],
            "n_empty_clients": m["n_empty_clients"],
        }
        if parameter == "beta":
            trial["n_floored_clients"] = stats["n_floored_clients"]
            trial["min_client_size"] = m["min_client_size"]
        trials.append(trial)

    ok = [t for t in trials if t["feasible"]]
    point = {
        parameter: float(value),
        "trials": trials,
        "n_feasible": len(ok),
        "all_feasible": len(ok) == len(trials),
    }
    for key in ("label_hd", "quantity_hd"):
        values = np.array([t[key] for t in ok])
        point[f"mean_{key}"] = float(values.mean()) if len(values) else None
        point[f"sd_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return point


def monotone_decreasing(y) -> np.ndarray:
    """Least-squares non-increasing fit (pool adjacent violators)."""
    blocks = []  # [sum, count]
    for value in -np.asarray(y, dtype=float):  # fit increasing on -y
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    return -np.concatenate([np.full(n, s / n) for s, n in blocks])


def compute_curve(
    labels,
    dataset: str,
    K: int,
    mode: str,
    size_cap: str = "none",
    min_size_policy: int | None = None,
    alphas=DEFAULT_ALPHAS,
    trials=DEFAULT_TRIALS,
    max_redraws: int = 1000,
    betas=DEFAULT_BETAS,
    quantity_floor: int | None = None,
) -> dict:
    parameter = _mode_parameter(mode)
    metric = FITTED_METRIC[parameter]
    min_size_policy = _min_size_policy(mode, min_size_policy)
    counts = np.bincount(np.asarray(labels))
    if parameter == "beta" and quantity_floor is None:
        quantity_floor = default_quantity_floor(len(counts))
    values = sorted(float(v) for v in (betas if parameter == "beta" else alphas))
    seeds = [derive_seeds(t).partition for t in trials]
    curve = [
        measure_realized_skew(
            labels, K, mode, v, seeds, size_cap, min_size_policy, max_redraws, quantity_floor
        )
        for v in values
    ]

    fitted = [p for p in curve if p["all_feasible"]]
    fit = monotone_decreasing([p[f"mean_{metric}"] for p in fitted]) if fitted else []
    for p in curve:
        p[f"monotone_{metric}"] = None
    for p, value in zip(fitted, fit):
        p[f"monotone_{metric}"] = float(value)

    cid = calibration_id(
        dataset, K, mode, size_cap, min_size_policy, max_redraws, values, seeds, quantity_floor
    )
    head = {"calibration_id": cid}
    if parameter == "beta":  # alpha files keep their original format
        head |= {"parameter": parameter, "fitted_metric": metric}
    tail = {"quantity_floor": quantity_floor} if parameter == "beta" else {}
    return {
        **head,
        "dataset": dataset,
        "num_classes": int(len(counts)),
        "class_counts": counts.tolist(),
        "K": K,
        "mode": mode,
        "size_cap": size_cap,
        "min_size_policy": min_size_policy,
        "max_redraws": max_redraws,
        **tail,
        "trials": list(trials),
        "seeds": seeds,
        f"{parameter}s": values,
        f"{metric}_range": ([float(min(fit)), float(max(fit))] if len(fit) else None),
        "curve": curve,
        "generated_at": utc_stamp(),
        **git_provenance(),
    }


def curve_parameter(calibration: dict) -> str:
    """ "alpha" or "beta"; a file without the key is an alpha curve."""
    return calibration.get("parameter", "alpha")


def _fit_points(calibration: dict):
    """(increasing fitted HD, matching log parameter) with ties collapsed."""
    parameter = curve_parameter(calibration)
    metric = FITTED_METRIC[parameter]
    points = [
        (p[f"monotone_{metric}"], np.log(p[parameter]))
        for p in calibration["curve"]
        if p[f"monotone_{metric}"] is not None
    ]
    by_hd: dict[float, list[float]] = {}
    for hd, log_value in points:
        by_hd.setdefault(hd, []).append(log_value)
    hds = sorted(by_hd)
    return np.array(hds), np.array([np.mean(by_hd[h]) for h in hds])


def _value_for_target(calibration, parameter, target, labels, tol, max_iter) -> float:
    if curve_parameter(calibration) != parameter:
        raise ValueError(
            f"calibration {calibration['calibration_id']} is indexed by "
            f"{curve_parameter(calibration)}, not {parameter}"
        )
    metric = FITTED_METRIC[parameter]
    hds, log_values = _fit_points(calibration)
    if len(hds) < 2 or not hds[0] <= target <= hds[-1]:
        raise ValueError(
            f"target {target} outside the calibrated {metric} range "
            f"{calibration[f'{metric}_range']}"
        )
    estimate = float(np.exp(np.interp(target, hds, log_values)))
    if labels is None:
        return estimate

    i = int(np.searchsorted(hds, target))
    lo, hi = log_values[min(i, len(hds) - 1)], log_values[max(i - 1, 0)]  # hd high, low
    log_value = np.log(estimate)
    for _ in range(max_iter):
        point = measure_realized_skew(
            labels,
            calibration["K"],
            calibration["mode"],
            np.exp(log_value),
            calibration["seeds"],
            calibration["size_cap"],
            calibration["min_size_policy"],
            calibration["max_redraws"],
            calibration.get("quantity_floor"),
        )
        if not point["all_feasible"]:
            raise ValueError(f"{parameter}={np.exp(log_value):.4g} is min-size infeasible")
        gap = point[f"mean_{metric}"] - target
        if abs(gap) <= tol:
            break
        if gap > 0:  # too skewed -> larger parameter
            lo = log_value
        else:
            hi = log_value
        log_value = (lo + hi) / 2
    return float(np.exp(log_value))


def alpha_for_target(
    calibration: dict, target_label_hd: float, labels=None, tol: float = 0.005, max_iter: int = 30
) -> float:
    """alpha whose mean realized label-HD is `target_label_hd`.

    Without `labels`: monotone interpolation of log(alpha) on the curve.
    With `labels`: then bisection on log(alpha) between the bracketing curve
    points, re-partitioning with the calibration's own seeds, until the mean
    is within `tol` of the target (or max_iter halvings).
    """
    return _value_for_target(calibration, "alpha", target_label_hd, labels, tol, max_iter)


def beta_for_target(
    calibration: dict,
    target_quantity_hd: float,
    labels=None,
    tol: float = 0.005,
    max_iter: int = 30,
) -> float:
    """beta whose mean realized quantity-HD (HD to uniform) is `target_quantity_hd`.

    Same method as `alpha_for_target`, on a quantity_only curve. For a target on
    FedArtML's pairwise scale, convert it first with
    `partitioning.metrics.quantity_hd_from_pairwise`.
    """
    return _value_for_target(calibration, "beta", target_quantity_hd, labels, tol, max_iter)


def save_calibration(calibration: dict, directory: str | Path = CALIBRATION_DIR) -> Path:
    path = Path(directory) / f"{calibration['calibration_id']}.json"
    atomic_write_text(path, json.dumps(calibration, indent=2) + "\n")
    return path


def load_calibration(calibration_id: str, directory: str | Path = CALIBRATION_DIR) -> dict:
    return json.loads((Path(directory) / f"{calibration_id}.json").read_text(encoding="utf-8"))


def resolve_targets(cell: dict, directory: str | Path = CALIBRATION_DIR) -> dict:
    """Fill a grid cell's null alpha and beta from its HD targets; returns the cell.

    `cell` is a config as a plain dict (`experiments.config.to_dict`). An axis
    is resolved when the mode uses its parameter, the parameter is null and
    the target is set. Target 0 gives `NO_SKEW`. Any other target is
    interpolated on the curve the axis's calibration field names
    (`alpha_for_target` / `beta_for_target`), after checking that the curve
    was made for this cell. A parameter set explicitly is kept; its target
    then only feeds the label check, the deviation fields and the level check.
    """
    p = cell["partition"]
    mode = PartitionMode(p["mode"])
    uses = {"alpha": mode is not PartitionMode.QUANTITY_ONLY, "beta": mode in FLOORED_MODES}
    for parameter, target_key, curve_key in _AXES:
        target = p.get(target_key)
        if not uses[parameter] or p.get(parameter) is not None or target is None:
            continue
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise ValueError(f"{target_key} must be a number, got {target!r}")
        if target == 0:
            p[parameter] = NO_SKEW
            continue
        if p.get(curve_key) is None:
            raise ValueError(f"{target_key}={target} needs {curve_key} to resolve {parameter}")
        curve = load_calibration(p[curve_key], directory)
        _check_curve_fits_cell(curve, cell, parameter)
        inverse = alpha_for_target if parameter == "alpha" else beta_for_target
        p[parameter] = inverse(curve, target)
    return cell


def _check_curve_fits_cell(curve: dict, cell: dict, parameter: str) -> None:
    """Raise unless `curve` was computed for this cell's dataset, K and partitioner."""
    p, dataset = cell["partition"], cell["dataset"]
    if parameter == "alpha":
        expected = {
            "mode": LABEL_CURVE_MODE.get(p["mode"], p["mode"]),
            "size_cap": p["size_cap"],
            "min_size_policy": p["min_size_policy"],
        }
    else:
        floor = p["quantity_floor"] or default_quantity_floor(dataset["num_classes"])
        expected = {"mode": "quantity_only", "quantity_floor": floor}
    expected |= {"dataset": dataset["name"], "K": p["K"]}
    wrong = [
        f"{key} {curve.get(key)!r}, cell needs {value!r}"
        for key, value in expected.items()
        if curve.get(key) != value
    ]
    if curve_parameter(curve) != parameter:
        wrong.insert(0, f"indexed by {curve_parameter(curve)}, cell needs {parameter}")
    if wrong:
        raise ValueError(
            f"calibration {curve['calibration_id']} does not fit this cell: " + "; ".join(wrong)
        )


def main() -> None:
    from experiments.config import DatasetConfig
    from experiments.datasets import load_labels

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--size-cap", default="none", choices=("none", "niid_bench"))
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="default 10; pure_label and quantity_only take 0 only",
    )
    parser.add_argument("--max-redraws", type=int, default=1000)
    parser.add_argument(
        "--trials",
        type=int,
        default=len(DEFAULT_TRIALS),
        help="trials 0..n-1 per point (default 3); 0-2 are the grid's own partitions",
    )
    parser.add_argument(
        "--quantity-floor",
        type=int,
        default=None,
        help="quantity_only floor in samples; default 3 * num_classes + 1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CALIBRATION_DIR,
        help="write elsewhere first to keep git_dirty honest across runs",
    )
    args = parser.parse_args()

    labels = load_labels(DatasetConfig(name=args.dataset, num_classes=args.num_classes))
    cal = compute_curve(
        labels,
        args.dataset,
        args.K,
        args.mode,
        args.size_cap,
        args.min_size,
        trials=tuple(range(args.trials)),
        max_redraws=args.max_redraws,
        quantity_floor=args.quantity_floor,
    )
    path = save_calibration(cal, args.out_dir)
    parameter = curve_parameter(cal)
    metric = FITTED_METRIC[parameter]
    print(f"{parameter:>10} {'label_hd':>18} {'quantity_hd':>18} {'monotone':>9} redraws/floored")
    for p in cal["curve"]:
        extra = [t.get("n_floored_clients", t["n_redraws"]) for t in p["trials"]]
        if not p["n_feasible"]:
            print(f"{p[parameter]:>10.4g} {'infeasible':>18} {'':>18} {'':>9} {extra}")
            continue
        sd = lambda k: f"{p[k]:.4f}" if p[k] is not None else "-"
        mono = p[f"monotone_{metric}"]
        mono = f"{mono:.4f}" if mono is not None else "-"
        print(
            f"{p[parameter]:>10.4g} {p['mean_label_hd']:>9.4f} +/- {sd('sd_label_hd'):>6} "
            f"{p['mean_quantity_hd']:>9.4f} +/- {sd('sd_quantity_hd'):>6} {mono:>9} {extra}"
        )
    print(f"{metric} range {cal[f'{metric}_range']}  -> {path}")


if __name__ == "__main__":
    main()
