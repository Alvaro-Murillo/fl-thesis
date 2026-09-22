"""alpha -> realized label-HD calibration curves (B2 phase 4, input to B6).

    python -m partitioning.calibration --dataset cifar10 --K 30 \\
        --mode natural_per_class --size-cap none

For one (dataset, K, mode, size_cap, min_size_policy) this partitions at a
dense log-spaced alpha grid (30 values in [0.01, 1000]) x 3 trials, records
mean realized label-HD (and quantity-HD) per alpha, and writes
calibration/{calibration_id}.json with the curve, the seeds and the git commit.

The trial seeds are the grid's own (`derive_seeds(trial).partition`), so a
calibration point and the RQ1 partition at the same (alpha, K, trial) are the
same partition. The same seeds at every alpha (common random numbers) keep the
curve smooth.

Inverse: `alpha_for_target` interpolates log(alpha) on a monotone
(non-increasing, pool-adjacent-violators) fit of the mean curve; pass `labels`
to refine by bisection on log(alpha) against fresh partitions. Alphas where any
trial exhausted its redraw budget are kept in the file but left out of the fit.
Target HD levels are not chosen here.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.recording import atomic_write_text, git_provenance, utc_stamp
from experiments.seeding import derive_seeds
from partitioning.dirichlet import MinSizeInfeasible, natural_per_class, natural_per_client
from partitioning.metrics import partition_metrics

CALIBRATION_DIR = Path("calibration")
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-2, 3, 30))
DEFAULT_TRIALS = (0, 1, 2)
MODES = ("natural_per_class", "natural_per_client")


def _partition(mode, size_cap, labels, K, alpha, rng, min_size, max_redraws, stats):
    if mode == "natural_per_class":
        return natural_per_class(
            labels,
            K,
            alpha,
            rng,
            min_size=min_size,
            max_redraws=max_redraws,
            size_cap=size_cap,
            stats=stats,
        )
    if mode == "natural_per_client":
        if size_cap != "none":
            raise ValueError("size_cap applies to natural_per_class only")
        return natural_per_client(
            labels, K, alpha, rng, min_size=min_size, max_redraws=max_redraws, stats=stats
        )
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def calibration_id(dataset, K, mode, size_cap, min_size_policy, max_redraws, alphas, seeds) -> str:
    payload = json.dumps(
        [
            dataset,
            K,
            mode,
            size_cap,
            min_size_policy,
            max_redraws,
            [round(a, 12) for a in alphas],
            list(seeds),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def evaluate(
    labels, K, mode, alpha, seeds, size_cap="none", min_size_policy=10, max_redraws=1000
) -> dict:
    """Realized skew at one alpha across the given seeds."""
    num_classes = len(np.unique(labels))
    trials = []
    for seed in seeds:
        stats = {}
        try:
            parts = _partition(
                mode,
                size_cap,
                labels,
                K,
                alpha,
                np.random.default_rng(seed),
                min_size_policy,
                max_redraws,
                stats,
            )
        except MinSizeInfeasible:
            trials.append({"seed": seed, "feasible": False, "n_redraws": stats["n_redraws"]})
            continue
        m = partition_metrics(labels, parts, num_classes)
        trials.append(
            {
                "seed": seed,
                "feasible": True,
                "n_redraws": stats["n_redraws"],
                "label_hd": m["realized_label_hd"],
                "quantity_hd": m["realized_quantity_hd"],
                "n_empty_clients": m["n_empty_clients"],
            }
        )

    ok = [t for t in trials if t["feasible"]]
    point = {
        "alpha": float(alpha),
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
    min_size_policy: int = 10,
    alphas=DEFAULT_ALPHAS,
    trials=DEFAULT_TRIALS,
    max_redraws: int = 1000,
) -> dict:
    alphas = sorted(float(a) for a in alphas)
    seeds = [derive_seeds(t).partition for t in trials]
    curve = [
        evaluate(labels, K, mode, a, seeds, size_cap, min_size_policy, max_redraws) for a in alphas
    ]

    fitted = [p for p in curve if p["all_feasible"]]
    fit = monotone_decreasing([p["mean_label_hd"] for p in fitted]) if fitted else []
    for p in curve:
        p["monotone_label_hd"] = None
    for p, value in zip(fitted, fit):
        p["monotone_label_hd"] = float(value)

    counts = np.bincount(np.asarray(labels))
    return {
        "calibration_id": calibration_id(
            dataset, K, mode, size_cap, min_size_policy, max_redraws, alphas, seeds
        ),
        "dataset": dataset,
        "num_classes": int(len(counts)),
        "class_counts": counts.tolist(),
        "K": K,
        "mode": mode,
        "size_cap": size_cap,
        "min_size_policy": min_size_policy,
        "max_redraws": max_redraws,
        "trials": list(trials),
        "seeds": seeds,
        "alphas": alphas,
        "label_hd_range": ([float(min(fit)), float(max(fit))] if len(fit) else None),
        "curve": curve,
        "generated_at": utc_stamp(),
        **git_provenance(),
    }


def _fit_points(calibration: dict):
    """(increasing label-HD, matching log alpha) with ties collapsed."""
    points = [
        (p["monotone_label_hd"], np.log(p["alpha"]))
        for p in calibration["curve"]
        if p["monotone_label_hd"] is not None
    ]
    by_hd: dict[float, list[float]] = {}
    for hd, log_alpha in points:
        by_hd.setdefault(hd, []).append(log_alpha)
    hds = sorted(by_hd)
    return np.array(hds), np.array([np.mean(by_hd[h]) for h in hds])


def alpha_for_target(
    calibration: dict, target_label_hd: float, labels=None, tol: float = 0.005, max_iter: int = 30
) -> float:
    """alpha whose mean realized label-HD is `target_label_hd`.

    Without `labels`: monotone interpolation of log(alpha) on the curve.
    With `labels`: then bisection on log(alpha) between the bracketing curve
    points, re-partitioning with the calibration's own seeds, until the mean
    is within `tol` of the target (or max_iter halvings).
    """
    hds, log_alphas = _fit_points(calibration)
    if len(hds) < 2 or not hds[0] <= target_label_hd <= hds[-1]:
        raise ValueError(
            f"target {target_label_hd} outside the calibrated label-HD "
            f"range {calibration['label_hd_range']}"
        )
    estimate = float(np.exp(np.interp(target_label_hd, hds, log_alphas)))
    if labels is None:
        return estimate

    i = int(np.searchsorted(hds, target_label_hd))
    lo, hi = log_alphas[min(i, len(hds) - 1)], log_alphas[max(i - 1, 0)]  # hd high, low
    log_alpha = np.log(estimate)
    for _ in range(max_iter):
        point = evaluate(
            labels,
            calibration["K"],
            calibration["mode"],
            np.exp(log_alpha),
            calibration["seeds"],
            calibration["size_cap"],
            calibration["min_size_policy"],
            calibration["max_redraws"],
        )
        if not point["all_feasible"]:
            raise ValueError(f"alpha={np.exp(log_alpha):.4g} is min-size infeasible")
        gap = point["mean_label_hd"] - target_label_hd
        if abs(gap) <= tol:
            break
        if gap > 0:  # too skewed -> larger alpha
            lo = log_alpha
        else:
            hi = log_alpha
        log_alpha = (lo + hi) / 2
    return float(np.exp(log_alpha))


def save_calibration(calibration: dict, directory: str | Path = CALIBRATION_DIR) -> Path:
    path = Path(directory) / f"{calibration['calibration_id']}.json"
    atomic_write_text(path, json.dumps(calibration, indent=2) + "\n")
    return path


def load_calibration(calibration_id: str, directory: str | Path = CALIBRATION_DIR) -> dict:
    return json.loads((Path(directory) / f"{calibration_id}.json").read_text(encoding="utf-8"))


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
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-redraws", type=int, default=1000)
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
        max_redraws=args.max_redraws,
    )
    path = save_calibration(cal, args.out_dir)
    print(f"{'alpha':>10} {'label_hd':>18} {'monotone':>9} {'quantity_hd':>18} redraws")
    for p in cal["curve"]:
        redraws = [t["n_redraws"] for t in p["trials"]]
        if not p["n_feasible"]:
            print(f"{p['alpha']:>10.4g} {'infeasible':>18} {'':>9} {'':>18} {redraws}")
            continue
        sd = lambda k: f"{p[k]:.4f}" if p[k] is not None else "-"
        mono = f"{p['monotone_label_hd']:.4f}" if p["monotone_label_hd"] is not None else "-"
        print(
            f"{p['alpha']:>10.4g} {p['mean_label_hd']:>9.4f} +/- {sd('sd_label_hd'):>6} "
            f"{mono:>9} {p['mean_quantity_hd']:>9.4f} +/- {sd('sd_quantity_hd'):>6} {redraws}"
        )
    print(f"label-HD range {cal['label_hd_range']}  -> {path}")


if __name__ == "__main__":
    main()
