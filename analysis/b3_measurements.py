"""B3 phase 2: what deliberate quantity skew reaches (inputs to E1 / RQ1 and B6).

    python -m analysis.b3_measurements [--seeds 20]

Sources, all CIFAR-10 (N = 50,000, C = 10), quantity_only, floor 31:

    grid     the b3_quantity grid rows in results/partitions.jsonl
             (30 betas x K in {10, 30, 100} x 3 trials; run it first)
    curve    the committed beta calibration curve (K = 30, 3 trials)
    seeds    fresh partitions at every curve beta, K = 30, trials 0..n-1
             (trials 0-2 are the grid's own partitions)

Prints the tables for the B3 report and writes, under results/analysis/:

    b3_k30_seeds.csv        one row per (beta, trial) of the seed sweep
    b3_targets.csv          beta for each target quantity-HD
    b3_scatter_clients.csv  per-client (size, label-HD) behind the scatter
    b3_label_hd_vs_size.png that scatter

Measurements only: nothing here chooses target levels or corrects anything.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import DatasetConfig
from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.recording import (
    atomic_write_text,
    git_provenance,
    partition_id,
    read_jsonl,
    utc_stamp,
)
from experiments.seeding import derive_seeds, seeds_for
from partitioning.calibration import (
    DEFAULT_BETAS,
    beta_for_target,
    evaluate,
    load_calibration,
)
from partitioning.metrics import (
    class_counts,
    hellinger,
    max_quantity_hd,
    partition_metrics,
    quantity_hd_from_pairwise,
)
from partitioning.quantity import default_quantity_floor, min_size_bound, quantity_only

GRID = Path("config/grids/b3_quantity.yaml")
CURVE_ID = "f09e0cfdf0"
K = 30
N_SEEDS = 20
TOLERANCE = 0.05  # B6's feasibility tolerance
SCATTER_BETA = DEFAULT_BETAS[5]  # 0.0108, the curve beta nearest 0.01
OUT = Path("results/analysis")

# (label, scale, value). Pairwise targets are the source paper's QHD levels on
# FedArtML's scale; "ours" is HD to uniform. RQ1 anchors are the B2 byproduct levels.
TARGETS = (
    ("paper QHD 0.10", "pairwise", 0.10),
    ("paper QHD 0.17", "pairwise", 0.17),
    ("ours 0.40", "ours", 0.40),
    ("ours 0.77", "ours", 0.77),
    *((f"RQ1 anchor {v:.2f}", "ours", v) for v in (0.10, 0.13, 0.36, 0.38, 0.42)),
)


def grid_rows() -> pd.DataFrame:
    """The latest partitions.jsonl row for every b3_quantity grid cell."""
    configs = expand(GRID)
    out_dir = configs[0].runtime.out_dir
    latest = {r["partition_id"]: r for r in read_jsonl(Path(out_dir) / "partitions.jsonl")}
    pids = [partition_id(c, seeds_for(c).partition) for c in configs]
    missing = [p for p in pids if p not in latest]
    if missing:
        raise SystemExit(f"{len(missing)} grid cells not generated; run the b3_quantity grid first")
    return pd.DataFrame([latest[p] for p in pids])


def seed_sweep(labels, floor: int, n_seeds: int) -> tuple[pd.DataFrame, list[list[np.ndarray]]]:
    num_classes = len(np.unique(labels))
    rows, scatter_parts = [], []
    for beta in DEFAULT_BETAS:
        for trial in range(n_seeds):
            seed = derive_seeds(trial).partition
            stats = {}
            parts = quantity_only(labels, K, beta, floor, np.random.default_rng(seed), stats)
            if beta == SCATTER_BETA:
                scatter_parts.append(parts)
            m = partition_metrics(labels, parts, num_classes)
            rows.append(
                {
                    "beta": beta,
                    "trial": trial,
                    "seed": seed,
                    "n_floored_clients": stats["n_floored_clients"],
                    "min_size_bound": min_size_bound(
                        len(labels), floor, stats["n_floored_clients"]
                    ),
                    **m,
                }
            )
    return pd.DataFrame(rows), scatter_parts


def equal_split_baseline(labels, n_seeds: int) -> pd.DataFrame:
    """Equal-size random split (np.array_split of a permutation), same K."""
    num_classes = len(np.unique(labels))
    rows = []
    for trial in range(n_seeds):
        rng = np.random.default_rng(derive_seeds(trial).partition)
        parts = np.array_split(rng.permutation(len(labels)), K)
        rows.append(partition_metrics(labels, parts, num_classes))
    return pd.DataFrame(rows)


def expected_sample_hd(labels, sizes, reps: int = 400) -> np.ndarray:
    """Mean HD(sample, global) of a uniform sample without replacement, per size."""
    counts = np.bincount(labels)
    global_dist = counts / counts.sum()
    rng = np.random.default_rng(0)
    out = []
    for n in sizes:
        draws = rng.multivariate_hypergeometric(counts, int(n), size=reps)
        out.append(np.mean([hellinger(d / n, global_dist) for d in draws]))
    return np.array(out)


def scatter(labels, parts_by_seed, baseline_hd: float, path: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(np.unique(labels))
    global_dist = np.bincount(labels) / len(labels)
    rows = []
    for trial, parts in enumerate(parts_by_seed):
        counts = class_counts(labels, parts, num_classes)
        for client, row in enumerate(counts):
            n = int(row.sum())
            rows.append(
                {
                    "trial": trial,
                    "client": client,
                    "size": n,
                    "label_hd": hellinger(row / n, global_dist),
                }
            )
    df = pd.DataFrame(rows)
    grid = np.unique(np.geomspace(df["size"].min(), df["size"].max(), 40).astype(int))
    reference = expected_sample_hd(labels, grid)

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.scatter(
        df["size"],
        df["label_hd"],
        s=16,
        color="#2a78d6",
        alpha=0.55,
        edgecolors=surface,
        linewidths=0.4,
        label=f"client ({len(parts_by_seed)} seeds x {K} clients)",
        zorder=3,
    )
    ax.plot(
        grid,
        reference,
        color="#eb6834",
        linewidth=1.5,
        label="expected, uniform sample of that size",
        zorder=4,
    )
    ax.axhline(
        baseline_hd,
        color=muted,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        label=f"equal-size random split, mean ({baseline_hd:.3f})",
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("client size (samples, log scale)", color=ink)
    ax.set_ylabel("client label-HD to global", color=ink)
    ax.set_title(
        f"quantity_only, CIFAR-10, K = {K}, beta = {SCATTER_BETA:.4g}: "
        "label-HD per client vs size",
        color=ink,
        fontsize=10,
        loc="left",
    )
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=muted, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return df


def summarise_range(grid: pd.DataFrame, sweep: pd.DataFrame, N: int, floor: int) -> pd.DataFrame:
    rows = []
    for k, group in grid.groupby("K"):
        sources = [("grid, 3 trials", group)]
        if k == K:
            sources.append((f"K=30 sweep, {sweep.trial.nunique()} seeds", sweep))
        for source, g in sources:
            smallest = int(g["min_client_size"].min())
            universal = min_size_bound(N, floor, k - 1)
            rows.append(
                {
                    "K": k,
                    "source": source,
                    "n_partitions": len(g),
                    "max_quantity_hd": g["realized_quantity_hd"].max(),
                    "ceiling_nominal_floor": max_quantity_hd(N, k, floor),
                    "ceiling_realized_min": max_quantity_hd(N, k, smallest),
                    "realized_min_size": smallest,
                    "universal_bound": universal,
                    "max_pairwise": g["realized_quantity_hd_pairwise"].max(),
                    "pairwise_ceiling": np.sqrt(1 / k),
                }
            )
    return pd.DataFrame(rows)


def per_beta(sweep: pd.DataFrame, curve: dict, baseline: pd.DataFrame, floor: int) -> pd.DataFrame:
    monotone = {p["beta"]: p["monotone_quantity_hd"] for p in curve["curve"]}
    rows = []
    for beta, g in sweep.groupby("beta"):
        q = g["realized_quantity_hd"]
        rows.append(
            {
                "beta": beta,
                "qhd_mean": q.mean(),
                "qhd_sd": q.std(ddof=1),
                "within_0.05_of_mean": (abs(q - q.mean()) <= TOLERANCE).mean(),
                "curve_value": monotone[beta],
                "within_0.05_of_curve": (abs(q - monotone[beta]) <= TOLERANCE).mean(),
                "pairwise_mean": g["realized_quantity_hd_pairwise"].mean(),
                "n_floored_mean": g["n_floored_clients"].mean(),
                "min_size_min": g["min_client_size"].min(),
                "bound_min": g["min_size_bound"].min(),
                "seeds_below_floor": int((g["min_client_size"] < floor).sum()),
                "label_hd_mean": g["realized_label_hd"].mean(),
                "label_hd_sd": g["realized_label_hd"].std(ddof=1),
                "label_hd_over_equal": g["realized_label_hd"].mean()
                / baseline["realized_label_hd"].mean(),
                "max_dev_mean": g["max_per_client_label_deviation"].mean(),
                "max_dev_max": g["max_per_client_label_deviation"].max(),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def resolve_targets(labels, curve: dict, n_seeds: int) -> pd.DataFrame:
    seeds = [derive_seeds(t).partition for t in range(n_seeds)]
    rows = []
    for label, scale, value in TARGETS:
        target = quantity_hd_from_pairwise(value, K) if scale == "pairwise" else value
        interpolated = beta_for_target(curve, target)
        refined = beta_for_target(curve, target, labels=labels, tol=0.005)
        check = evaluate(labels, K, "quantity_only", refined, seeds)
        realized = np.array([t["quantity_hd"] for t in check["trials"]])
        rows.append(
            {
                "target": label,
                "target_qhd_ours": target,
                "beta_interpolated": interpolated,
                "beta_refined": refined,
                f"qhd_mean_{n_seeds}_seeds": realized.mean(),
                "qhd_sd": realized.std(ddof=1),
                "within_0.05": (abs(realized - target) <= TOLERANCE).mean(),
                "label_hd_mean": check["mean_label_hd"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    labels = load_labels(DatasetConfig(name="cifar10", num_classes=10))
    N, floor = len(labels), default_quantity_floor(10)
    curve = load_calibration(CURVE_ID)
    grid = grid_rows()
    sweep, scatter_parts = seed_sweep(labels, floor, args.seeds)
    baseline = equal_split_baseline(labels, args.seeds)
    provenance = {**git_provenance(), "generated_at": utc_stamp()}

    conversion_gap = max(
        abs(
            quantity_hd_from_pairwise(r.realized_quantity_hd_pairwise, r.K) - r.realized_quantity_hd
        )
        for df in (grid, sweep.assign(K=K))
        for r in df.itertuples()
    )
    range_table = summarise_range(grid, sweep, N, floor)
    beta_table = per_beta(sweep, curve, baseline, floor)
    targets = resolve_targets(labels, curve, args.seeds)
    scatter_df = scatter(
        labels,
        scatter_parts,
        baseline["realized_label_hd"].mean(),
        OUT / "b3_label_hd_vs_size.png",
    )

    for name, df in (
        ("b3_k30_seeds.csv", sweep),
        ("b3_targets.csv", targets),
        ("b3_scatter_clients.csv", scatter_df),
    ):
        atomic_write_text(
            OUT / name, df.assign(**provenance).to_csv(index=False, lineterminator="\n")
        )

    fmt = lambda v: f"{v:.4g}"
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("== 1. Achievable range ==")
        print(range_table.to_string(index=False, float_format=fmt))
        print(
            f"\nmax |quantity_hd_from_pairwise - direct| over {len(grid) + len(sweep)} rows: "
            f"{conversion_gap:.2e}"
        )
        print("\n== 2-3. Targets (paper translation, RQ1 anchors) ==")
        print(targets.to_string(index=False, float_format=fmt))
        print(f"\n== 4-6. Per beta, K = {K}, {args.seeds} seeds ==")
        print(beta_table.to_string(index=False, float_format=fmt))
        b = baseline
        print(
            f"\nequal-size random split, K = {K}, {len(b)} seeds: label-HD "
            f"{b.realized_label_hd.mean():.4f} +/- {b.realized_label_hd.std(ddof=1):.4f}, "
            f"max per-client deviation {b.max_per_client_label_deviation.mean():.4f} "
            f"(max {b.max_per_client_label_deviation.max():.4f})"
        )
        print("\n== grid, K in {10, 30, 100}: mean over 3 trials at every 4th beta ==")
        picks = grid[grid.beta.isin(DEFAULT_BETAS[::4])]
        table = picks.groupby(["K", "beta"])[
            ["realized_quantity_hd", "realized_label_hd", "n_floored_clients", "min_client_size"]
        ].mean()
        print(table.to_string(float_format=fmt))
    print(f"\n-> {OUT}  ({provenance})")


if __name__ == "__main__":
    main()
