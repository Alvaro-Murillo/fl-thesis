"""Quantity-level check: is every quantity target in a grid a feasible level?

    python -m experiments.quantity_levels --grid config/grids/rq3.yaml

On the quantity axis a target is a level name. Every size draw is kept, in
every mode, and the analysis uses each row's realized quantity-HD. The +/-0.05
tolerance applies to the level: it is feasible if its mean realized
quantity-HD over 20 seeds (trials 0-19) is within 0.05 of the target. The
label-axis tolerance stays per partition (`run_partition.feasible`).

For every distinct quantity level of a grid - (dataset, K, mode,
quantity_floor, beta, target_quantity_hd) among cells that set a quantity
target - this runs that cell's own partitioner at trials 0-19 and appends one
row per level to {out_dir}/quantity_levels.jsonl: per-seed realized
quantity-HD, mean, sd, min, max, the share of seeds within 0.05 of the target,
and `level_feasible`. It saves no partitions, redraws nothing and excludes
nothing; a failing level is reported, not removed. Rows join to partition rows
on the level fields.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.recording import append_jsonl, git_provenance, utc_stamp
from experiments.run_partition import (
    FEASIBILITY_TOLERANCE,
    build_partition,
    resolved_quantity_floor,
)
from experiments.seeding import generator, seeds_for
from partitioning.dirichlet import MinSizeInfeasible
from partitioning.metrics import partition_metrics

LEVEL_SEEDS = 20


def level_fields(config) -> dict:
    """The fields that identify a quantity level, as they appear in partition rows."""
    p = config.partition
    return {
        "dataset": config.dataset.name,
        "K": p.K,
        "mode": p.mode.value,
        "quantity_floor": resolved_quantity_floor(config),
        "beta": p.beta,
        "target_quantity_hd": p.target_quantity_hd,
    }


def distinct_levels(configs) -> list:
    """One representative config per distinct quantity level, in grid order."""
    levels = {}
    for config in configs:
        if config.partition.target_quantity_hd is not None:
            levels.setdefault(tuple(level_fields(config).values()), config)
    return list(levels.values())


def check_level(config, labels, n_seeds: int = LEVEL_SEEDS) -> dict:
    """Realized quantity-HD of `config`'s partitioner at trials 0..n_seeds-1, and the verdict."""
    if config.runtime.partition_seed is not None:
        raise ValueError("a level is measured over trials; unset runtime.partition_seed")
    target = config.partition.target_quantity_hd
    realized, infeasible = [], 0
    for trial in range(n_seeds):
        cell = replace(config, runtime=replace(config.runtime, trial=trial))
        try:
            _, parts = build_partition(cell, generator(seeds_for(cell).partition), labels, {})
        except MinSizeInfeasible:  # natural modes only; that cell has no partition
            infeasible += 1
            continue
        metrics = partition_metrics(labels, parts, config.dataset.num_classes)
        realized.append(metrics["realized_quantity_hd"])
    q = np.array(realized)
    mean = float(q.mean()) if len(q) else None
    return {
        **level_fields(config),
        "quantity_calibration_id": config.partition.quantity_calibration_id,
        "trials": list(range(n_seeds)),
        "n_infeasible_min_size": infeasible,
        "realized_quantity_hd": q.tolist(),
        "mean_quantity_hd": mean,
        "sd_quantity_hd": float(q.std(ddof=1)) if len(q) > 1 else None,
        "min_quantity_hd": float(q.min()) if len(q) else None,
        "max_quantity_hd": float(q.max()) if len(q) else None,
        "share_within_tolerance": (
            float(np.mean(np.abs(q - target) <= FEASIBILITY_TOLERANCE)) if len(q) else None
        ),
        "mean_deviation": None if mean is None else mean - target,
        "level_feasible": bool(mean is not None and abs(mean - target) <= FEASIBILITY_TOLERANCE),
        "generated_at": utc_stamp(),
        **git_provenance(),
    }


def describe_level(row: dict) -> str:
    verdict = "pass" if row["level_feasible"] else "FAIL"
    beta = "-" if row["beta"] is None else f"{row['beta']:.4g}"
    head = (
        f"{row['mode']:<14} K={row['K']:<3} target={row['target_quantity_hd']:<6g} beta={beta:<9}"
    )
    if row["mean_quantity_hd"] is None:
        return f"{head} no feasible partition  {verdict}"
    sd = "-" if row["sd_quantity_hd"] is None else f"{row['sd_quantity_hd']:.3f}"
    return (
        f"{head} mean={row['mean_quantity_hd']:.3f} sd={sd} "
        f"range {row['min_quantity_hd']:.3f}-{row['max_quantity_hd']:.3f} "
        f"within={row['share_within_tolerance']:.0%}  {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=LEVEL_SEEDS)
    args = parser.parse_args()

    configs = expand(args.grid, for_training=False)
    levels = distinct_levels(configs)
    if not levels:
        print(f"{args.grid} sets no quantity target - nothing to check")
        return
    labels = {}
    for config in levels:
        name = config.dataset.name
        if name not in labels:
            labels[name] = load_labels(config.dataset)
        row = check_level(config, labels[name], args.seeds)
        append_jsonl(Path(config.runtime.out_dir) / "quantity_levels.jsonl", row)
        print(describe_level(row), flush=True)


if __name__ == "__main__":
    main()
