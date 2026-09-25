"""Generate one partition and record it.

    python -m experiments.run_partition --config config/smoke.yaml

Partitions are generated separately from training and cached by `partition_id`,
so RQ1 costs no GPU time and the feasibility gate runs before any training
does. Re-running with the same config is a no-op unless --force is passed.

A cell whose min-size redraw budget (`partition.max_redraws`) runs out is
recorded as an explicitly infeasible row - status `infeasible_min_size`, no
.npz, no metrics - rather than looping on or relaxing the minimum.

quantity_only never redraws: its rows carry the resolved `quantity_floor` and
`n_floored_clients` (null for the other modes) and `n_redraws` 0.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import PartitionMode, load_config
from experiments.datasets import load_labels
from experiments.recording import (
    git_provenance,
    infeasible_recorded,
    partition_exists,
    partition_id,
    record_infeasible,
    save_partition,
    utc_stamp,
)
from experiments.seeding import generator, seeds_for
from partitioning.dirichlet import MinSizeInfeasible, natural_per_class, natural_per_client
from partitioning.metrics import partition_metrics
from partitioning.quantity import default_quantity_floor, quantity_only

FEASIBILITY_TOLERANCE = 0.05


def resolved_quantity_floor(config) -> int | None:
    """The floor quantity_only applies, in samples; None for every other mode."""
    p = config.partition
    if p.mode is not PartitionMode.QUANTITY_ONLY:
        return None
    if p.quantity_floor is not None:
        return p.quantity_floor
    return default_quantity_floor(config.dataset.num_classes)


def build_partition(config, rng, labels=None, stats: dict | None = None):
    """(labels, client_indices) for a config. Raises MinSizeInfeasible."""
    p = config.partition
    if p.mode is PartitionMode.QUANTITY_ONLY:
        labels = load_labels(config.dataset) if labels is None else labels
        floor = resolved_quantity_floor(config)
        client_indices = quantity_only(labels, p.K, p.beta, floor, rng, stats=stats)
        if stats is not None:
            stats["n_redraws"] = 0
        return labels, client_indices
    if p.mode is PartitionMode.NATURAL_PER_CLASS:
        partitioner, kwargs = natural_per_class, {"size_cap": p.size_cap}
    elif p.mode is PartitionMode.NATURAL_PER_CLIENT:
        partitioner, kwargs = natural_per_client, {}
    else:
        raise NotImplementedError(f"mode '{p.mode.value}' is not implemented yet")
    labels = load_labels(config.dataset) if labels is None else labels
    client_indices = partitioner(
        labels,
        p.K,
        p.alpha,
        rng,
        min_size=p.min_size_policy,
        max_redraws=p.max_redraws,
        stats=stats,
        **kwargs,
    )
    return labels, client_indices


def feasible(metrics: dict, config) -> bool:
    """|realized - target| <= 0.05 for every target the config actually sets."""
    checks = (
        ("realized_label_hd", config.partition.target_label_hd),
        ("realized_quantity_hd", config.partition.target_quantity_hd),
    )
    return all(
        abs(metrics[key] - target) <= FEASIBILITY_TOLERANCE
        for key, target in checks
        if target is not None
    )


def _row_header(config, pid: str, partition_seed: int) -> dict:
    p = config.partition
    return {
        "partition_id": pid,
        "dataset": config.dataset.name,
        "K": p.K,
        "mode": p.mode.value,
        "alpha": p.alpha,
        "beta": p.beta,
        "target_label_hd": p.target_label_hd,
        "target_quantity_hd": p.target_quantity_hd,
        "min_size_policy": p.min_size_policy,
        "size_cap": p.size_cap,
        "max_redraws": p.max_redraws,
        "quantity_floor": resolved_quantity_floor(config),
        "trial": config.runtime.trial,
        "partition_seed": partition_seed,
    }


def generate_partition(config, labels=None, force: bool = False) -> dict | None:
    """Generate, record and return one partition row; None if already cached."""
    seeds = seeds_for(config)
    pid = partition_id(config, seeds.partition)
    out_dir = config.runtime.out_dir
    cached = partition_exists(out_dir, pid) or infeasible_recorded(
        out_dir, pid, config.partition.max_redraws
    )
    if cached and not force:
        return None

    stats: dict = {}
    header = _row_header(config, pid, seeds.partition)
    try:
        labels, client_indices = build_partition(config, generator(seeds.partition), labels, stats)
    except MinSizeInfeasible:
        row = {
            **header,
            "status": "infeasible_min_size",
            "n_redraws": stats["n_redraws"],
            "feasible": False,
            "generated_at": utc_stamp(),
            **git_provenance(),
        }
        record_infeasible(out_dir, pid, row)
        return row

    metrics = partition_metrics(labels, client_indices, config.dataset.num_classes)
    row = {
        **header,
        "status": "generated",
        "n_redraws": stats["n_redraws"],
        "n_floored_clients": stats.get("n_floored_clients"),
        **metrics,
        "feasible": feasible(metrics, config),
        "generated_at": utc_stamp(),
        **git_provenance(),
    }
    save_partition(out_dir, pid, client_indices, row)
    return row


def describe(row: dict) -> str:
    quantity = row["mode"] == PartitionMode.QUANTITY_ONLY.value
    if quantity:
        knobs = f"floor={row['quantity_floor']:<4} K={row['K']:<3} beta={row['beta']:<8.4g}"
    else:
        knobs = f"cap={row['size_cap']:<10} K={row['K']:<3} alpha={row['alpha']:<7g}"
    head = f"{row['partition_id']}  {row['mode']:<18} {knobs} trial={row['trial']}"
    if row["status"] != "generated":
        return f"{head}  INFEASIBLE after {row['n_redraws']} draws"
    tail = f"floored={row['n_floored_clients']}" if quantity else f"redraws={row['n_redraws']}"
    return (
        f"{head}  label_hd={row['realized_label_hd']:.4f}  "
        f"quantity_hd={row['realized_quantity_hd']:.4f}  "
        f"sizes {row['min_client_size']}-{row['max_client_size']}  {tail}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="regenerate even if the partition is already cached"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    row = generate_partition(config, force=args.force)
    if row is None:
        pid = partition_id(config, seeds_for(config).partition)
        print(f"{pid} already recorded in {config.runtime.out_dir} - nothing to do")
    else:
        print(describe(row))


if __name__ == "__main__":
    main()
