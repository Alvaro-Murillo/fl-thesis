"""Generate one partition and record it.

    python -m experiments.run_partition --config config/smoke.yaml

Partitions are generated separately from training and cached by `partition_id`,
so RQ1 costs no GPU time and the feasibility gate runs before any training
does. Re-running with the same config is a no-op unless --force is passed.

A cell whose min-size redraw budget (`partition.max_redraws`) runs out is
recorded as an explicitly infeasible row - status `infeasible_min_size`, no
.npz, no metrics - rather than looping on or relaxing the minimum.

quantity_only never redraws: its rows carry the resolved `quantity_floor` and
`n_floored_clients` (null for the other modes) and `n_redraws` 0. pure_label
never redraws either (`n_redraws` 0): every client holds N // K or N // K + 1
samples. combined takes its client sizes from the same draw as quantity_only,
so its rows carry `quantity_floor` and `n_floored_clients` too, and `n_redraws` 0.

pure_label and combined draw a label mix per client and then fit it to the
class stock, so a client's realized mix differs from its draw. Their rows
record that drift: `label_drift` per client (Hellinger, in client order; null
for an empty client), with `mean_label_drift` and `max_label_drift` over the
non-empty clients. The three fields are null for the other modes.

Targets: `feasible` checks the label target only, |realized - target| <= 0.05
per partition. On the quantity axis a target is a level name: every size draw
is kept, and the tolerance applies to the level's mean over 20 seeds
(`experiments.quantity_levels`), never to one partition. Every row records
`label_hd_deviation` and `quantity_hd_deviation` (realized - target, null
without a target) and `client_sizes`. Rows with the same dataset, K, mode,
quantity_floor, beta and partition_seed therefore share one client-size
vector, and the field shows it.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.config import FLOORED_MODES, PartitionMode, load_config
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
from partitioning.allocations import pure_label
from partitioning.dirichlet import MinSizeInfeasible, natural_per_class, natural_per_client
from partitioning.metrics import partition_metrics
from partitioning.quantity import default_quantity_floor, quantity_only
from partitioning.two_stage_method import combined

FEASIBILITY_TOLERANCE = 0.05


def resolved_quantity_floor(config) -> int | None:
    """The size floor quantity_only and combined apply, in samples; None for the other modes."""
    p = config.partition
    if p.mode not in FLOORED_MODES:
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
    if p.mode is PartitionMode.PURE_LABEL:
        labels = load_labels(config.dataset) if labels is None else labels
        client_indices = pure_label(labels, p.K, p.alpha, rng, stats=stats)
        if stats is not None:
            stats["n_redraws"] = 0
        return labels, client_indices
    if p.mode is PartitionMode.COMBINED:
        labels = load_labels(config.dataset) if labels is None else labels
        floor = resolved_quantity_floor(config)
        client_indices = combined(labels, p.K, p.alpha, p.beta, floor, rng, stats=stats)
        if stats is not None:
            stats["n_redraws"] = 0
        return labels, client_indices
    if p.mode is PartitionMode.NATURAL_PER_CLASS:
        partitioner = natural_per_class
        kwargs = {"size_cap": p.size_cap, "class_order": p.class_order}
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


def label_drift_fields(stats: dict) -> dict:
    """Row fields for the drift of each client's realized label mix from its draw.

    All null for a mode that draws no per-client mix (stats has no `label_drift`).
    An empty client's drift is NaN in stats and null here, so the row stays valid JSON.
    """
    drift = stats.get("label_drift")
    if drift is None:
        return {"label_drift": None, "mean_label_drift": None, "max_label_drift": None}
    drift = np.asarray(drift, dtype=float)
    return {
        "label_drift": [None if np.isnan(d) else float(d) for d in drift],
        "mean_label_drift": float(np.nanmean(drift)),
        "max_label_drift": float(np.nanmax(drift)),
    }


def feasible(metrics: dict, config) -> bool:
    """|realized - target| <= 0.05 for the label target, if the config sets one.

    The quantity target never enters it: that tolerance applies to a level's
    20-seed mean (`experiments.quantity_levels`), not to one partition.
    """
    target = config.partition.target_label_hd
    return target is None or abs(metrics["realized_label_hd"] - target) <= FEASIBILITY_TOLERANCE


def target_deviations(metrics: dict, config) -> dict:
    """Realized minus target HD per axis; null where the config sets no target. Diagnostic only."""
    p = config.partition
    return {
        f"{axis}_hd_deviation": (
            None if target is None else metrics[f"realized_{axis}_hd"] - target
        )
        for axis, target in (("label", p.target_label_hd), ("quantity", p.target_quantity_hd))
    }


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
        "class_order": p.class_order,
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
        **target_deviations(metrics, config),
        "client_sizes": [len(idx) for idx in client_indices],
        **label_drift_fields(stats),
        "feasible": feasible(metrics, config),
        "generated_at": utc_stamp(),
        **git_provenance(),
    }
    save_partition(out_dir, pid, client_indices, row)
    return row


def describe(row: dict) -> str:
    floored = row["mode"] in {m.value for m in FLOORED_MODES}
    if row["mode"] == PartitionMode.QUANTITY_ONLY.value:
        knobs = f"floor={row['quantity_floor']:<4} K={row['K']:<3} beta={row['beta']:<8.4g}"
    elif row["mode"] == PartitionMode.COMBINED.value:
        knobs = (
            f"floor={row['quantity_floor']:<4} K={row['K']:<3} alpha={row['alpha']:<7g} "
            f"beta={row['beta']:<8.4g}"
        )
    else:
        knobs = f"cap={row['size_cap']:<10} K={row['K']:<3} alpha={row['alpha']:<7g}"
        if row.get("class_order", "index") != "index":  # absent from rows recorded before it
            knobs += f" order={row['class_order']}"
    head = f"{row['partition_id']}  {row['mode']:<18} {knobs} trial={row['trial']}"
    if row["status"] != "generated":
        return f"{head}  INFEASIBLE after {row['n_redraws']} draws"
    tail = f"floored={row['n_floored_clients']}" if floored else f"redraws={row['n_redraws']}"
    if row.get("mean_label_drift") is not None:  # absent from rows recorded before it existed
        tail += f"  drift={row['mean_label_drift']:.4f}"
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
