"""Generate one partition and record it (work-plan task A4.3).

    python -m experiments.run_partition --config config/smoke.yaml

Partitions are generated separately from training and cached by `partition_id`,
so RQ1 costs no GPU time and the feasibility gate runs before any training
does. Re-running with the same config is a no-op unless --force is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import PartitionMode, load_config
from experiments.datasets import load_labels
from experiments.recording import (
    git_provenance, partition_exists, partition_id, save_partition, utc_stamp,
)
from experiments.seeding import generator, seeds_for
from partitioning.dirichlet import natural_per_class, natural_per_client
from partitioning.metrics import partition_metrics

FEASIBILITY_TOLERANCE = 0.05


def build_partition(config, rng):
    p = config.partition
    if p.mode is PartitionMode.NATURAL_PER_CLASS:
        partitioner, kwargs = natural_per_class, {"size_cap": p.size_cap}
    elif p.mode is PartitionMode.NATURAL_PER_CLIENT:
        partitioner, kwargs = natural_per_client, {}
    else:
        raise NotImplementedError(
            f"mode '{p.mode.value}' arrives with its work-plan task "
            "(B3 quantity_only, B4 pure_label, B5 combined)")
    labels = load_labels(config.dataset)
    client_indices = partitioner(labels, p.K, p.alpha, rng,
                                 min_size=p.min_size_policy, **kwargs)
    return labels, client_indices


def feasible(metrics: dict, config) -> bool:
    """|realized - target| <= 0.05 for every target the config actually sets."""
    checks = (("realized_label_hd", config.partition.target_label_hd),
              ("realized_quantity_hd", config.partition.target_quantity_hd))
    return all(abs(metrics[key] - target) <= FEASIBILITY_TOLERANCE
               for key, target in checks if target is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the partition is already cached")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = seeds_for(config)
    pid = partition_id(config, seeds.partition)
    out_dir = config.runtime.out_dir

    if partition_exists(out_dir, pid) and not args.force:
        print(f"{pid} already cached in {out_dir}/partitions - nothing to do")
        return

    labels, client_indices = build_partition(config, generator(seeds.partition))
    metrics = partition_metrics(labels, client_indices, config.dataset.num_classes)

    partition = config.partition
    row = {
        "partition_id": pid,
        "dataset": config.dataset.name,
        "K": partition.K,
        "mode": partition.mode.value,
        "alpha": partition.alpha,
        "beta": partition.beta,
        "target_label_hd": partition.target_label_hd,
        "target_quantity_hd": partition.target_quantity_hd,
        "min_size_policy": partition.min_size_policy,
        "size_cap": partition.size_cap,
        "partition_seed": seeds.partition,
        **metrics,
        "feasible": feasible(metrics, config),
        "generated_at": utc_stamp(),
        **git_provenance(),
    }
    save_partition(out_dir, pid, client_indices, row)

    print(f"{pid}  K={partition.K} alpha={partition.alpha} "
          f"label_hd={metrics['realized_label_hd']:.4f} "
          f"(pairwise {metrics['realized_label_hd_pairwise']})  "
          f"quantity_hd={metrics['realized_quantity_hd']:.4f}  "
          f"sizes {metrics['min_client_size']}-{metrics['max_client_size']}  "
          f"feasible={row['feasible']}")


if __name__ == "__main__":
    main()
