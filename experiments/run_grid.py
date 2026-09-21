"""Generate every partition in a grid (partition stage only; no training).

    python -m experiments.run_grid --grid config/grids/rq1.yaml

Labels are loaded once per dataset. Cached cells (an .npz, or an infeasible
row with the same redraw budget) are skipped unless --force is passed, so a
killed sweep resumes where it stopped. Blocked grids are refused.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.run_partition import describe, generate_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configs = expand(args.grid)
    labels = {}
    generated = skipped = infeasible = 0
    for i, config in enumerate(configs, 1):
        name = config.dataset.name
        if name not in labels:
            labels[name] = load_labels(config.dataset)
        row = generate_partition(config, labels[name], force=args.force)
        if row is None:
            skipped += 1
            continue
        generated += 1
        infeasible += row["status"] != "generated"
        print(f"[{i}/{len(configs)}] {describe(row)}", flush=True)
    print(f"{len(configs)} cells: {generated} generated ({infeasible} infeasible), "
          f"{skipped} already recorded")


if __name__ == "__main__":
    main()
