"""Training results across trials: final accuracy and rounds-to-accuracy, mean and sd.

    python -m analysis.training_summary [--grid GRID ...] [--out-dir DIR] [--prefix NAME]

Reads the completed runs in {out_dir}/runs.jsonl (the latest row per run), each
run's resolved config and history.csv, and the realized HD of its partition
from partitions.jsonl. With --grid only that grid's runs are read, and out_dir
is the grid's. Writes, under {out_dir}/analysis/:

    {prefix}_runs.csv      one row per completed run: its condition, trial,
                           final_acc, best_acc, rta, and its partition's
                           realized label-HD and quantity-HD
    {prefix}_summary.csv   one row per condition: number of runs and their
                           trials, mean and sd of final_acc, best_acc and rta,
                           mean realized label-HD and quantity-HD, and
                           rta_of_mean_curve

A condition is a config without its runtime section (dataset, partition, model
and federated settings), so its trials are its replicates; `condition_id` is
the config_hash it would have with the default runtime. A target that names a
level is not its value: the realized HD columns are what the analysis uses.

final_acc is the test accuracy after the last round and best_acc the highest
after any round; round 0, the untrained model, counts for neither. rta is the
first round reaching 90% of the run's own best accuracy
(`experiments.recording.rounds_to_accuracy`). The sd is the sample standard
deviation (n - 1 in the denominator), empty for a single run. The source paper
reports the mean and standard deviation across trials without saying which
(Jimenez-Gutierrez et al., arXiv:2503.17070 v2, Section 4.3).

rta_of_mean_curve is a second reading of RTA across trials: the accuracy curves
of a condition's runs are averaged round by round, and RTA is taken on that
mean curve, which gives one integer per condition. The source paper's RTA
tables give one integer per condition without saying how its trials were
combined; rta_mean is the mean of the per-run values.

Reports only: no run is excluded or rerun here.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import RuntimeConfig, config_hash, load_config
from experiments.grid import expand
from experiments.recording import (
    atomic_write_text,
    git_provenance,
    read_jsonl,
    rounds_to_accuracy,
    runs_path,
    utc_stamp,
)

PREFIX = "training"
CONDITION_COLUMNS = (
    "dataset",
    "K",
    "mode",
    "size_cap",
    "alpha",
    "beta",
    "target_label_hd",
    "target_quantity_hd",
    "algorithm",
    "mu",
    "rounds",
    "local_epochs",
)
METRICS = ("final_acc", "best_acc", "rta")
HD_COLUMNS = ("realized_label_hd", "realized_quantity_hd")


def condition_id(config) -> str:
    """config_hash with the default runtime, so every trial of a cell shares it."""
    return config_hash(replace(config, runtime=RuntimeConfig()))


def condition_fields(config) -> dict:
    p, f = config.partition, config.federated
    return {
        "dataset": config.dataset.name,
        "K": p.K,
        "mode": p.mode.value,
        "size_cap": p.size_cap,
        "alpha": p.alpha,
        "beta": p.beta,
        "target_label_hd": p.target_label_hd,
        "target_quantity_hd": p.target_quantity_hd,
        "algorithm": f.algorithm,
        "mu": f.mu,
        "rounds": f.rounds,
        "local_epochs": f.local_epochs,
    }


def runs_table(out_dir: str | Path, run_ids=None) -> pd.DataFrame:
    """One row per completed run (latest runs.jsonl row), restricted to `run_ids` if given."""
    out_dir = Path(out_dir)
    latest = {row["run_id"]: row for row in read_jsonl(runs_path(out_dir))}
    partitions = {r["partition_id"]: r for r in read_jsonl(out_dir / "partitions.jsonl")}
    rows = []
    for run_id, row in latest.items():
        if row["status"] != "completed" or (run_ids is not None and run_id not in run_ids):
            continue
        config = load_config(out_dir / "runs" / run_id / "config.yaml")
        partition = partitions.get(row["partition_id"], {})
        rows.append(
            {
                "condition_id": condition_id(config),
                **condition_fields(config),
                "trial": config.runtime.trial,
                "run_id": run_id,
                "partition_id": row["partition_id"],
                **{column: partition.get(column) for column in HD_COLUMNS},
                **{metric: row[metric] for metric in METRICS},
            }
        )
    columns = ["condition_id", *CONDITION_COLUMNS, "trial", "run_id", "partition_id"]
    columns += [*HD_COLUMNS, *METRICS]
    table = pd.DataFrame(rows, columns=columns)
    for column in (*HD_COLUMNS, *METRICS):
        table[column] = table[column].astype(float)
    return table.sort_values(["condition_id", "trial"], ignore_index=True)


def accuracy_curve(out_dir: str | Path, run_id: str) -> np.ndarray:
    """Test accuracy after rounds 1, 2, ... (round 0 left out)."""
    history = pd.read_csv(Path(out_dir) / "runs" / run_id / "history.csv")
    return history.loc[history["round"] >= 1, "global_test_acc"].to_numpy(dtype=float)


def mean_curve_rta(out_dir: str | Path, run_ids) -> int | None:
    """RTA of the round-by-round mean accuracy curve of `run_ids`."""
    curves = [accuracy_curve(out_dir, run_id) for run_id in run_ids]
    if len({len(curve) for curve in curves}) != 1:
        raise ValueError(f"runs {list(run_ids)} have accuracy curves of different lengths")
    return rounds_to_accuracy(np.mean(curves, axis=0))


def summary_table(runs: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """One row per condition, in the order the conditions first appear in `runs`."""
    groups = runs.groupby("condition_id", sort=False, dropna=False)
    summary = groups[list(CONDITION_COLUMNS)].first()
    summary["n_runs"] = groups.size()
    summary["trials"] = groups["trial"].agg(lambda t: " ".join(map(str, sorted(t))))
    for column in HD_COLUMNS:
        summary[f"{column}_mean"] = groups[column].mean()
    for metric in METRICS:
        summary[f"{metric}_mean"] = groups[metric].mean()
        summary[f"{metric}_sd"] = groups[metric].std(ddof=1)
    summary["rta_of_mean_curve"] = groups["run_id"].agg(lambda ids: mean_curve_rta(out_dir, ids))
    return summary.reset_index()


def grid_runs(grids) -> tuple[set[str], dict[str, int], Path]:
    """The grids' run ids, the number of runs each condition should have, and the out_dir."""
    run_ids, expected, out_dirs = set(), {}, set()
    for grid in grids:
        for config in expand(grid, for_training=False):  # reads runs; trains nothing
            run_ids.add(config_hash(config))
            key = condition_id(config)
            expected[key] = expected.get(key, 0) + 1
            out_dirs.add(config.runtime.out_dir)
    if len(out_dirs) > 1:
        raise SystemExit(f"the grids write to different out_dirs: {sorted(out_dirs)}")
    return run_ids, expected, Path(out_dirs.pop())


def print_summary(summary: pd.DataFrame) -> None:
    def pm(mean, sd, fmt):
        return f"{mean:{fmt}}" + ("" if pd.isna(sd) else f" +/- {sd:{fmt}}")  # ASCII console

    report = summary[["condition_id", "mode", "alpha", "beta", "algorithm", "mu"]].copy()
    report["hd label/qty"] = [
        f"{a:.3f} / {b:.3f}"
        for a, b in zip(summary["realized_label_hd_mean"], summary["realized_quantity_hd_mean"])
    ]
    report["n"] = summary["n_runs"]
    report["final acc"] = [
        pm(m, s, ".4f") for m, s in zip(summary["final_acc_mean"], summary["final_acc_sd"])
    ]
    report["rta"] = [pm(m, s, ".1f") for m, s in zip(summary["rta_mean"], summary["rta_sd"])]
    report["rta of mean"] = summary["rta_of_mean_curve"]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(report.to_string(index=False, float_format=lambda v: f"{v:g}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grid", action="append", type=Path, help="repeatable; default: all runs")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="without --grid")
    parser.add_argument("--prefix", default=PREFIX, help=f"output file prefix (default {PREFIX})")
    args = parser.parse_args()

    run_ids, expected, out_dir = None, None, args.out_dir
    if args.grid:
        run_ids, expected, out_dir = grid_runs(args.grid)
    runs = runs_table(out_dir, run_ids)
    if runs.empty:
        raise SystemExit(f"no completed runs in {runs_path(out_dir)}")
    summary = summary_table(runs, out_dir)
    if expected is not None:
        summary.insert(
            summary.columns.get_loc("n_runs") + 1,
            "n_expected",
            summary["condition_id"].map(expected),
        )
        missing = len(set(expected) - set(summary["condition_id"]))
        incomplete = int((summary["n_runs"] < summary["n_expected"]).sum())
        print(
            f"{len(expected)} conditions in the grid: {missing} with no completed run, "
            f"{incomplete} with only some of their runs"
        )
    provenance = {**git_provenance(), "generated_at": utc_stamp()}

    out = out_dir / "analysis"
    for name, df in (("runs", runs), ("summary", summary)):
        atomic_write_text(
            out / f"{args.prefix}_{name}.csv",
            df.assign(**provenance).to_csv(index=False, lineterminator="\n"),
        )
    print_summary(summary)
    print(f"\n-> {out}/{args.prefix}_*  ({provenance})")


if __name__ == "__main__":
    main()
