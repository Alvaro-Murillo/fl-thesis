"""Inputs of the paper's results outputs: runs, partitions and per-class results by arm.

Everything is read from the recorded results, read-only: runs.jsonl and each run's own
files, partitions.jsonl and the stored partitions, the per-class results, and the
calibration curves. Which runs belong to an output is decided by the grid files: a grid
is expanded (`experiments.grid.expand`, for_training=False), each config's run id is its
config_hash, and the configs are not modified, so the hashes are the recorded ones even
when the results are read from another checkout (`results_dir`).

Arms carry the paper's display names (`arm_name`): pure, natural uncapped, natural capped,
natural per-client, combined; "natural capped, reversed order" for the class-order check
and "quantity-only" for the quantity_only partitions. The code names of the size cap never
appear in an output.

Trials: CIFAR-10 reports trials 0, 1, 2, 5 and 6 and Covtype trials 0-4; CIFAR-10 trials
3-4 and Covtype trials 5-6 tuned mu and appear only in the mu sweep. Both sets are
parameters. A reported grid with a run that is missing or not completed at a requested
trial stops the outputs (`MissingRuns`); nothing falls back to fewer trials. The one grid
allowed to be absent is the post-hoc 400-epoch Covtype check, whose rows read "pending"
until it is trained.

Covtype FedProx (covtype_rq2.yaml's FedProx cases) is excluded unless `covtype_fedprox` is
set: its configs are dropped before any run id is looked up, so no row, history or
per-class file of those runs is read. covtype_mu_tuning.yaml's FedProx runs are a separate,
finished grid and are read for the mu sweep.

Size-weighted label-HD is the client-size-weighted mean over non-empty clients of the
Hellinger distance between the client's label distribution and the empirical global one,
the weighting of the Covtype partition measurements of 2026-11-08; the unweighted mean is
the recorded `realized_label_hd` (`partitioning.metrics.partition_metrics`), and both are
recomputed here from the stored partition so that one can be checked against the record.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.training_summary import accuracy_curve, runs_table
from experiments.config import config_hash
from experiments.datasets import load_labels
from experiments.grid import expand
from experiments.recording import load_partition, partition_id, read_jsonl
from experiments.seeding import seeds_for
from partitioning.metrics import class_counts, hellinger

GRIDS = Path("config/grids")
CIFAR_TRIALS = (0, 1, 2, 5, 6)
COVTYPE_TRIALS = (0, 1, 2, 3, 4)
CIFAR_TUNING_TRIALS = (3, 4)
COVTYPE_TUNING_TRIALS = (5, 6)
DATASET_NAMES = {"cifar10": "CIFAR-10", "covtype": "Covtype"}

# The trained grids by dataset; the centralized 400-epoch grid is the post-hoc check.
TRAINED_GRIDS = {
    "cifar10": ("rq2", "rq3", "rq2_class_order", "centralized"),
    "covtype": ("covtype_rq2", "covtype_centralized", "covtype_centralized_steps"),
}
MU_GRIDS = {"cifar10": "mu_tuning", "covtype": "covtype_mu_tuning"}
OPTIONAL_GRIDS = {"covtype_centralized_steps"}  # rows read "pending" until trained
COVTYPE_FEDPROX_GRID = "covtype_rq2"  # its FedProx half is read only with covtype_fedprox
POST_HOC_GRIDS = {"covtype_centralized_steps"}

ARM_ORDER = (
    "natural uncapped",
    "natural capped",
    "pure",
    "natural per-client",
    "natural capped, reversed order",
    "combined",
    "quantity-only",
    "centralized",
)


class MissingRuns(RuntimeError):
    """A reported grid has a run that is missing or not completed at a requested trial."""


def arm_name(partition) -> str:
    """The paper's display name of a partition's arm; "centralized" for no partition."""
    if partition is None:
        return "centralized"
    return arm_label(partition.mode.value, partition.size_cap, partition.class_order)


def arm_label(mode: str, size_cap: str = "none", class_order: str = "index") -> str:
    """The display name for a partition mode, size cap and class order."""
    if mode == "natural_per_class":
        if size_cap == "none":
            return "natural uncapped"
        return "natural capped" + (", reversed order" if class_order == "reversed" else "")
    return {
        "natural_per_client": "natural per-client",
        "pure_label": "pure",
        "combined": "combined",
        "quantity_only": "quantity-only",
    }[mode]


@dataclass(frozen=True)
class Sources:
    """Where the recorded results are and which trials are reported."""

    results_dir: Path = Path("results")
    data_root: Path = Path("data")
    cifar_trials: tuple[int, ...] = CIFAR_TRIALS
    covtype_trials: tuple[int, ...] = COVTYPE_TRIALS
    covtype_fedprox: bool = False

    def trials(self, dataset: str) -> tuple[int, ...]:
        return self.cifar_trials if dataset == "cifar10" else self.covtype_trials

    def excluded(self, config, grid: str) -> bool:
        """covtype_rq2.yaml's FedProx runs while they are off: never looked up."""
        return (
            not self.covtype_fedprox
            and grid == COVTYPE_FEDPROX_GRID
            and config.federated.algorithm == "fedprox"
        )


def grid_configs(name: str, trials=None) -> list:
    """A grid's configs, unmodified, at `trials` (all of the grid's trials when None)."""
    configs = expand(GRIDS / f"{name}.yaml", for_training=False)
    return [c for c in configs if trials is None or c.runtime.trial in trials]


def latest_rows(results_dir: Path) -> dict[str, dict]:
    """run_id -> its latest runs.jsonl row."""
    return {row["run_id"]: row for row in read_jsonl(Path(results_dir) / "runs.jsonl")}


def _config_columns(config, grid: str) -> dict:
    p = config.partition
    return {
        "grid": grid,
        "arm": arm_name(p),
        "class_order": None if p is None else p.class_order,
        "post_hoc": grid in POST_HOC_GRIDS,
    }


def load_runs(sources: Sources, grids_by_dataset=TRAINED_GRIDS, trials_by_dataset=None):
    """(runs, pending): one row per completed run of the grids at the reported trials.

    `pending` lists (grid, run_id, trial, status) of the optional grids' runs that are not
    completed. A missing run of any other grid raises MissingRuns.
    """
    rows = latest_rows(sources.results_dir)
    selected, columns, missing, pending = {}, {}, [], []
    for dataset, grids in grids_by_dataset.items():
        trials = (trials_by_dataset or {}).get(dataset, sources.trials(dataset))
        for grid in grids:
            for config in grid_configs(grid, trials):
                if sources.excluded(config, grid):
                    continue
                run_id = config_hash(config)
                status = rows.get(run_id, {}).get("status", "absent")
                if status != "completed":
                    entry = (grid, run_id, config.runtime.trial, status)
                    (pending if grid in OPTIONAL_GRIDS else missing).append(entry)
                    continue
                selected[run_id] = config
                columns[run_id] = _config_columns(config, grid)
    if missing:
        listing = "\n".join(f"  {g} {r} trial {t}: {s}" for g, r, t, s in missing)
        raise MissingRuns(f"{len(missing)} reported runs are not completed:\n{listing}")

    table = runs_table(sources.results_dir, set(selected))
    extra = pd.DataFrame.from_dict(columns, orient="index")
    extra.index.name = "run_id"
    table = table.join(extra, on="run_id")
    for key in ("final_balanced_acc", "best_balanced_acc", "git_commit", "git_dirty"):
        table[key] = [rows[r].get(key) for r in table["run_id"]]
    for key in ("final_balanced_acc", "best_balanced_acc"):
        table[key] = table[key].astype(float)
    table["best_round"] = [
        int(np.argmax(accuracy_curve(sources.results_dir, r))) + 1 for r in table["run_id"]
    ]
    return table, pending


# ---------------------------------------------------------------- partitions
def partition_rows(results_dir: Path) -> dict[str, dict]:
    """partition_id -> its latest partitions.jsonl row."""
    return {r["partition_id"]: r for r in read_jsonl(Path(results_dir) / "partitions.jsonl")}


def size_weighted_label_hd(counts: np.ndarray, global_dist: np.ndarray) -> tuple[float, float]:
    """(size-weighted, unweighted) mean label-HD over the non-empty clients of a count matrix."""
    sizes = counts.sum(axis=1)
    nonempty = sizes > 0
    hd = np.array([hellinger(c / c.sum(), global_dist) for c in counts[nonempty]])
    return float((hd * sizes[nonempty]).sum() / sizes[nonempty].sum()), float(hd.mean())


class Labels:
    """Training labels per dataset, loaded once, from `data_root`."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, dataset_config) -> np.ndarray:
        name = dataset_config.name
        if name not in self._cache:
            self._cache[name] = load_labels(replace(dataset_config, root=str(self.data_root)))
        return self._cache[name]


def partition_facts(sources: Sources, configs, labels: Labels) -> pd.DataFrame:
    """One row per distinct partition of `configs`: recorded fields and size-weighted label-HD.

    `recomputed_label_hd` is the unweighted mean recomputed from the stored partition, for
    the check against the recorded `realized_label_hd`.
    """
    recorded = partition_rows(sources.results_dir)
    out, seen = [], set()
    for config in configs:
        if config.partition is None:
            continue
        pid = partition_id(config, seeds_for(config).partition)
        if pid in seen:
            continue
        seen.add(pid)
        row = recorded[pid]
        y = labels(config.dataset)
        global_dist = np.bincount(y, minlength=config.dataset.num_classes) / len(y)
        counts = class_counts(
            y, load_partition(sources.results_dir, pid), config.dataset.num_classes
        )
        weighted, unweighted = size_weighted_label_hd(counts, global_dist)
        out.append(
            {
                "partition_id": pid,
                "dataset": config.dataset.name,
                "arm": arm_name(config.partition),
                "trial": config.runtime.trial,
                "target_label_hd": config.partition.target_label_hd,
                "target_quantity_hd": config.partition.target_quantity_hd,
                "realized_label_hd": row["realized_label_hd"],
                "realized_quantity_hd": row["realized_quantity_hd"],
                "size_weighted_label_hd": weighted,
                "recomputed_label_hd": unweighted,
                "max_client_size": row.get("max_client_size"),
                "mean_client_size": row.get("mean_client_size"),
                "mean_label_drift": row.get("mean_label_drift"),
                "max_label_drift": row.get("max_label_drift"),
                "n_redraws": row.get("n_redraws"),
                "git_commit": row.get("git_commit"),
                "git_dirty": row.get("git_dirty"),
            }
        )
    return pd.DataFrame(out)


def identical_partitions(results_dir: Path, pid_a: str, pid_b: str) -> bool:
    """True if two stored partitions hold the same client index arrays, client by client."""
    a, b = load_partition(results_dir, pid_a), load_partition(results_dir, pid_b)
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def identical_histories(results_dir: Path, run_a: str, run_b: str) -> bool:
    """True if two runs recorded the same test accuracy after every round."""
    return np.array_equal(accuracy_curve(results_dir, run_a), accuracy_curve(results_dir, run_b))


# ---------------------------------------------------------------- RQ1
RQ1_COLUMNS = (
    "dataset",
    "arm",
    "K",
    "alpha",
    "trial",
    "status",
    "n_redraws",
    "realized_label_hd",
    "realized_label_hd_pairwise",
    "realized_quantity_hd",
    "realized_quantity_hd_pairwise",
    "max_client_size",
    "mean_client_size",
    "partition_id",
    "source",
)


def rq1_from_grid(results_dir: Path, grid: str, trials=None) -> pd.DataFrame | None:
    """RQ1 rows from a partition-only grid's recorded partitions; None if any is missing."""
    recorded = partition_rows(results_dir)
    rows = []
    for config in grid_configs(grid, trials):
        pid = partition_id(config, seeds_for(config).partition)
        row = recorded.get(pid)
        if row is None:
            return None
        rows.append(
            {
                "dataset": config.dataset.name,
                "arm": arm_name(config.partition),
                "K": config.partition.K,
                "alpha": config.partition.alpha,
                "trial": config.runtime.trial,
                "status": row.get("status", "generated"),
                "partition_id": pid,
                "source": f"{grid}.yaml",
                **{
                    c: row.get(c)
                    for c in RQ1_COLUMNS
                    if c not in ("dataset", "arm", "K", "alpha", "trial", "status")
                    and c not in ("partition_id", "source")
                },
            }
        )
    return pd.DataFrame(rows, columns=list(RQ1_COLUMNS))


def rq1_from_session(path: Path, K: int = 30) -> pd.DataFrame:
    """Covtype RQ1 rows from the partition measurements of 2026-11-08 (`rq1` key).

    Those partitions were built through `experiments.run_partition.build_partition` from
    configs, but not recorded through the runner, on an uncommitted tree: PROVISIONAL.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for r in data["rq1"]:
        rows.append(
            {
                "dataset": "covtype",
                "arm": arm_label(r["mode"], r["size_cap"]),
                "K": K,
                "alpha": r["alpha"],
                "trial": r["trial"],
                "status": r["status"],
                "partition_id": r["partition_id"],
                "source": f"{Path(path).name} (provisional)",
                **{
                    c: r.get(c)
                    for c in RQ1_COLUMNS
                    if c not in ("dataset", "arm", "K", "alpha", "trial", "status")
                    and c not in ("partition_id", "source")
                },
            }
        )
    return pd.DataFrame(rows, columns=list(RQ1_COLUMNS))


def quantity_only_curve_points(calibration: dict) -> pd.DataFrame:
    """A quantity_only curve's points: beta, mean and sd of realized label- and quantity-HD."""
    return pd.DataFrame(
        [
            {
                "beta": p["beta"],
                "n": p["n_feasible"],
                "mean_quantity_hd": p["mean_quantity_hd"],
                "sd_quantity_hd": p["sd_quantity_hd"],
                "mean_label_hd": p["mean_label_hd"],
                "sd_label_hd": p["sd_label_hd"],
            }
            for p in calibration["curve"]
        ]
    )


# ---------------------------------------------------------------- per-class
def per_class_final(results_dir: Path, run_id: str) -> np.ndarray:
    """The final global model's per-class accuracy (recall), from the run's checked result."""
    from experiments.per_class_accuracy import per_class_accuracy, per_class_result

    result = per_class_result(results_dir, run_id)
    return np.asarray(per_class_accuracy(result["confusions"][-1]), dtype=float)
