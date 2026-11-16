"""The paper's results, computed once from the recorded runs and partitions, and its tables.

`prepare` reads every reported run and partition (`analysis.paper_data`) and makes the
checks the outputs rest on; the builders below turn the result into the tables, and
`analysis.paper_figures` and `analysis.paper_outputs` into the figures and numbers.

Cells and arms. A cell is a dataset, a grid, an arm, a label level L and, in RQ3, a
quantity level Q. The levels are the grids' target names; every HD value reported for a
cell is the mean of its partitions' realized values over the reported trials. On CIFAR-10 at
label level 0 the two natural arms are the same partition (the cap does not bind at alpha
1000): `prepare` checks that the stored partitions are identical client by client and that
the runs recorded the same accuracy after every round, then reports them as one arm,
"natural", counted once (the uncapped runs). On Covtype the two differ at level 0 and stay
apart.

Pending. `pending` names the (dataset, algorithm) pairs whose runs are still training:
(covtype, fedprox) until `Sources.covtype_fedprox` is set. Every cell they would fill reads
"pending" and every number from them is a placeholder; the code that fills them is the same
as for the other pairs, so switching the flag fills them. A post-hoc grid that is not
trained (Covtype centralized at 400 epochs) has its row marked pending the same way.

Accuracy is the final-round test accuracy; RTA is the first round reaching 90% of the run's
own best accuracy, averaged over trials (mean of the per-run values). On Covtype accuracy is
primary, as in the source paper; balanced accuracy is ours and labelled so.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper_data import (
    COVTYPE_TUNING_TRIALS,
    CIFAR_TUNING_TRIALS,
    DATASET_NAMES,
    GRIDS,
    MU_GRIDS,
    TRAINED_GRIDS,
    Labels,
    Sources,
    grid_configs,
    identical_histories,
    identical_partitions,
    load_runs,
    partition_facts,
    per_class_final,
    quantity_only_curve_points,
    rq1_from_grid,
    rq1_from_session,
)
from analysis.paper_stats import (
    deltas,
    fit_regression,
    interactions,
    lack_of_fit,
    mean_sd,
    model_interaction,
    paired,
)
from analysis.paper_tables import ACC, HD, RTA, Col, Table

NATURAL = "natural"  # CIFAR-10 label level 0: both natural arms, one partition
SHORT_ARM = {
    "natural uncapped": "uncapped",
    "natural capped": "capped",
    "pure": "pure",
    NATURAL: "natural",
    "centralized": "centralized",
}
RQ2_ARMS = ("natural uncapped", "natural capped", "pure")
RQ2_PAIRS = (
    ("pure", "natural uncapped"),
    ("pure", "natural capped"),
    ("natural capped", "natural uncapped"),
)
PAIR_NAMES = {
    ("pure", "natural uncapped"): "pure − uncapped",
    ("pure", "natural capped"): "pure − capped",
    ("natural capped", "natural uncapped"): "capped − uncapped",
    ("pure", NATURAL): "pure − natural",
}
ALGORITHMS = ("fedavg", "fedprox")
ALGORITHM_NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "centralized": "centralized"}
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
LEVEL_TOLERANCE = 0.05
RQ2_GRID = {"cifar10": "rq2", "covtype": "covtype_rq2"}
CENTRALIZED_GRID = {"cifar10": "centralized", "covtype": "covtype_centralized"}
CALIBRATION_DIR = Path("calibration")
CIFAR_QUANTITY_CURVE = "8f995f9546"  # quantity_only, CIFAR-10, K = 30, 20 trials
COVTYPE_QUANTITY_CURVE = None  # none committed yet: generated with the Covtype RQ1 grids


def trial_set_label(trials) -> str:
    return "{" + ",".join(str(t) for t in sorted(trials)) + "}"


@dataclass
class Bundle:
    sources: Sources
    runs: pd.DataFrame
    parts: pd.DataFrame
    pending_runs: list
    pending: set
    checks: dict
    mu_runs: pd.DataFrame
    labels: Labels
    rq1: pd.DataFrame | None = None
    rq1_status: dict = field(default_factory=dict)

    def trials(self, dataset: str) -> tuple[int, ...]:
        return tuple(sorted(self.sources.trials(dataset)))

    def trial_note(self, *datasets) -> str:
        return "; ".join(
            f"{DATASET_NAMES[d]}: n = {len(self.trials(d))} trials {trial_set_label(self.trials(d))}"
            for d in datasets
        )

    def is_pending(self, dataset: str, algorithm: str) -> bool:
        return (dataset, algorithm) in self.pending

    def covtype_level0_note(self) -> str | None:
        same = self.checks.get("covtype_level0_identical_trials") or []
        if not same:
            return None
        trials = ", ".join(str(t) for t, _ in same)
        runs = "and so are its FedAvg runs" if all(h for _, h in same) else "but its runs differ"
        return (
            f"Covtype level 0, trial {trials}: the cap does not bind in that draw, so the capped "
            f"and uncapped partitions are identical client by client, {runs}; the arms are kept "
            "apart as at every other trial."
        )


# ---------------------------------------------------------------- prepare
def _capped_levels() -> dict[float, float]:
    """rq2.yaml's capped alpha -> its label level, for the class-order cells."""
    out = {}
    for c in grid_configs("rq2"):
        if c.partition.size_cap == "niid_bench":
            out[c.partition.alpha] = c.partition.target_label_hd
    return out


def reported_configs(sources: Sources) -> list:
    configs = []
    for dataset, grids in TRAINED_GRIDS.items():
        for grid in grids:
            configs += [
                c
                for c in grid_configs(grid, sources.trials(dataset))
                if not sources.excluded(c, grid)
            ]
    return configs


def prepare(sources: Sources, covtype_rq1_session: Path | None = None) -> Bundle:
    runs, pending_runs = load_runs(sources)
    levels = _capped_levels()
    order = runs["grid"] == "rq2_class_order"
    runs.loc[order, "target_label_hd"] = runs.loc[order, "alpha"].map(levels)
    runs["L"] = runs["target_label_hd"]
    runs["Q"] = runs["target_quantity_hd"]

    labels = Labels(sources.data_root)
    parts = partition_facts(sources, reported_configs(sources), labels)
    runs = runs.merge(
        parts[["partition_id", "size_weighted_label_hd"]], on="partition_id", how="left"
    )
    checks = {
        "max_abs_recomputed_label_hd_difference": float(
            (parts["recomputed_label_hd"] - parts["realized_label_hd"]).abs().max()
        )
    }

    # CIFAR-10 level 0: the two natural arms must be one partition and one set of runs.
    runs["cell_arm"] = runs["arm"]
    runs["merged_out"] = False
    level0 = []
    zero = (runs["grid"] == "rq2") & (runs["L"] == 0.0)
    for trial in sources.cifar_trials:
        for algorithm in ALGORITHMS:
            sel = zero & (runs["trial"] == trial) & (runs["algorithm"] == algorithm)
            unc = runs[sel & (runs["arm"] == "natural uncapped")].iloc[0]
            cap = runs[sel & (runs["arm"] == "natural capped")].iloc[0]
            same = (
                identical_partitions(sources.results_dir, unc.partition_id, cap.partition_id)
                and identical_histories(sources.results_dir, unc.run_id, cap.run_id)
                and unc.final_acc == cap.final_acc
            )
            level0.append((trial, algorithm, unc.run_id, cap.run_id, same))
    if not all(s for *_, s in level0):
        raise RuntimeError(f"CIFAR-10 level-0 natural arms differ: {level0}")
    checks["cifar_level0_identical"] = level0
    runs.loc[zero & (runs["arm"] == "natural uncapped"), "cell_arm"] = NATURAL
    runs.loc[zero & (runs["arm"] == "natural capped"), "merged_out"] = True
    parts["cell_arm"] = parts["arm"]

    # Covtype level 0: the cap binds in most draws, so the arms stay apart; record the trials
    # where it did not, whose two partitions (and FedAvg runs) are then the same.
    same = []
    czero = (runs["grid"] == "covtype_rq2") & (runs["L"] == 0.0) & (runs["algorithm"] == "fedavg")
    for trial in sources.covtype_trials:
        sel = czero & (runs["trial"] == trial)
        unc = runs[sel & (runs["arm"] == "natural uncapped")].iloc[0]
        cap = runs[sel & (runs["arm"] == "natural capped")].iloc[0]
        if identical_partitions(sources.results_dir, unc.partition_id, cap.partition_id):
            same.append((trial, identical_histories(sources.results_dir, unc.run_id, cap.run_id)))
    checks["covtype_level0_identical_trials"] = same

    mu_runs, _ = load_runs(
        sources,
        grids_by_dataset={d: (g,) for d, g in MU_GRIDS.items()},
        trials_by_dataset={"cifar10": CIFAR_TUNING_TRIALS, "covtype": COVTYPE_TUNING_TRIALS},
    )
    pending = set() if sources.covtype_fedprox else {("covtype", "fedprox")}
    bundle = Bundle(sources, runs, parts, pending_runs, pending, checks, mu_runs, labels)
    load_rq1(bundle, covtype_rq1_session)
    return bundle


def load_rq1(bundle: Bundle, covtype_session: Path | None) -> None:
    """RQ1 rows: CIFAR-10 from rq1.yaml; Covtype from covtype_rq1.yaml, else the session data."""
    cifar = rq1_from_grid(bundle.sources.results_dir, "rq1")
    frames, status = [cifar], {"cifar10": "final"}
    covtype = None
    if (GRIDS / "covtype_rq1.yaml").exists():
        covtype = rq1_from_grid(bundle.sources.results_dir, "covtype_rq1")
    if covtype is not None:
        status["covtype"] = "final"
    elif covtype_session is not None:
        covtype = rq1_from_session(covtype_session)
        status["covtype"] = "provisional"
    else:
        status["covtype"] = "pending"
    if covtype is not None:
        frames.append(covtype)
    bundle.rq1 = pd.concat(frames, ignore_index=True)
    bundle.rq1_status = status


# ---------------------------------------------------------------- cell summaries
def _summ(group: pd.DataFrame, column: str) -> tuple[float, float, int]:
    return mean_sd(group[column].tolist())


def cell_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, grid, arm, L, Q, algorithm): n, trials, means and sds."""
    use = runs[~runs["merged_out"]]
    keys = ["dataset", "grid", "cell_arm", "L", "Q", "algorithm", "post_hoc"]
    out = []
    for key, g in use.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        row["n"] = len(g)
        row["trials"] = trial_set_label(g["trial"])
        for metric in (
            "final_acc",
            "best_acc",
            "rta",
            "final_balanced_acc",
            "realized_label_hd",
            "realized_quantity_hd",
            "size_weighted_label_hd",
        ):
            m, s, n = _summ(g, metric)
            row[f"{metric}_mean"], row[f"{metric}_sd"], row[f"{metric}_n"] = m, s, n
        row["best_round_mean"] = g["best_round"].mean()
        row["best_round_min"] = g["best_round"].min()
        row["best_round_max"] = g["best_round"].max()
        out.append(row)
    return pd.DataFrame(out)


def _cell(summary: pd.DataFrame, **match) -> pd.Series | None:
    sel = np.ones(len(summary), dtype=bool)
    for k, v in match.items():
        col = summary[k]
        sel &= col.isna().to_numpy() if v is None else (col == v).to_numpy()
    hit = summary[sel]
    if len(hit) > 1:
        raise ValueError(f"{len(hit)} cells match {match}")
    return None if hit.empty else hit.iloc[0]


def rq2_arms(dataset: str, level: float) -> tuple[str, ...]:
    if dataset == "cifar10" and level == 0.0:
        return (NATURAL, "pure")
    return RQ2_ARMS


def rq2_levels(bundle: Bundle, dataset: str) -> list[float]:
    sel = bundle.runs["grid"] == RQ2_GRID[dataset]
    return sorted(bundle.runs.loc[sel, "L"].unique())


def by_trial(runs: pd.DataFrame, metric: str = "final_acc", **match) -> dict[int, float]:
    sel = ~runs["merged_out"]
    for k, v in match.items():
        sel &= runs[k] == v
    hit = runs[sel]
    if hit["trial"].duplicated().any():
        raise ValueError(f"more than one run per trial for {match}")
    return dict(zip(hit["trial"], hit[metric]))


def part_means(bundle: Bundle, dataset: str, grid: str, arm: str, L, Q=None) -> dict:
    """Mean realized HD values of a cell's partitions (algorithm-independent)."""
    sel = (bundle.runs["grid"] == grid) & (bundle.runs["cell_arm"] == arm) & (bundle.runs["L"] == L)
    sel &= bundle.runs["algorithm"] == "fedavg"
    if Q is not None:
        sel &= bundle.runs["Q"] == Q
    g = bundle.runs[sel & ~bundle.runs["merged_out"]]
    return {
        "label_hd": g["realized_label_hd"].mean(),
        "qty_hd": g["realized_quantity_hd"].mean(),
        "sw_hd": g["size_weighted_label_hd"].mean(),
        "n_partitions": len(g),
    }


# ---------------------------------------------------------------- setup
def _parameter_count(config) -> int:
    from models.networks import build_model

    shape = (3, 32, 32) if config.dataset.name == "cifar10" else (config.model.in_channels,)
    model = build_model(config.model, shape, init_seed=0)
    return int(sum(p.numel() for p in model.parameters()))


def setup_table(bundle: Bundle) -> Table:
    columns = {}
    for dataset, rq2, central, rq3 in (
        ("cifar10", "rq2", "centralized", "rq3"),
        ("covtype", "covtype_rq2", "covtype_centralized", None),
    ):
        configs = grid_configs(rq2)
        c0 = configs[0]
        f = c0.federated
        mus = sorted({c.federated.mu for c in configs if c.federated.algorithm == "fedprox"})
        levels = sorted({c.partition.target_label_hd for c in configs})
        arms = []
        for c in configs:
            if _arm(c) not in arms:
                arms.append(_arm(c))
        cent = grid_configs(central)[0]
        rq3_configs = grid_configs(rq3) if rq3 else []
        q_levels = sorted({c.partition.target_quantity_hd for c in rq3_configs})
        arch = "CNN" if c0.model.arch == "cnn" else "DNN"
        optimizer = "Adam" if f.optimizer == "adam" else f.optimizer
        columns[dataset] = {
            "Model": arch,
            "Parameters": f"{_parameter_count(c0):,}",
            "Clients K": str(c0.partition.K),
            "Rounds × local epochs": f"{f.rounds} × {f.local_epochs}",
            "Batch size": str(f.batch_size),
            "Optimizer, learning rate": f"{optimizer}, {f.lr:g}",
            "FedProx μ": ", ".join(f"{m:g}" for m in mus),
            "Label levels": "/".join(_lvl(x) for x in levels),
            "Quantity levels (RQ3)": "/".join(_lvl(x) for x in q_levels) if q_levels else "--",
            "Arms, RQ2": ", ".join(arms),
            "Arms, RQ3": "combined" if rq3 else "--",
            "Centralized epochs": str(cent.federated.local_epochs),
            "Reported trials": trial_set_label(bundle.trials(dataset)),
            "Runs, RQ2": str(len(configs)),
            "Runs, RQ3": str(len(rq3_configs)) if rq3 else "--",
            "Runs, centralized": str(len(grid_configs(central))),
        }
    rows = []
    for setting in columns["cifar10"]:
        a, b = columns["cifar10"][setting], columns["covtype"][setting]
        rows.append(
            {
                "setting": setting,
                "cifar10": a,
                "covtype": b,
                "both": a,
                "row_kind": "reference" if a == b else "cell",
                "pending": "",
            }
        )
    df = pd.DataFrame(rows)
    cols = [
        Col("setting", "", "text", ("setting",), align="l"),
        Col(
            "cifar10", "CIFAR-10", "text", ("cifar10",), align="l", ref=("both",), ref_group="both"
        ),
        Col("covtype", "Covtype", "text", ("covtype",), align="l", ref=("both",), ref_group="both"),
    ]
    notes = [
        "Every value is read from the grid and config files (config/grids/rq2.yaml, rq3.yaml, "
        "centralized.yaml, covtype_rq2.yaml, covtype_centralized.yaml); parameter counts from "
        "models.networks.build_model.",
        "Runs count every trial in the grid file and both algorithms; reported trials are the "
        "trials every reported number uses. A value shared by both datasets spans both columns.",
        "Label levels are target label-HD values (HD to the global label distribution); quantity "
        "levels are target quantity-HD values (HD to uniform client sizes). Realized values are "
        "reported in the results tables.",
        "mu was tuned once per dataset on trials no reported number uses (CIFAR-10 trials 3-4, "
        "Covtype trials 5-6).",
    ]
    return Table("setup", df, cols, notes, tabcolsep_pt=4, trial_set=_trial_sets(bundle))


def _arm(config) -> str:
    from analysis.paper_data import arm_name

    return arm_name(config.partition)


def _lvl(x) -> str:
    return f"{x:.2f}" if x else "0"


def _trial_sets(bundle: Bundle) -> str:
    return f"cifar10 {trial_set_label(bundle.trials('cifar10'))}; covtype {trial_set_label(bundle.trials('covtype'))}"


# ---------------------------------------------------------------- main results
def _algo_fields(row: dict, summary, dataset, grid, arm, L, Q, prefix_metrics, bundle, pending):
    for algorithm in ALGORITHMS:
        cell = _cell(
            summary, dataset=dataset, grid=grid, cell_arm=arm, L=L, Q=Q, algorithm=algorithm
        )
        for metric, key in prefix_metrics:
            if bundle.is_pending(dataset, algorithm):
                pending.append(f"{algorithm}_{key}")
                continue
            if cell is None:
                row[f"{algorithm}_{key}_mean"] = np.nan
                continue
            row[f"{algorithm}_{key}_mean"] = cell[f"{metric}_mean"]
            row[f"{algorithm}_{key}_sd"] = cell[f"{metric}_sd"]
            row[f"{algorithm}_{key}_n"] = cell[f"{metric}_n"]


def _reference_row(summary, dataset, grid, label, metrics, section, post_hoc=False):
    cell = _cell(summary, dataset=dataset, grid=grid, algorithm="centralized")
    row = {"section": section, "row_kind": "reference", "cell": label, "arm_short": label}
    pending = []
    if cell is None:
        pending = [m for _, m in metrics]
    else:
        for metric, key in metrics:
            row[f"ref_{key}_mean"] = cell[f"{metric}_mean"]
            row[f"ref_{key}_sd"] = cell[f"{metric}_sd"]
            row[f"ref_{key}_n"] = cell[f"{metric}_n"]
    row["pending"] = ";".join(pending)
    row["post_hoc"] = post_hoc
    return row


def main_results_body(bundle: Bundle, summary: pd.DataFrame) -> Table:
    rows = []
    for dataset in ("cifar10", "covtype"):
        section = DATASET_NAMES[dataset]
        grid = RQ2_GRID[dataset]
        for L in rq2_levels(bundle, dataset):
            for arm in rq2_arms(dataset, L):
                hd = part_means(bundle, dataset, grid, arm, L)
                row = {
                    "section": section,
                    "row_kind": "cell",
                    "dataset": dataset,
                    "level": L,
                    "arm": arm,
                    "arm_short": SHORT_ARM[arm],
                    "label_hd": hd["label_hd"],
                    "qty_hd": hd["qty_hd"],
                }
                pending = []
                _algo_fields(
                    row,
                    summary,
                    dataset,
                    grid,
                    arm,
                    L,
                    None,
                    [("final_acc", "final")],
                    bundle,
                    pending,
                )
                row["pending"] = ";".join(pending)
                rows.append(row)
        rows.append(
            _reference_row(
                summary,
                dataset,
                CENTRALIZED_GRID[dataset],
                "centralized",
                [("final_acc", "final")],
                section,
            )
            | {"dataset": dataset, "level": np.nan}
        )
    df = pd.DataFrame(rows)
    cols = [
        Col(
            "level",
            "Level",
            "num",
            ("level",),
            decimals=2,
            align="l",
            ref=("arm_short",),
            ref_group="label",
        ),
        Col("arm", "Arm", "text", ("arm_short",), align="l", ref=("arm_short",), ref_group="label"),
        Col("label_hd", "label-\nHD", "num", ("label_hd",), decimals=HD),
        Col("qty_hd", "quantity-\nHD", "num", ("qty_hd",), decimals=HD),
        Col(
            "fedavg_final",
            "FedAvg",
            "pm",
            ("fedavg_final_mean", "fedavg_final_sd"),
            ACC,
            ref=("ref_final_mean", "ref_final_sd"),
            ref_group="final",
        ),
        Col(
            "fedprox_final",
            "FedProx",
            "pm",
            ("fedprox_final_mean", "fedprox_final_sd"),
            ACC,
            ref=("ref_final_mean", "ref_final_sd"),
            ref_group="final",
        ),
    ]
    notes = [
        "Final-round test accuracy, mean ± sample sd over trials; "
        + bundle.trial_note("cifar10", "covtype")
        + ".",
        "Level: the target label-HD of the cell. label-HD (HD to the global label distribution) and "
        "quantity-HD (HD to uniform client sizes): mean realized values over the cell's partitions.",
        "Arms: uncapped and capped are the natural label-skew arms (Dirichlet per class) without and "
        "with the per-client size cap; pure holds client sizes equal. On CIFAR-10 at level 0 the two "
        "natural arms are the same partition and the same runs (checked trial by trial), so they "
        "are one row, natural, counted once. On Covtype they differ at level 0 and are both shown.",
        "Centralized: the pooled training set, 10 epochs, one run per trial; its value spans the "
        "algorithm columns.",
        "FedAvg RTA is in the appendix tables (dropped here for width).",
    ]
    if bundle.covtype_level0_note():
        notes.append(bundle.covtype_level0_note())
    if bundle.is_pending("covtype", "fedprox"):
        notes.append("Covtype FedProx: pending (runs still training).")
    return Table(
        "main_results_body", df, cols, notes, tabcolsep_pt=4, trial_set=_trial_sets(bundle)
    )


def _lq_label(L, Q=None) -> str:
    return f"{L:.2f}" + ("" if Q is None or pd.isna(Q) else f" × {Q:.2f}")


def main_results_full(bundle: Bundle, summary: pd.DataFrame) -> Table:
    metrics = [
        ("final_acc", "final"),
        ("best_acc", "best"),
        ("rta", "rta"),
        ("final_balanced_acc", "bal"),
    ]
    rows = []

    def cell_row(section, dataset, grid, arm, L, Q, label):
        hd = part_means(bundle, dataset, grid, arm, L, Q)
        row = {
            "section": section,
            "row_kind": "cell",
            "dataset": dataset,
            "grid": grid,
            "cell": label,
            "level": L,
            "quantity_level": Q,
            "label_hd": hd["label_hd"],
            "qty_hd": hd["qty_hd"],
            "sw_hd": hd["sw_hd"],
        }
        pending = []
        _algo_fields(row, summary, dataset, grid, arm, L, Q, metrics, bundle, pending)
        if dataset != "covtype":
            for algorithm in ALGORITHMS:
                row.pop(f"{algorithm}_bal_mean", None)
        row["pending"] = ";".join(
            p for p in pending if not (dataset != "covtype" and p.endswith("_bal"))
        )
        return row

    for L in rq2_levels(bundle, "cifar10"):
        for arm in rq2_arms("cifar10", L):
            rows.append(
                cell_row("CIFAR-10, RQ2", "cifar10", "rq2", arm, L, None, f"{arm}, {L:.2f}")
            )
    order = bundle.runs[bundle.runs["grid"] == "rq2_class_order"]
    for L in sorted(order["L"].unique()):
        arm = "natural capped, reversed order"
        rows.append(
            cell_row(
                "CIFAR-10, class-order check",
                "cifar10",
                "rq2_class_order",
                arm,
                L,
                None,
                f"{arm}, {L:.2f}",
            )
        )
    rq3 = bundle.runs[bundle.runs["grid"] == "rq3"]
    for L in sorted(rq3["L"].unique()):
        for Q in sorted(rq3["Q"].unique()):
            rows.append(
                cell_row(
                    "CIFAR-10, RQ3",
                    "cifar10",
                    "rq3",
                    "combined",
                    L,
                    Q,
                    f"combined, {L:.2f} × {Q:.2f}",
                )
            )
    rows.append(
        _reference_row(
            summary,
            "cifar10",
            "centralized",
            "centralized, 10 epochs",
            metrics,
            "CIFAR-10, centralized",
        )
        | {"dataset": "cifar10"}
    )
    for L in rq2_levels(bundle, "covtype"):
        for arm in rq2_arms("covtype", L):
            rows.append(
                cell_row("Covtype, RQ2", "covtype", "covtype_rq2", arm, L, None, f"{arm}, {L:.2f}")
            )
    rows.append(
        _reference_row(
            summary,
            "covtype",
            "covtype_centralized",
            "centralized, 10 epochs",
            metrics,
            "Covtype, centralized",
        )
        | {"dataset": "covtype"}
    )
    rows.append(
        _reference_row(
            summary,
            "covtype",
            "covtype_centralized_steps",
            "centralized, 400 epochs (post-hoc sensitivity)",
            metrics,
            "Covtype, centralized",
            post_hoc=True,
        )
        | {"dataset": "covtype"}
    )
    df = pd.DataFrame(rows)
    for key in ("ref_bal_mean", "ref_bal_sd"):
        if key in df:
            df.loc[df["dataset"] != "covtype", key] = np.nan

    def pm(key, header, algorithm, decimals, group):
        return Col(
            f"{algorithm}_{key}",
            header,
            "pm",
            (f"{algorithm}_{key}_mean", f"{algorithm}_{key}_sd"),
            decimals,
            ref=(f"ref_{key}_mean", f"ref_{key}_sd"),
            ref_group=group,
        )

    cell = Col("cell", "Cell", "text", ("cell",), align="l")
    accuracy_cols = [
        cell,
        Col("label_hd", "label-\nHD", "num", ("label_hd",), decimals=HD),
        Col("qty_hd", "quantity-\nHD", "num", ("qty_hd",), decimals=HD),
        Col("sw_hd", "size-weighted\nlabel-HD", "num", ("sw_hd",), decimals=HD),
        pm("final", "FedAvg\nfinal", "fedavg", ACC, "final"),
        pm("final", "FedProx\nfinal", "fedprox", ACC, "final"),
        pm("best", "FedAvg\nbest", "fedavg", ACC, "best"),
    ]
    rta_cols = [
        cell,
        pm("rta", "FedAvg\nRTA", "fedavg", RTA, "rta"),
        pm("rta", "FedProx\nRTA", "fedprox", RTA, "rta"),
        pm("bal", "FedAvg balanced\nacc. (ours)", "fedavg", ACC, "bal"),
        pm("bal", "FedProx balanced\nacc. (ours)", "fedprox", ACC, "bal"),
    ]
    notes = [
        "Mean ± sample sd over trials; " + bundle.trial_note("cifar10", "covtype") + ".",
        "Final and best: final-round and best-round test accuracy. RTA: first round reaching 90% of "
        "the run's own best accuracy, mean of the per-run values (in epochs for the centralized rows).",
        "label-HD, quantity-HD and size-weighted label-HD: mean realized values over the cell's "
        "partitions. Size-weighted label-HD weights each client's HD to the global label "
        "distribution by its size; label-HD is the unweighted mean over clients.",
        "Balanced accuracy (mean per-class recall) is our addition, Covtype only; accuracy is the "
        "source paper's metric.",
        "CIFAR-10 level 0: the two natural arms are one partition, one row (natural). Class-order "
        "check: the capped arm with the classes split in reversed order, FedAvg only.",
        "Centralized rows: the pooled training set, one run per trial; values span the algorithm "
        "columns; the 400-epoch Covtype row is a post-hoc sensitivity check added after seeing the "
        "10-epoch result, not the pre-specified baseline.",
    ]
    if bundle.covtype_level0_note():
        notes.append(bundle.covtype_level0_note())
    if bundle.is_pending("covtype", "fedprox"):
        notes.append("Covtype FedProx: pending (runs still training).")
    if any(g == "covtype_centralized_steps" for g, *_ in bundle.pending_runs):
        notes.append("Covtype centralized 400 epochs: pending (not trained; grid blocked).")
    split = (
        "The full results are one dataframe, written as two LaTeX fragments for width: accuracy "
        "(main_results_full_accuracy) and RTA with balanced accuracy (main_results_full_rta_balanced); "
        "both CSVs hold every column."
    )
    return [
        Table(
            "main_results_full_accuracy",
            df,
            accuracy_cols,
            notes + [split],
            trial_set=_trial_sets(bundle),
        ),
        Table(
            "main_results_full_rta_balanced",
            df,
            rta_cols,
            notes + [split],
            trial_set=_trial_sets(bundle),
        ),
    ]


# ---------------------------------------------------------------- RQ2 paired gaps (accuracy and RTA)
def rq2_pairs(dataset: str, level: float):
    if dataset == "cifar10" and level == 0.0:
        return (("pure", NATURAL),)
    return RQ2_PAIRS


def rq2_gaps(bundle: Bundle, metric: str = "final_acc") -> pd.DataFrame:
    """One row per dataset, level, algorithm and arm pair: the paired gap a - b."""
    rows = []
    for dataset in ("cifar10", "covtype"):
        grid = RQ2_GRID[dataset]
        for L in rq2_levels(bundle, dataset):
            for algorithm in ALGORITHMS:
                for a, b in rq2_pairs(dataset, L):
                    ha = part_means(bundle, dataset, grid, a, L)
                    hb = part_means(bundle, dataset, grid, b, L)
                    row = {
                        "section": DATASET_NAMES[dataset],
                        "dataset": dataset,
                        "level": L,
                        "algorithm": algorithm,
                        "algorithm_name": ALGORITHM_NAMES[algorithm],
                        "arm_a": a,
                        "arm_b": b,
                        "comparison": PAIR_NAMES[(a, b)],
                        "qty_hd_a": ha["qty_hd"],
                        "qty_hd_b": hb["qty_hd"],
                        "sw_hd_a": ha["sw_hd"],
                        "sw_hd_b": hb["sw_hd"],
                        "label_hd_a": ha["label_hd"],
                        "label_hd_b": hb["label_hd"],
                        "pending": "",
                    }
                    if bundle.is_pending(dataset, algorithm):
                        row["pending"] = "gap;signs;a;b"
                    else:
                        va = by_trial(
                            bundle.runs, metric, grid=grid, cell_arm=a, L=L, algorithm=algorithm
                        )
                        vb = by_trial(
                            bundle.runs, metric, grid=grid, cell_arm=b, L=L, algorithm=algorithm
                        )
                        p = paired(va, vb)
                        ma, sa, _ = mean_sd(va.values())
                        mb, sb, _ = mean_sd(vb.values())
                        row.update(
                            {
                                "a_mean": ma,
                                "a_sd": sa,
                                "b_mean": mb,
                                "b_sd": sb,
                                "gap_mean": p.mean,
                                "gap_sd": p.sd,
                                "n": p.n,
                                "n_positive": p.n_positive,
                                "n_negative": p.n_negative,
                                "n_zero": p.n_zero,
                                "trials": trial_set_label(p.trials),
                                "per_trial": " ".join(f"{d:+.4f}" for d in p.differences),
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def rq2_gaps_table(bundle: Bundle, gaps: pd.DataFrame) -> Table:
    cols = [
        Col("level", "Level", "num", ("level",), decimals=2, align="l"),
        Col("algorithm", "Algorithm", "text", ("algorithm_name",), align="l"),
        Col("comparison", "Comparison a − b", "text", ("comparison",), align="l"),
        Col("gap", "Gap\n(final acc.)", "pm", ("gap_mean", "gap_sd"), ACC),
        Col("signs", "Trials\n+/−", "signs", ("n_positive", "n_negative", "n_zero")),
        Col("qa", "quantity-\nHD a", "num", ("qty_hd_a",), decimals=HD),
        Col("qb", "quantity-\nHD b", "num", ("qty_hd_b",), decimals=HD),
        Col("sa", "size-weighted\nlabel-HD a", "num", ("sw_hd_a",), decimals=HD),
        Col("sb", "size-weighted\nlabel-HD b", "num", ("sw_hd_b",), decimals=HD),
    ]
    notes = [
        "Gap a − b in final-round test accuracy, paired by trial: mean ± sample sd of the per-trial "
        "differences, and the number of trials with a positive and a negative difference; "
        + bundle.trial_note("cifar10", "covtype")
        + ".",
        "a and b are the two arms of the comparison, in that order; their quantity-HD and "
        "size-weighted label-HD are the mean realized values over the cell's partitions.",
        "CIFAR-10 level 0: the two natural arms are one partition, so only pure − natural is shown.",
        "Trials +/−: positive and negative per-trial differences; '=' counts trials with a "
        "difference of exactly zero.",
    ]
    if bundle.covtype_level0_note():
        notes.append(
            bundle.covtype_level0_note() + " Its capped − uncapped difference is exactly 0."
        )
    if bundle.is_pending("covtype", "fedprox"):
        notes.append(
            "Covtype FedProx rows: gaps pending (runs still training); HD values are the partitions'."
        )
    return Table("rq2_paired_gaps", gaps, cols, notes, trial_set=_trial_sets(bundle))


def rq2_rta_table(bundle: Bundle, gaps: pd.DataFrame) -> Table:
    cols = [
        Col("level", "Level", "num", ("level",), decimals=2, align="l"),
        Col("algorithm", "Algorithm", "text", ("algorithm_name",), align="l"),
        Col("comparison", "Comparison a − b", "text", ("comparison",), align="l"),
        Col("a", "RTA a", "pm", ("a_mean", "a_sd"), RTA),
        Col("b", "RTA b", "pm", ("b_mean", "b_sd"), RTA),
        Col("gap", "Gap (RTA)", "pm", ("gap_mean", "gap_sd"), RTA),
        Col("signs", "Trials\n+/−", "signs", ("n_positive", "n_negative", "n_zero")),
    ]
    notes = [
        "RTA: first round reaching 90% of the run's own best accuracy. Gap a − b paired by trial: "
        "mean ± sample sd of the per-trial differences and the trials with each sign; "
        + bundle.trial_note("cifar10", "covtype")
        + ". A positive gap: arm a takes more rounds. '=' counts trials with equal RTA.",
    ]
    if bundle.is_pending("covtype", "fedprox"):
        notes.append("Covtype FedProx rows: pending (runs still training).")
    return Table("rq2_rta_paired", gaps, cols, notes, trial_set=_trial_sets(bundle))


# ---------------------------------------------------------------- RQ3
def rq3_acc(bundle: Bundle, algorithm: str, metric: str = "final_acc") -> dict:
    r = bundle.runs[(bundle.runs["grid"] == "rq3") & (bundle.runs["algorithm"] == algorithm)]
    return {(L, Q, t): v for L, Q, t, v in zip(r["L"], r["Q"], r["trial"], r[metric])}


def rq3_levels(bundle: Bundle) -> tuple[list, list]:
    r = bundle.runs[bundle.runs["grid"] == "rq3"]
    return sorted(r["L"].unique()), sorted(r["Q"].unique())


def rq3_additivity(bundle: Bundle, metric: str = "final_acc") -> pd.DataFrame:
    Ls, Qs = rq3_levels(bundle)
    rows = []
    for algorithm in ALGORITHMS:
        acc = rq3_acc(bundle, algorithm, metric)
        inter = interactions(acc, Ls, Qs)
        for (L, Q), per_trial in inter.items():
            d = deltas(acc, L, Q, Ls[0], Qs[0])
            m, s, n = mean_sd(per_trial.values())
            hd = part_means(bundle, "cifar10", "rq3", "combined", L, Q)
            rows.append(
                {
                    "section": ALGORITHM_NAMES[algorithm],
                    "algorithm": algorithm,
                    "level": L,
                    "quantity_level": Q,
                    "label_hd": hd["label_hd"],
                    "qty_hd": hd["qty_hd"],
                    "sw_hd": hd["sw_hd"],
                    "delta_combined": mean_sd(d["combined"].values())[0],
                    "delta_label": mean_sd(d["label"].values())[0],
                    "delta_quantity": mean_sd(d["quantity"].values())[0],
                    "interaction_mean": m,
                    "interaction_sd": s,
                    "n": n,
                    "n_negative": sum(v < 0 for v in per_trial.values()),
                    "per_trial": " ".join(f"{t}:{v:+.4f}" for t, v in sorted(per_trial.items())),
                    "pending": "",
                }
            )
    return pd.DataFrame(rows)


def rq3_additivity_table(bundle: Bundle, add: pd.DataFrame) -> Table:
    cols = [
        Col("level", "Label\nlevel", "num", ("level",), decimals=2, align="l"),
        Col("q", "Quantity\nlevel", "num", ("quantity_level",), decimals=2, align="l"),
        Col("label_hd", "label-\nHD", "num", ("label_hd",), decimals=HD),
        Col("qty_hd", "quantity-\nHD", "num", ("qty_hd",), decimals=HD),
        Col("sw_hd", "size-weighted\nlabel-HD", "num", ("sw_hd",), decimals=HD),
        Col("dc", "Δ\ncombined", "num", ("delta_combined",), decimals=ACC),
        Col("dl", "Δ\nlabel-only", "num", ("delta_label",), decimals=ACC),
        Col("dq", "Δ\nquantity-only", "num", ("delta_quantity",), decimals=ACC),
        Col("int", "Interaction", "pm", ("interaction_mean", "interaction_sd"), ACC),
        Col("neg", "Negative\ntrials", "count", ("n_negative", "n")),
    ]
    notes = [
        "CIFAR-10, RQ3 (combined partitions), final-round test accuracy; "
        + bundle.trial_note("cifar10")
        + ".",
        "Per trial, against that trial's (0, 0) corner: Δ combined = acc(L, Q) − acc(0, 0), Δ "
        "label-only = acc(L, 0) − acc(0, 0), Δ quantity-only = acc(0, Q) − acc(0, 0); "
        "interaction = Δ combined − Δ label-only − Δ quantity-only. The Δ columns are means over "
        "trials; the interaction is mean ± sample sd over trials, with the number of trials in which "
        "it is negative.",
        "label-HD, quantity-HD, size-weighted label-HD: mean realized values of the cell's partitions.",
    ]
    return Table("rq3_additivity", add, cols, notes, trial_set=_trial_sets(bundle))


def rq3_rta_table(bundle: Bundle, summary: pd.DataFrame, rta_add: pd.DataFrame) -> Table:
    Ls, Qs = rq3_levels(bundle)
    rows = []
    for algorithm in ALGORITHMS:
        for L in Ls:
            for Q in Qs:
                c = _cell(
                    summary,
                    dataset="cifar10",
                    grid="rq3",
                    cell_arm="combined",
                    L=L,
                    Q=Q,
                    algorithm=algorithm,
                )
                hit = rta_add[
                    (rta_add["algorithm"] == algorithm)
                    & (rta_add["level"] == L)
                    & (rta_add["quantity_level"] == Q)
                ]
                rows.append(
                    {
                        "section": ALGORITHM_NAMES[algorithm],
                        "algorithm": algorithm,
                        "level": L,
                        "quantity_level": Q,
                        "rta_mean": c["rta_mean"],
                        "rta_sd": c["rta_sd"],
                        "n": c["rta_n"],
                        "best_round_mean": c["best_round_mean"],
                        "best_round_range": f"{int(c['best_round_min'])}–{int(c['best_round_max'])}",
                        "rta_interaction_mean": (
                            hit["interaction_mean"].iloc[0] if len(hit) else np.nan
                        ),
                        "rta_interaction_sd": hit["interaction_sd"].iloc[0] if len(hit) else np.nan,
                        "rta_interaction_neg": hit["n_negative"].iloc[0] if len(hit) else np.nan,
                        "pending": "",
                    }
                )
    df = pd.DataFrame(rows)
    cols = [
        Col("level", "Label\nlevel", "num", ("level",), decimals=2, align="l"),
        Col("q", "Quantity\nlevel", "num", ("quantity_level",), decimals=2, align="l"),
        Col("rta", "RTA", "pm", ("rta_mean", "rta_sd"), RTA),
        Col("best", "Best round,\nmean", "num", ("best_round_mean",), decimals=1),
        Col("range", "Best round,\nrange", "text", ("best_round_range",)),
        Col("int", "RTA\ninteraction", "pm", ("rta_interaction_mean", "rta_interaction_sd"), RTA),
        Col("neg", "Negative\ntrials", "count", ("rta_interaction_neg", "n")),
    ]
    notes = [
        "CIFAR-10, RQ3; RTA = first round reaching 90% of the run's own best accuracy, mean ± sample "
        "sd of the per-run values; " + bundle.trial_note("cifar10") + ".",
        "Best round: the round of the run's highest test accuracy (rounds 1-40). Where runs peak "
        "early, RTA measures the climb to that early peak.",
        "RTA interaction: the additivity contrast of the accuracy table, applied to RTA (non-edge cells).",
    ]
    return Table("rq3_rta", df, cols, notes, trial_set=_trial_sets(bundle))


def fedprox_wins(bundle: Bundle) -> dict:
    """RQ3: (cell, trial) pairs where FedProx's final accuracy is above FedAvg's."""
    a, p = rq3_acc(bundle, "fedavg"), rq3_acc(bundle, "fedprox")
    keys = sorted(set(a) & set(p))
    wins = sum(p[k] > a[k] for k in keys)
    ties = sum(p[k] == a[k] for k in keys)
    return {"wins": wins, "ties": ties, "pairs": len(keys)}


# ---------------------------------------------------------------- RQ3 regression
def rq3_regression(bundle: Bundle) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """(interaction table rows, coefficient rows, LOF per algorithm, fits for the residual plot)."""
    Ls, Qs = rq3_levels(bundle)
    model_free = rq3_additivity(bundle)
    rows, coef_rows, lof, fits = [], [], {}, {}
    for algorithm in ALGORITHMS:
        r = bundle.runs[(bundle.runs["grid"] == "rq3") & (bundle.runs["algorithm"] == algorithm)]
        data = pd.DataFrame(
            {
                "acc": r["final_acc"].to_numpy(),
                "L": r["realized_label_hd"].to_numpy(),
                "L_level": r["L"].to_numpy(),
                "Q_level": r["Q"].to_numpy(),
                "trial": r["trial"].to_numpy(),
            }
        )
        cell_L = data.groupby(["L_level", "Q_level"])["L"].mean().to_dict()
        trials = sorted(data["trial"].unique())
        quad = fit_regression(data, quadratic=True)
        lin = fit_regression(data, quadratic=False)
        fits[algorithm] = (data, quad)
        lof[algorithm] = lack_of_fit(data) | {
            "n": quad.n,
            "p_quadratic": quad.p,
            "rss_quadratic": quad.rss,
            "p_linear": lin.p,
            "rss_linear": lin.rss,
        }
        for L in Ls[1:]:
            for Q in Qs[1:]:
                mf = model_free[
                    (model_free["algorithm"] == algorithm)
                    & (model_free["level"] == L)
                    & (model_free["quantity_level"] == Q)
                ].iloc[0]
                rows.append(
                    {
                        "section": ALGORITHM_NAMES[algorithm],
                        "algorithm": algorithm,
                        "level": L,
                        "quantity_level": Q,
                        "cell_label_hd": cell_L[(L, Q)],
                        "label_only_label_hd": cell_L[(L, Qs[0])],
                        "model_free_mean": mf["interaction_mean"],
                        "model_free_sd": mf["interaction_sd"],
                        "model_quadratic": model_interaction(quad, cell_L, L, Q, Qs, trials, True),
                        "model_linear": model_interaction(lin, cell_L, L, Q, Qs, trials, False),
                        "n": quad.n,
                        "pending": "",
                    }
                )
        for name, value in zip(quad.names, quad.coef):
            coef_rows.append(
                {
                    "section": ALGORITHM_NAMES[algorithm],
                    "algorithm": algorithm,
                    "model": "quadratic in L",
                    "term": name,
                    "estimate": value,
                    "pending": "",
                }
            )
        for name, value in zip(lin.names, lin.coef):
            coef_rows.append(
                {
                    "section": ALGORITHM_NAMES[algorithm],
                    "algorithm": algorithm,
                    "model": "linear in L",
                    "term": name,
                    "estimate": value,
                    "pending": "",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(coef_rows), lof, fits


def rq3_regression_tables(bundle: Bundle, rows, coefs, lof) -> list[Table]:
    cols = [
        Col("level", "Label\nlevel", "num", ("level",), decimals=2, align="l"),
        Col("q", "Quantity\nlevel", "num", ("quantity_level",), decimals=2, align="l"),
        Col("lc", "Cell\nlabel-HD", "num", ("cell_label_hd",), decimals=HD),
        Col("ll", "Label-only\nlabel-HD", "num", ("label_only_label_hd",), decimals=HD),
        Col("mf", "Interaction,\nmodel-free", "pm", ("model_free_mean", "model_free_sd"), ACC),
        Col("mq", "Model-implied\n(L + L²)", "num", ("model_quadratic",), decimals=ACC),
        Col("ml", "Model-implied\n(L)", "num", ("model_linear",), decimals=ACC),
    ]
    n = len(bundle.trials("cifar10"))
    notes = [
        f"CIFAR-10, RQ3, final-round test accuracy, {12 * n} runs per algorithm ({n} trials "
        f"{trial_set_label(bundle.trials('cifar10'))} × 12 cells, every run kept, unweighted).",
        "Model: acc ~ (L + L²) × C(quantity level) + C(trial), L the run's realized label-HD, ordinary "
        "least squares. Model-implied interaction at a cell: [m(Lc, Q) − m(Lq, Q)] − [m(Lc, 0) − "
        "m(L0, 0)], m the fitted mean with trial effects averaged, Lc the cell's mean realized "
        "label-HD, Lq the quantity-only cell's (label level 0), L0 the corner's: the interaction at "
        "matched label-HD. Model-free: the additivity table's value, which compares with the "
        "label-only cell at its own, lower, label-HD.",
        "Model-implied (L): the same with the L² terms dropped (sensitivity check).",
    ]
    for algorithm, v in lof.items():
        notes.append(
            f"{ALGORITHM_NAMES[algorithm]}: lack of fit against the cell-means model C(label level) × "
            f"C(quantity level) + C(trial), with L replaced by its cell mean so the models nest: "
            f"F({v['df_num']}, {v['df_den']}) = {v['F']:.3f}; RSS {v['rss_regression_cell_L']:.6f} "
            f"(regression, {v['p_regression']} parameters) vs {v['rss_cell_means']:.6f} (cell means, "
            f"{v['p_cell_means']}). Per-run L fit: RSS {v['rss_quadratic']:.6f} ({v['p_quadratic']} "
            f"parameters); linear-L fit: RSS {v['rss_linear']:.6f} ({v['p_linear']}). Descriptive; no "
            "p-value."
        )
    t1 = Table("rq3_regression", rows, cols, notes, trial_set=_trial_sets(bundle))
    t2 = Table(
        "rq3_regression_coefficients",
        coefs,
        [
            Col("model", "Model", "text", ("model",), align="l"),
            Col("term", "Term", "text", ("term",), align="l"),
            Col("estimate", "Estimate", "num", ("estimate",), decimals=4),
        ],
        [
            "Least-squares estimates of the RQ3 regression; reference levels: quantity level 0, the "
            "lowest trial. Descriptive."
        ],
        trial_set=_trial_sets(bundle),
    )
    return [t1, t2]


# ---------------------------------------------------------------- mu sweep
def mu_sweep(bundle: Bundle) -> tuple[Table, dict]:
    rows, selected = [], {}
    for dataset in ("cifar10", "covtype"):
        r = bundle.mu_runs[bundle.mu_runs["dataset"] == dataset]
        means = {}
        for mu, g in sorted(
            r.groupby(r["mu"].where(r["algorithm"] == "fedprox", 0.0)), key=lambda kv: kv[0]
        ):
            m, s, n = mean_sd(g["final_acc"])
            means[mu] = m
            rows.append(
                {
                    "section": DATASET_NAMES[dataset],
                    "dataset": dataset,
                    "mu": mu,
                    "mu_label": "0 (FedAvg)" if mu == 0 else f"{mu:g}",
                    "per_trial": " ".join(
                        f"{t}:{v:.4f}" for t, v in zip(g["trial"], g["final_acc"])
                    ),
                    "trials": trial_set_label(g["trial"]),
                    "mean": m,
                    "sd": s,
                    "n": n,
                    "pending": "",
                }
            )
        top = max(means.values())
        chosen = min(mu for mu, m in means.items() if m >= top - 0.01)
        selected[dataset] = chosen
        for row in rows:
            if row["dataset"] == dataset:
                row["selected"] = "selected" if row["mu"] == chosen else ""
    df = pd.DataFrame(rows)
    cols = [
        Col("mu", "μ", "text", ("mu_label",), align="l"),
        Col("trials", "Trials", "text", ("trials",), align="l"),
        Col("acc", "Final acc.", "pm", ("mean", "sd"), ACC),
        Col("sel", "", "text", ("selected",), align="l"),
    ]
    notes = [
        "Tuning runs, not reported trials: CIFAR-10 at trials 3-4 (combined, label-HD 0.45 × "
        "quantity-HD 0.35), Covtype at trials 5-6 (pure, label-HD 0.35). These are the only outputs "
        "that show those trials; no reported number uses them.",
        "Rule fixed before the runs: the mu with the highest mean final accuracy over the two trials "
        "(FedAvg as mu = 0); of those within 0.01 of the top mean, the smallest.",
    ]
    return Table("mu_sweep", df, cols, notes, trial_set="cifar10 {3,4}; covtype {5,6}"), selected


# ---------------------------------------------------------------- RQ1
def rq1_summary(bundle: Bundle) -> pd.DataFrame:
    rows = []
    for (dataset, arm, K, alpha), g in bundle.rq1.groupby(
        ["dataset", "arm", "K", "alpha"], sort=False
    ):
        ok = g[g["status"] == "generated"]
        lm, ls, ln = mean_sd(ok["realized_label_hd"])
        qm, qs, _ = mean_sd(ok["realized_quantity_hd"])
        rows.append(
            {
                "section": DATASET_NAMES[dataset]
                + ("" if bundle.rq1_status.get(dataset) == "final" else " (PROVISIONAL)"),
                "dataset": dataset,
                "arm": arm,
                "K": K,
                "alpha": alpha,
                "n_total": len(g),
                "n_generated": len(ok),
                "n_infeasible": int((g["status"] == "infeasible_min_size").sum()),
                "label_hd_mean": lm,
                "label_hd_sd": ls,
                "qty_hd_mean": qm,
                "qty_hd_sd": qs,
                "label_pairwise_mean": ok["realized_label_hd_pairwise"].astype(float).mean(),
                "qty_pairwise_mean": ok["realized_quantity_hd_pairwise"].astype(float).mean(),
                "max_size_mean": ok["max_client_size"].astype(float).mean(),
                "mean_size_mean": ok["mean_client_size"].astype(float).mean(),
                "redraws_mean": g["n_redraws"].astype(float).mean(),
                "redraws_max": g["n_redraws"].astype(float).max(),
                "source": g["source"].iloc[0],
                "status": bundle.rq1_status.get(dataset),
                "pending": "",
            }
        )
    df = pd.DataFrame(rows)
    arm_order = {"natural uncapped": 0, "natural capped": 1, "natural per-client": 2}
    df["_o"] = df["arm"].map(arm_order)
    return df.sort_values(["dataset", "K", "_o", "alpha"]).drop(columns="_o").reset_index(drop=True)


def rq1_summary_table(bundle: Bundle, s: pd.DataFrame) -> Table:
    s = s.copy()
    s["section"] = s["section"] + ", K = " + s["K"].astype(str)
    s["n_label"] = [f"{a}/{b}" for a, b in zip(s["n_generated"], s["n_total"])]
    cols = [
        Col("arm", "Variant", "text", ("arm",), align="l"),
        Col("alpha", "α", "num", ("alpha",), decimals=2, align="l"),
        Col("n", "Generated", "text", ("n_label",)),
        Col("lhd", "label-HD", "pm", ("label_hd_mean", "label_hd_sd"), HD),
        Col("qhd", "quantity-HD", "pm", ("qty_hd_mean", "qty_hd_sd"), HD),
        Col("max", "Max client\nsize", "num", ("max_size_mean",), decimals=0),
        Col("mean", "Mean client\nsize", "num", ("mean_size_mean",), decimals=0),
        Col("red", "Redraws,\nmean", "num", ("redraws_mean",), decimals=1),
        Col("redmax", "Redraws,\nmax", "num", ("redraws_max",), decimals=0),
    ]
    notes = [
        "RQ1: natural label-skew partitions (no training). Mean ± sample sd over the generated "
        "partitions of each cell; CIFAR-10 3 trials (0-2, config/grids/rq1.yaml), Covtype 20 trials "
        "(0-19). Client sizes are means over trials of each partition's max and mean client size.",
        "Generated: partitions drawn of those attempted; the others ran out of their 1000-redraw "
        "min-size budget (min_size 10) and have no partition.",
        "label-HD: HD to the global label distribution; quantity-HD: HD to uniform client sizes.",
    ]
    if bundle.rq1_status.get("covtype") == "provisional":
        notes.append(
            "Covtype rows PROVISIONAL: partition measurements of 2026-11-08 built through the "
            "partitioner from configs but not recorded through the runner (uncommitted tree); "
            "replaced by config/grids/covtype_rq1.yaml once generated."
        )
    return Table(
        "rq1_partition_summary", s, cols, notes, trial_set="cifar10 {0,1,2}; covtype {0-19}"
    )


def _quantity_curve(dataset: str):
    from partitioning.calibration import load_calibration

    curve_id = CIFAR_QUANTITY_CURVE if dataset == "cifar10" else COVTYPE_QUANTITY_CURVE
    return None if curve_id is None else load_calibration(curve_id, CALIBRATION_DIR)


def rq1_max_incidental(bundle: Bundle, s: pd.DataFrame) -> pd.DataFrame:
    from partitioning.calibration import beta_for_target

    rows = []
    for (dataset, arm), g in s[s["K"] == 30].groupby(["dataset", "arm"], sort=False):
        top = g.loc[g["qty_hd_mean"].idxmax()]
        curve = _quantity_curve(dataset)
        beta, pending = np.nan, ""
        if curve is None:
            pending = "beta"
        else:
            beta = beta_for_target(curve, float(top["qty_hd_mean"]))
        rows.append(
            {
                "section": top["section"],
                "dataset": dataset,
                "arm": arm,
                "alpha": top["alpha"],
                "qty_hd_mean": top["qty_hd_mean"],
                "qty_hd_sd": top["qty_hd_sd"],
                "label_hd_mean": top["label_hd_mean"],
                "n": top["n_generated"],
                "beta_equivalent": beta,
                "curve": curve["calibration_id"] if curve else None,
                "pending": pending,
            }
        )
    return pd.DataFrame(rows)


def rq1_max_table(bundle: Bundle, m: pd.DataFrame) -> Table:
    cols = [
        Col("arm", "Variant", "text", ("arm",), align="l"),
        Col("alpha", "α at the\nmaximum", "num", ("alpha",), decimals=2),
        Col("qhd", "Max incidental\nquantity-HD", "pm", ("qty_hd_mean", "qty_hd_sd"), HD),
        Col("lhd", "label-HD\nthere", "num", ("label_hd_mean",), decimals=HD),
        Col("beta", "quantity-only\nβ", "num", ("beta_equivalent",), decimals=4),
    ]
    notes = [
        "K = 30. The largest mean incidental quantity-HD over the RQ1 alpha values, per variant; "
        "CIFAR-10 3 trials, Covtype 20.",
        "quantity-only β: the β of deliberate quantity skew (quantity-only partitions) whose mean "
        "quantity-HD equals that value, by `partitioning.calibration.beta_for_target` (log-linear "
        f"interpolation, no refinement) on the 20-trial curve calibration/{CIFAR_QUANTITY_CURVE}.json "
        "for CIFAR-10.",
    ]
    if COVTYPE_QUANTITY_CURVE is None:
        notes.append("Covtype β: pending; no Covtype quantity-only curve exists yet.")
    if bundle.rq1_status.get("covtype") == "provisional":
        notes.append("Covtype rows PROVISIONAL (see the RQ1 summary table).")
    return Table(
        "rq1_max_incidental_quantity", m, cols, notes, trial_set="cifar10 {0,1,2}; covtype {0-19}"
    )


def rq1_matched(bundle: Bundle, s: pd.DataFrame, targets=(0.25, 0.45, 0.70)) -> pd.DataFrame:
    """Incidental quantity-HD at matched label-HD, by linear interpolation along each variant."""
    rows = []
    for (dataset, arm), g in s[s["K"] == 30].groupby(["dataset", "arm"], sort=False):
        g = g.dropna(subset=["label_hd_mean"]).sort_values("label_hd_mean")
        x, y = g["label_hd_mean"].to_numpy(), g["qty_hd_mean"].to_numpy()
        for t in targets:
            inside = x[0] <= t <= x[-1]
            rows.append(
                {
                    "section": DATASET_NAMES[dataset],
                    "dataset": dataset,
                    "arm": arm,
                    "label_hd": t,
                    "qty_hd": float(np.interp(t, x, y)) if inside else np.nan,
                    "pending": "",
                }
            )
    df = pd.DataFrame(rows)
    wide = df.pivot_table(
        index=["arm", "label_hd"], columns="dataset", values="qty_hd", dropna=False
    ).reset_index()
    order = {"natural uncapped": 0, "natural capped": 1, "natural per-client": 2}
    wide = wide.sort_values(
        ["arm", "label_hd"], key=lambda s: s.map(order) if s.name == "arm" else s
    )
    wide["section"] = "K = 30"
    wide["pending"] = ""
    return wide.reset_index(drop=True)


def rq1_matched_table(bundle: Bundle, w: pd.DataFrame) -> Table:
    cols = [
        Col("arm", "Variant", "text", ("arm",), align="l"),
        Col("lhd", "label-HD", "num", ("label_hd",), decimals=2),
        Col("c", "CIFAR-10\nquantity-HD", "num", ("cifar10",), decimals=HD),
        Col("v", "Covtype\nquantity-HD", "num", ("covtype",), decimals=HD),
    ]
    notes = [
        "Incidental quantity-HD at matched label-HD, K = 30: linear interpolation of the mean "
        "quantity-HD against the mean label-HD along each variant's alpha points (RQ1 summary table); "
        "'--' where the variant's label-HD range does not reach the value.",
    ]
    if bundle.rq1_status.get("covtype") == "provisional":
        notes.append("Covtype column PROVISIONAL.")
    return Table(
        "rq1_matched_label_hd", w, cols, notes, trial_set="cifar10 {0,1,2}; covtype {0-19}"
    )


def rq1_pairwise_table(bundle: Bundle, s: pd.DataFrame) -> Table:
    t = s[s["K"] == 30].copy()
    cols = [
        Col("arm", "Variant", "text", ("arm",), align="l"),
        Col("alpha", "α", "num", ("alpha",), decimals=2, align="l"),
        Col("lhd", "label-HD\n(ours)", "num", ("label_hd_mean",), decimals=HD),
        Col("lp", "label, pairwise\n(FedArtML)", "num", ("label_pairwise_mean",), decimals=HD),
        Col("qhd", "quantity-HD\n(ours)", "num", ("qty_hd_mean",), decimals=HD),
        Col("qp", "quantity, pairwise\n(FedArtML)", "num", ("qty_pairwise_mean",), decimals=HD),
    ]
    notes = [
        "K = 30, means over the generated partitions. Ours: HD of each client to the global label "
        "distribution (mean over clients), and HD of the client sizes to uniform. Pairwise: FedArtML's "
        "values on the same partitions (partitioning.fedartml_compat).",
        "The quantity scales convert exactly: ours = sqrt(1 − sqrt(1 − (K − 1) · pairwise²)) "
        "(partitioning.metrics.quantity_hd_from_pairwise), per partition; the label scales have no "
        "closed-form conversion.",
    ]
    if bundle.rq1_status.get("covtype") == "provisional":
        notes.append("Covtype rows PROVISIONAL.")
    return Table(
        "rq1_pairwise_translation", t, cols, notes, trial_set="cifar10 {0,1,2}; covtype {0-19}"
    )


def rq1_infeasible_table(bundle: Bundle) -> Table:
    r = bundle.rq1[bundle.rq1["status"] != "generated"].copy()
    r["section"] = r["dataset"].map(DATASET_NAMES)
    r["pending"] = ""
    cols = [
        Col("arm", "Variant", "text", ("arm",), align="l"),
        Col("K", "K", "num", ("K",), decimals=0),
        Col("alpha", "α", "num", ("alpha",), decimals=2),
        Col("trial", "Trial", "num", ("trial",), decimals=0),
        Col("status", "Status", "text", ("status",), align="l"),
    ]
    notes = [
        "Partitions whose min-size redraw budget (1000 redraws, min_size 10) ran out: recorded with "
        "status infeasible_min_size and no partition; they are counted, not dropped.",
    ]
    if bundle.rq1_status.get("covtype") == "provisional":
        notes.append("Covtype rows PROVISIONAL.")
    return Table("rq1_infeasible", r, cols, notes, trial_set="cifar10 {0,1,2}; covtype {0-19}")


# ---------------------------------------------------------------- per-class (RQ2, CIFAR-10)
def per_class_differences(bundle: Bundle, level: float) -> dict:
    """{'index'|'reversed': (trials, classes) array of arm - uncapped, final per-class accuracy}."""
    runs = bundle.runs
    base = runs[(runs["algorithm"] == "fedavg") & (runs["L"] == level)]

    def ids(grid, arm):
        g = base[(base["grid"] == grid) & (base["arm"] == arm)]
        return dict(zip(g["trial"], g["run_id"]))

    unc = ids("rq2", "natural uncapped")
    out = {"trials": tuple(sorted(unc))}
    for key, grid, arm in (
        ("index", "rq2", "natural capped"),
        ("reversed", "rq2_class_order", "natural capped, reversed order"),
    ):
        other = ids(grid, arm)
        if set(other) != set(unc):
            raise ValueError(
                f"class-order level {level}: trials differ {sorted(other)} vs {sorted(unc)}"
            )
        out[key] = np.array(
            [
                per_class_final(bundle.sources.results_dir, other[t])
                - per_class_final(bundle.sources.results_dir, unc[t])
                for t in out["trials"]
            ]
        )
    return out


# ---------------------------------------------------------------- tolerance and partition facts
def tolerance(bundle: Bundle) -> pd.DataFrame:
    """Every (cell, trial) partition of the reported grids outside +/- 0.05 of a target."""
    rows = []
    r = bundle.runs[
        (bundle.runs["algorithm"] == "fedavg")
        & bundle.runs["grid"].isin(["rq2", "rq3", "covtype_rq2"])
    ]
    for _, x in r.iterrows():
        if x["merged_out"]:
            continue
        dl = x["realized_label_hd"] - x["L"]
        rows.append(
            {
                "dataset": x["dataset"],
                "grid": x["grid"],
                "arm": x["cell_arm"],
                "level": x["L"],
                "quantity_level": x["Q"],
                "trial": x["trial"],
                "partition_id": x["partition_id"],
                "label_deviation": dl,
                "quantity_deviation": (
                    (x["realized_quantity_hd"] - x["Q"]) if pd.notna(x["Q"]) else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows)
    df["label_outside"] = df["label_deviation"].abs() > LEVEL_TOLERANCE
    df["quantity_outside"] = df["quantity_deviation"].abs() > LEVEL_TOLERANCE
    return df


def covtype_partition_facts(bundle: Bundle) -> pd.DataFrame:
    rows = []
    for L in rq2_levels(bundle, "covtype"):
        for arm in RQ2_ARMS:
            hd = part_means(bundle, "covtype", "covtype_rq2", arm, L)
            p = bundle.parts[
                (bundle.parts["dataset"] == "covtype")
                & (bundle.parts["arm"] == arm)
                & (bundle.parts["target_label_hd"] == L)
            ]
            rows.append(
                {
                    "level": L,
                    "arm": arm,
                    "label_hd": hd["label_hd"],
                    "sw_hd": hd["sw_hd"],
                    "qty_hd": hd["qty_hd"],
                    "mean_label_drift": p["mean_label_drift"].astype(float).mean(),
                    "max_label_drift": p["max_label_drift"].astype(float).max(),
                }
            )
    return pd.DataFrame(rows)


def tolerance_table(bundle: Bundle, tol: pd.DataFrame) -> Table:
    """The (cell, trial) partitions outside +/- 0.05 of a target, label or quantity."""
    out = tol[tol["label_outside"] | tol["quantity_outside"]].copy()
    out = out.sort_values(["dataset", "grid", "arm", "level", "quantity_level", "trial"])
    out["section"] = out["dataset"].map(DATASET_NAMES)
    out["which"] = np.where(out["label_outside"], "label", "quantity")
    out["pending"] = ""
    cols = [
        Col("arm", "Arm", "text", ("arm",), align="l"),
        Col("level", "Label\nlevel", "num", ("level",), decimals=2),
        Col("q", "Quantity\nlevel", "num", ("quantity_level",), decimals=2),
        Col("trial", "Trial", "num", ("trial",), decimals=0),
        Col("which", "Outside", "text", ("which",), align="l"),
        Col("dl", "label-HD\n− target", "num", ("label_deviation",), decimals=HD),
        Col("dq", "quantity-HD\n− target", "num", ("quantity_deviation",), decimals=HD),
    ]
    counts = "; ".join(
        f"{DATASET_NAMES[d]}: {int(g['label_outside'].sum())} of {len(g)} (cell, trial) "
        f"partitions outside on label-HD, {int(g['quantity_outside'].sum())} on quantity-HD"
        for d, g in tol.groupby("dataset")
    )
    notes = [
        "Partitions of the reported RQ2 and RQ3 cells whose realized HD is more than 0.05 from the "
        f"cell's target; {counts}. " + bundle.trial_note("cifar10", "covtype") + ".",
        "The label check is per partition. Quantity targets are level names, judged on their "
        "20-seed mean (decided 2026-10-06), so a quantity deviation is descriptive. Nothing is "
        "excluded: every partition listed was trained and enters the results by its realized HD.",
    ]
    return Table("partition_tolerance", out, cols, notes, trial_set=_trial_sets(bundle))
