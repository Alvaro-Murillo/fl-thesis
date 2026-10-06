"""Tests for the partition validation tables, verdicts and figures."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.recording import load_partition, read_jsonl
from partitioning.metrics import hellinger

LABELS = np.repeat(np.arange(10), 600)  # N = 6000; K * 31 <= N at K = 10


def write_grid(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def label_grid(tmp_path, cases=None):
    """Two label-axis arms at one label target, 3 trials."""
    return write_grid(
        tmp_path / "label_arms.yaml",
        "base: config/smoke.yaml\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "  partition.target_label_hd: 0.3\n"
        "cases:\n"
        + (
            cases
            or "  - {partition.mode: natural_per_class, partition.alpha: 0.5}\n"
            "  - {partition.mode: pure_label, partition.alpha: 1000.0, "
            "partition.min_size_policy: 0}\n"
        )
        + "sweep:\n"
        "  runtime.trial: [0, 1, 2]\n",
    )


def two_axis_grid(tmp_path):
    """combined at 2 label targets x 2 quantity levels, 3 trials."""
    return write_grid(
        tmp_path / "two_axis.yaml",
        "base: config/smoke_combined.yaml\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "cases:\n"
        "  - {partition.beta: 1000.0, partition.target_quantity_hd: 0.0}\n"
        "  - {partition.beta: 0.3, partition.target_quantity_hd: 0.4}\n"
        "sweep:\n"
        "  partition.target_label_hd: [0.3, 0.6]\n"  # alpha stays 0.5; targets feed the checks
        "  runtime.trial: [0, 1, 2]\n",
    )


@pytest.fixture
def labels_everywhere(monkeypatch):
    import analysis.partition_validation as validation
    import experiments.quantity_levels as levels
    import experiments.run_grid as run_grid

    for module in (validation, levels, run_grid):
        monkeypatch.setattr(module, "load_labels", lambda dataset: LABELS)


def run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, argv)])
    module.main()


def generate(monkeypatch, grid, seeds=None):
    import experiments.quantity_levels as levels
    import experiments.run_grid as run_grid

    run(run_grid, monkeypatch, "--grid", grid)
    if seeds is not None:
        run(levels, monkeypatch, "--grid", grid, "--seeds", seeds)


def validate(monkeypatch, *grids):
    import analysis.partition_validation as validation

    args = [a for g in grids for a in ("--grid", g)]
    run(validation, monkeypatch, *args)


def tables(tmp_path):
    out = tmp_path / "analysis"
    return tuple(
        pd.read_csv(out / f"partition_validation_{n}.csv")
        for n in ("partitions", "clients", "cells")
    )


# --- end to end -------------------------------------------------------------


def test_writes_every_table_and_figure(tmp_path, monkeypatch, labels_everywhere):
    one, two = label_grid(tmp_path), two_axis_grid(tmp_path)
    generate(monkeypatch, one)
    generate(monkeypatch, two, seeds=5)
    validate(monkeypatch, one, two)
    for name in ("partitions.csv", "clients.csv", "cells.csv", "label.png", "quantity.png"):
        assert (tmp_path / "analysis" / f"partition_validation_{name}").exists()
    assert (tmp_path / "analysis" / "partition_validation_feasibility.png").exists()
    partitions, clients, cells = tables(tmp_path)
    assert len(partitions) == 2 * 3 + 4 * 3
    assert len(cells) == 2 + 4
    assert len(clients) == len(partitions) * 10


def test_partition_rows_are_the_recorded_rows_with_the_label_check(
    tmp_path, monkeypatch, labels_everywhere
):
    grid = two_axis_grid(tmp_path)
    generate(monkeypatch, grid, seeds=5)
    validate(monkeypatch, grid)
    partitions, _, _ = tables(tmp_path)
    recorded = {r["partition_id"]: r for r in read_jsonl(tmp_path / "partitions.jsonl")}
    assert len(recorded) == 6 and len(partitions) == 12  # both label targets share alpha 0.5
    for row in partitions.itertuples():
        source = recorded[row.partition_id]
        deviation = source["realized_label_hd"] - row.target_label_hd
        assert row.realized_label_hd == pytest.approx(source["realized_label_hd"])
        assert row.label_hd_deviation == pytest.approx(deviation)
        assert row.label_within_tolerance == (abs(deviation) <= 0.05)
        assert row.quantity_hd_deviation == pytest.approx(source["quantity_hd_deviation"])
        assert json.loads(row.client_sizes) == source["client_sizes"]
        if row.target_label_hd == source["target_label_hd"]:  # the cell that generated it
            assert row.label_within_tolerance == source["feasible"]
    assert set(partitions.groupby("partition_id")["target_label_hd"].nunique()) == {2}


def test_client_rows_are_each_clients_label_histogram(tmp_path, monkeypatch, labels_everywhere):
    grid = two_axis_grid(tmp_path)
    generate(monkeypatch, grid, seeds=5)
    validate(monkeypatch, grid)
    partitions, clients, _ = tables(tmp_path)
    classes = [f"class_{c}" for c in range(10)]
    recorded = {r["partition_id"]: r for r in read_jsonl(tmp_path / "partitions.jsonl")}
    for (pid, _), g in clients.groupby(["partition_id", "target_label_hd"]):
        parts = load_partition(tmp_path, pid)
        expected = np.stack([np.bincount(LABELS[p], minlength=10) for p in parts])
        assert np.array_equal(g.sort_values("client")[classes].to_numpy(), expected)
        assert g["size"].tolist() == recorded[pid]["client_sizes"]
        global_dist = np.full(10, 0.1)
        hds = [hellinger(row / row.sum(), global_dist) for row in expected]
        assert np.allclose(g["label_hd"], hds)
        assert g["label_hd"].mean() == pytest.approx(recorded[pid]["realized_label_hd"])
        assert np.allclose(g["label_drift"], recorded[pid]["label_drift"])


def test_cell_verdicts_follow_the_rows_and_the_levels(tmp_path, monkeypatch, labels_everywhere):
    one, two = label_grid(tmp_path), two_axis_grid(tmp_path)
    generate(monkeypatch, one)
    generate(monkeypatch, two, seeds=5)
    validate(monkeypatch, one, two)
    partitions, _, cells = tables(tmp_path)
    levels = {
        (r["beta"], r["target_quantity_hd"]): r
        for r in read_jsonl(tmp_path / "quantity_levels.jsonl")
    }
    for cell in cells.itertuples():
        rows = partitions[
            (partitions["arm"] == cell.arm)
            & (partitions["target_label_hd"] == cell.target_label_hd)
        ]
        if not pd.isna(cell.target_quantity_hd):
            rows = rows[rows["target_quantity_hd"] == cell.target_quantity_hd]
        assert cell.n_partitions == len(rows) == 3
        assert cell.n_label_within_tolerance == rows["label_within_tolerance"].sum()
        assert cell.mean_realized_label_hd == pytest.approx(rows["realized_label_hd"].mean())
        label_ok = bool(rows["label_within_tolerance"].all())
        assert cell.label_feasible == label_ok
        if pd.isna(cell.target_quantity_hd):
            assert pd.isna(cell.level_feasible)
            assert cell.cell_feasible == label_ok
        else:
            level = levels[(cell.beta, cell.target_quantity_hd)]
            assert cell.level_n_seeds == 5
            assert cell.level_mean_quantity_hd == pytest.approx(level["mean_quantity_hd"])
            assert cell.level_feasible == level["level_feasible"]
            assert cell.cell_feasible == (label_ok and level["level_feasible"])
    # pure_label at alpha 1000 holds about the global mix, nowhere near label-HD 0.3.
    assert not cells.loc[cells["arm"] == "pure_label", "cell_feasible"].any()


def test_training_cases_share_one_partition(tmp_path, monkeypatch, labels_everywhere):
    grid = write_grid(
        tmp_path / "train_cases.yaml",
        "base: config/smoke_pure_label.yaml\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "  partition.target_label_hd: 0.3\n"
        "cases:\n"
        "  - {federated.algorithm: fedavg}\n"
        "  - {federated.algorithm: fedprox, federated.mu: 0.01}\n"
        "sweep:\n"
        "  runtime.trial: [0, 1]\n",
    )
    generate(monkeypatch, grid)
    validate(monkeypatch, grid)
    partitions, _, cells = tables(tmp_path)
    assert len(partitions) == 2 and partitions["partition_id"].is_unique
    assert len(cells) == 1 and cells["n_partitions"].iloc[0] == 2


def test_label_only_grid_needs_no_level_check(tmp_path, monkeypatch, labels_everywhere):
    grid = label_grid(tmp_path)
    generate(monkeypatch, grid)
    validate(monkeypatch, grid)
    _, _, cells = tables(tmp_path)
    assert cells["target_quantity_hd"].isna().all() and cells["level_feasible"].isna().all()
    assert not (tmp_path / "analysis" / "partition_validation_quantity.png").exists()


def test_untargeted_cells_are_left_out(tmp_path, monkeypatch, labels_everywhere, capsys):
    import analysis.partition_validation as validation

    grid = label_grid(
        tmp_path,
        "  - {partition.mode: natural_per_class, partition.alpha: 0.5}\n"
        "  - {partition.mode: natural_per_class, partition.alpha: 5.0, "
        "partition.target_label_hd: null}\n",
    )
    generate(monkeypatch, grid)
    records, levels, _, _ = validation.load_cells([grid])
    assert len(records) == 3 and {r["alpha"] for r in records} == {0.5}
    assert levels == {}
    assert "3 cells set no HD target" in capsys.readouterr().out


# --- what is missing ----------------------------------------------------------


def test_missing_partitions_name_the_grid_to_run(tmp_path, monkeypatch, labels_everywhere):
    import analysis.partition_validation as validation

    grid = two_axis_grid(tmp_path)
    # 4 cells x 3 trials, but the two label targets share alpha, hence partitions.
    with pytest.raises(SystemExit, match=r"run_grid --grid .*two_axis\.yaml   \(6 missing\)"):
        validation.load_cells([grid])


def test_missing_levels_name_the_level_check(tmp_path, monkeypatch, labels_everywhere):
    import analysis.partition_validation as validation

    grid = two_axis_grid(tmp_path)
    generate(monkeypatch, grid)  # no level check
    with pytest.raises(SystemExit, match=r"quantity_levels --grid .*two_axis\.yaml"):
        validation.load_cells([grid])


def test_a_grid_without_targets_is_refused(tmp_path):
    import analysis.partition_validation as validation

    grid = write_grid(
        tmp_path / "none.yaml",
        f"base: config/smoke.yaml\nconstant:\n  runtime.out_dir: {tmp_path.as_posix()}\n",
    )
    with pytest.raises(SystemExit, match="sets an HD target"):
        validation.load_cells([grid])


# --- the verdict rule, on hand-made rows ------------------------------------


def record(trial, realized, target=0.3, status="generated", **fields):
    base = {
        "grid": "g",
        "arm": "combined",
        "dataset": "cifar10",
        "K": 10,
        "mode": "combined",
        "size_cap": "none",
        "min_size_policy": 0,
        "quantity_floor": 31,
        "target_label_hd": target,
        "target_quantity_hd": 0.4,
        "alpha": 0.5,
        "beta": 0.3,
        "trial": trial,
        "partition_id": f"p{trial}-{realized}",
        "status": status,
        "realized_label_hd": realized if status == "generated" else None,
        "realized_quantity_hd": 0.4,
        "mean_label_drift": 0.1,
        "max_label_drift": 0.2,
        "min_client_size": 31,
        "n_empty_clients": 0,
    }
    return base | fields


def level(feasible):
    return {
        ("cifar10", 10, "combined", 31, 0.3, 0.4): {
            "realized_quantity_hd": [0.4] * 20,
            "mean_quantity_hd": 0.4 if feasible else 0.5,
            "sd_quantity_hd": 0.0,
            "mean_deviation": 0.0 if feasible else 0.1,
            "share_within_tolerance": 1.0 if feasible else 0.0,
            "level_feasible": feasible,
        }
    }


def verdict(records, levels):
    from analysis.partition_validation import cells_table, partitions_table

    return cells_table(partitions_table(records), levels).iloc[0]


@pytest.mark.parametrize(
    "realized, level_ok, expected",
    [
        ((0.30, 0.34, 0.26), True, True),  # every trial within 0.05
        ((0.30, 0.34, 0.36), True, False),  # one trial out: the mean (0.333) is not enough
        ((0.30, 0.30, 0.30), False, False),  # the label holds, the quantity level does not
    ],
)
def test_a_cell_needs_every_trial_and_its_level(realized, level_ok, expected):
    cell = verdict([record(t, r) for t, r in enumerate(realized)], level(level_ok))
    assert cell["n_label_within_tolerance"] == sum(abs(r - 0.3) <= 0.05 for r in realized)
    assert bool(cell["cell_feasible"]) is expected


def test_a_trial_without_a_partition_fails_the_label_check():
    records = [record(0, 0.3), record(1, 0.3), record(2, None, status="infeasible_min_size")]
    cell = verdict(records, level(True))
    assert cell["n_infeasible_min_size"] == 1 and cell["n_label_within_tolerance"] == 2
    assert not cell["cell_feasible"]
    assert cell["mean_realized_label_hd"] == pytest.approx(0.3)


def test_a_cell_without_a_label_target_is_judged_by_its_level():
    records = [record(t, 0.1, target=None) for t in range(3)]
    assert verdict(records, level(True))["cell_feasible"]
    assert not verdict(records, level(False))["cell_feasible"]
    assert verdict(records, level(True))["label_feasible"] is None


def test_one_cell_with_two_parameters_is_refused():
    from analysis.partition_validation import cells_table, partitions_table

    records = [record(0, 0.3), record(1, 0.3, alpha=5.0)]
    with pytest.raises(ValueError, match="more than one alpha/beta"):
        cells_table(partitions_table(records), level(True))
