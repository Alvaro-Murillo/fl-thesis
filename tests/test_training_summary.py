"""Tests for the across-trial training summary."""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis import training_summary
from analysis.training_summary import (
    condition_id,
    grid_runs,
    mean_curve_rta,
    runs_table,
    summary_table,
)
from experiments.config import config_hash, dump_config, from_dict, load_config, to_dict
from experiments.recording import RunRecorder, append_jsonl


def cell(out_dir, trial=0, algorithm="fedavg", mu=0.0):
    data = to_dict(load_config("config/smoke.yaml"))
    data["federated"].update(algorithm=algorithm, mu=mu)
    data["runtime"].update(trial=trial, out_dir=str(out_dir))
    return from_dict(data)


def record(config, accuracies, pid=None, hd=None, fail=False):
    """A run with these per-round accuracies, and its partition row if `hd` is given."""
    pid = pid or f"p{config.runtime.trial}"
    if hd is not None:
        append_jsonl(
            config.runtime.out_dir + "/partitions.jsonl",
            {"partition_id": pid, "realized_label_hd": hd[0], "realized_quantity_hd": hd[1]},
        )
    try:
        with RunRecorder(config, pid) as rec:
            rec.log_initial(0.1, 2.3)
            for r, acc in enumerate(accuracies, 1):
                rec.log_round(r, acc, 1.0, 1.0, 0.5)
                if fail and r == 2:
                    raise RuntimeError("boom")
    except RuntimeError:
        pass


def test_condition_id_ignores_the_runtime_only(tmp_path):
    base = cell(tmp_path)
    assert condition_id(base) == condition_id(cell(tmp_path / "elsewhere", trial=2))
    assert condition_id(base) != condition_id(cell(tmp_path, algorithm="fedprox", mu=0.01))
    assert config_hash(base) != config_hash(cell(tmp_path, trial=2))


def test_summary_mean_sd_and_realized_hd(tmp_path):
    curves = [[0.2, 0.5, 0.6], [0.3, 0.4, 0.5], [0.1, 0.6, 0.7]]
    for trial, curve in enumerate(curves):
        record(cell(tmp_path, trial), curve, hd=(0.30 + trial / 100, 0.10 + trial / 100))
    prox = cell(tmp_path, 0, "fedprox", 0.01)
    record(prox, [0.25, 0.45, 0.55], pid="p0")

    runs = runs_table(tmp_path)
    assert len(runs) == 4
    summary = summary_table(runs, tmp_path).set_index("condition_id")
    row = summary.loc[condition_id(cell(tmp_path))]
    finals = [c[-1] for c in curves]
    assert row["n_runs"] == 3 and row["trials"] == "0 1 2"
    assert row["final_acc_mean"] == pytest.approx(np.mean(finals))
    assert row["final_acc_sd"] == pytest.approx(np.std(finals, ddof=1))
    assert row["best_acc_mean"] == pytest.approx(np.mean([max(c) for c in curves]))
    # per-run rta: 0.54 -> round 3, 0.45 -> round 3, 0.63 -> round 3
    assert row["rta_mean"] == 3 and row["rta_sd"] == 0
    assert row["realized_label_hd_mean"] == pytest.approx(0.31)
    assert row["realized_quantity_hd_mean"] == pytest.approx(0.11)

    single = summary.loc[condition_id(prox)]
    assert single["n_runs"] == 1 and np.isnan(single["final_acc_sd"])
    assert single["realized_label_hd_mean"] == pytest.approx(0.30)  # shares trial 0's partition


def test_rta_of_the_mean_curve_differs_from_the_mean_rta(tmp_path):
    """Two fast runs and a slow one: rta 1, 1 and 3, but the mean curve gets there at 3."""
    record(cell(tmp_path, 0), [0.60, 0.60, 0.60])
    record(cell(tmp_path, 1), [0.60, 0.60, 0.60])
    record(cell(tmp_path, 2), [0.10, 0.30, 0.60])
    runs = runs_table(tmp_path)
    assert sorted(runs["rta"]) == [1, 1, 3]
    assert mean_curve_rta(tmp_path, runs["run_id"]) == 3  # mean curve 0.433, 0.5, 0.6
    row = summary_table(runs, tmp_path).iloc[0]
    assert row["rta_mean"] == pytest.approx(5 / 3) and row["rta_of_mean_curve"] == 3


def test_only_the_latest_completed_row_counts(tmp_path):
    record(cell(tmp_path, 0), [0.2, 0.3, 0.4], fail=True)  # failed
    record(cell(tmp_path, 1), [0.2, 0.3, 0.4], fail=True)
    record(cell(tmp_path, 1), [0.5, 0.6, 0.7])  # retried to completion
    killed = RunRecorder(cell(tmp_path, 2), "p2").start()  # a "running" row and nothing after
    killed.log_round(1, 0.9, 1.0, 1.0, 0.5)
    for handle in killed._files:
        handle.close()
    runs = runs_table(tmp_path)
    assert list(runs["trial"]) == [1]
    assert runs["final_acc"].tolist() == [0.7]


def test_a_mean_curve_needs_equal_lengths(tmp_path):
    record(cell(tmp_path, 0), [0.2, 0.3])
    record(cell(tmp_path, 1), [0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="different lengths"):
        mean_curve_rta(tmp_path, runs_table(tmp_path)["run_id"])


def test_command_line_on_a_grid(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.yaml"
    dump_config(cell(tmp_path), base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"base": str(base), "sweep": {"runtime.trial": [0, 1, 2]}}))
    record(cell(tmp_path, 0), [0.2, 0.5, 0.6], hd=(0.3, 0.1))
    record(cell(tmp_path, 1), [0.3, 0.4, 0.5], hd=(0.3, 0.1))
    record(cell(tmp_path, 7), [0.9, 0.9, 0.9])  # not in the grid
    run_ids, expected, out_dir = grid_runs([grid])
    assert len(run_ids) == 3 and expected == {condition_id(cell(tmp_path)): 3}
    assert out_dir == tmp_path

    monkeypatch.setattr(sys, "argv", ["training_summary", "--grid", str(grid)])
    training_summary.main()
    out = capsys.readouterr().out
    assert (
        "1 conditions in the grid: 0 with no completed run, 1 with only some of their runs" in out
    )
    summary = pd.read_csv(tmp_path / "analysis" / "training_summary.csv")
    assert summary["n_runs"].tolist() == [2] and summary["n_expected"].tolist() == [3]
    assert summary["final_acc_mean"].tolist() == pytest.approx([0.55])
    runs = pd.read_csv(tmp_path / "analysis" / "training_runs.csv")
    assert runs["trial"].tolist() == [0, 1] and "git_commit" in runs
