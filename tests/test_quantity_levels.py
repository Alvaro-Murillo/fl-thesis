"""Tests for the quantity-axis tolerance: per level over 20 seeds, never per partition."""

import sys

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import load_partition, partition_exists, read_jsonl
from experiments.seeding import derive_seeds
from partitioning.metrics import hellinger
from partitioning.two_stage_method import combined

LABELS = np.repeat(np.arange(10), 600)  # N = 6000; K * 31 <= N at K = 10
N, K = len(LABELS), 10


def combined_config(tmp_path, **partition):
    data = to_dict(load_config("config/smoke_combined.yaml"))  # K = 10, alpha 0.5, beta 1.0
    data["partition"].update(partition)
    data["runtime"]["out_dir"] = str(tmp_path)
    return from_dict(data)


def realized_quantity_hd(alpha, beta, trial):
    rng = np.random.default_rng(derive_seeds(trial).partition)
    sizes = [len(p) for p in combined(LABELS, K, alpha, beta, 31, rng)]
    return hellinger(np.array(sizes) / N, np.full(K, 1 / K))


# --- per-partition rows -----------------------------------------------------


def test_a_quantity_miss_never_makes_a_row_infeasible(tmp_path):
    import experiments.run_partition as runner

    config = combined_config(tmp_path, beta=1000.0, target_quantity_hd=0.6)
    row = runner.generate_partition(config, labels=LABELS)
    assert row["realized_quantity_hd"] < 0.1  # far from the 0.6 level name
    assert row["status"] == "generated" and row["feasible"] is True
    assert partition_exists(tmp_path, row["partition_id"])
    assert row["quantity_hd_deviation"] == pytest.approx(row["realized_quantity_hd"] - 0.6)
    assert row["label_hd_deviation"] is None


def test_the_label_tolerance_stays_per_partition(tmp_path):
    import experiments.run_partition as runner

    hit = runner.generate_partition(combined_config(tmp_path, target_label_hd=0.0), labels=LABELS)
    assert hit["feasible"] is False and hit["realized_label_hd"] > 0.05
    assert hit["label_hd_deviation"] == pytest.approx(hit["realized_label_hd"])
    near = combined_config(tmp_path, target_label_hd=hit["realized_label_hd"] - 0.04)
    ok = runner.generate_partition(near, labels=LABELS, force=True)  # same partition_id
    assert ok["feasible"] is True


def test_rows_without_targets_carry_null_deviations(tmp_path):
    import experiments.run_partition as runner

    row = runner.generate_partition(combined_config(tmp_path), labels=LABELS)
    assert row["label_hd_deviation"] is None and row["quantity_hd_deviation"] is None
    assert row["feasible"] is True


def test_client_sizes_pair_cells_that_share_a_size_draw(tmp_path):
    import experiments.run_partition as runner

    def row(**partition):
        return runner.generate_partition(combined_config(tmp_path, **partition), labels=LABELS)

    a, b, c = row(alpha=0.1), row(alpha=5.0), row(alpha=0.1, beta=0.3)
    assert a["partition_seed"] == b["partition_seed"] and a["beta"] == b["beta"]
    assert a["client_sizes"] == b["client_sizes"]  # same trial and beta, other label level
    assert a["client_sizes"] != c["client_sizes"]  # other beta
    parts = load_partition(tmp_path, a["partition_id"])
    assert a["client_sizes"] == [len(p) for p in parts]


def test_run_grid_records_every_cell_whatever_its_quantity_deviation(tmp_path, monkeypatch):
    import experiments.run_grid as run_grid

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "base: config/smoke_combined.yaml\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "  partition.target_quantity_hd: 0.6\n"
        "sweep:\n"
        "  partition.beta: [0.05, 1.0, 1000.0]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_grid, "load_labels", lambda dataset: LABELS)
    monkeypatch.setattr(sys, "argv", ["run_grid", "--grid", str(grid)])
    run_grid.main()
    rows = read_jsonl(tmp_path / "partitions.jsonl")
    assert len(rows) == 3
    assert all(r["status"] == "generated" and r["feasible"] for r in rows)
    assert all(partition_exists(tmp_path, r["partition_id"]) for r in rows)
    assert max(abs(r["quantity_hd_deviation"]) for r in rows) > 0.05


# --- the level check --------------------------------------------------------


def test_level_check_is_the_20_seed_mean_of_the_cells_own_partitioner(tmp_path):
    from experiments.quantity_levels import check_level

    config = combined_config(tmp_path, beta=0.3, target_quantity_hd=0.4)
    level = check_level(config, LABELS)
    expected = np.array([realized_quantity_hd(0.5, 0.3, t) for t in range(20)])
    assert level["trials"] == list(range(20))
    assert np.allclose(level["realized_quantity_hd"], expected)
    assert level["mean_quantity_hd"] == pytest.approx(expected.mean())
    assert level["sd_quantity_hd"] == pytest.approx(expected.std(ddof=1))
    assert level["share_within_tolerance"] == pytest.approx(np.mean(np.abs(expected - 0.4) <= 0.05))
    assert level["mean_deviation"] == pytest.approx(expected.mean() - 0.4)
    assert (level["K"], level["mode"], level["beta"], level["quantity_floor"]) == (
        10,
        "combined",
        0.3,
        31,
    )


def test_level_verdict_turns_at_exactly_0_05_of_the_mean(tmp_path):
    from experiments.quantity_levels import check_level

    mean = np.mean([realized_quantity_hd(0.5, 0.3, t) for t in range(20)])
    inside = combined_config(tmp_path, beta=0.3, target_quantity_hd=mean + 0.0499)
    outside = combined_config(tmp_path, beta=0.3, target_quantity_hd=mean + 0.0501)
    assert check_level(inside, LABELS)["level_feasible"] is True
    assert check_level(outside, LABELS)["level_feasible"] is False


def test_level_check_builds_one_partition_per_seed_and_saves_none(tmp_path, monkeypatch):
    import experiments.quantity_levels as levels

    calls = []
    real = levels.build_partition
    monkeypatch.setattr(levels, "build_partition", lambda *a: calls.append(1) or real(*a))
    levels.check_level(combined_config(tmp_path, target_quantity_hd=0.3), LABELS)
    assert len(calls) == 20
    assert not (tmp_path / "partitions").exists()


def test_levels_cli_writes_one_row_per_distinct_level(tmp_path, monkeypatch):
    import experiments.quantity_levels as levels

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "base: config/smoke_combined.yaml\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "cases:\n"
        "  - {partition.beta: 1000.0, partition.target_quantity_hd: 0.0}\n"
        "  - {partition.beta: 0.3, partition.target_quantity_hd: 0.4}\n"
        "sweep:\n"
        "  partition.alpha: [0.1, 5.0]\n"  # label levels share each quantity level
        "  runtime.trial: [0, 1, 2]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(levels, "load_labels", lambda dataset: LABELS)
    monkeypatch.setattr(sys, "argv", ["quantity_levels", "--grid", str(grid), "--seeds", "5"])
    levels.main()
    rows = read_jsonl(tmp_path / "quantity_levels.jsonl")
    assert [(r["beta"], r["target_quantity_hd"]) for r in rows] == [(1000.0, 0.0), (0.3, 0.4)]
    assert all(len(r["realized_quantity_hd"]) == 5 for r in rows)
    assert not (tmp_path / "partitions").exists()
