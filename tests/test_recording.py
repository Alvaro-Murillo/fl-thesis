"""Tests for the run records: schema, resume, and crash behaviour."""

import csv
import subprocess
import sys
import textwrap

import pytest

from experiments.config import config_hash, from_dict, load_config, to_dict
from experiments.recording import (
    RunRecorder,
    completed_runs,
    read_jsonl,
    runs_path,
    skip_if_exists,
)

REQUIRED_RUN_FIELDS = {
    "run_id",
    "partition_id",
    "config_hash",
    "algorithm",
    "mu",
    "model",
    "rounds",
    "local_epochs",
    "batch_size",
    "lr",
    "trial",
    "init_seed",
    "train_seed",
    "deterministic",
    "final_acc",
    "best_acc",
    "rta",
    "wall_clock_s",
    "status",
    "git_commit",
    "git_dirty",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "started_at",
    "finished_at",
}


def smoke_config(**runtime):
    data = to_dict(load_config("config/smoke.yaml"))
    data["runtime"].update(runtime)
    return from_dict(data)


def dummy_run(config, out_dir, rounds=3, rta_target=None, fail_with=None):
    with RunRecorder(config, "pid123", out_dir, rta_target=rta_target) as rec:
        for r in range(1, rounds + 1):
            rec.log_round(r, 0.1 * r, 1.0 / r, 2.0 / r, 0.5)
            if fail_with is not None and r == 2:
                raise fail_with
    return rec


def test_recorder_schema(tmp_path):
    dummy_run(smoke_config(), tmp_path)
    rows = read_jsonl(runs_path(tmp_path))

    assert [row["status"] for row in rows] == ["running", "completed"]
    for row in rows:
        assert set(row) == REQUIRED_RUN_FIELDS
    final = rows[-1]
    assert final["final_acc"] == pytest.approx(0.3)
    assert final["best_acc"] == pytest.approx(0.3)
    assert final["wall_clock_s"] is not None
    assert final["finished_at"] and final["started_at"]


def test_no_tmp_files_left_behind(tmp_path):
    dummy_run(smoke_config(), tmp_path)
    assert list(tmp_path.rglob("*.tmp")) == []


def test_history_csv_is_long_form(tmp_path):
    dummy_run(smoke_config(), tmp_path)
    run_dir = tmp_path / "runs" / config_hash(smoke_config())

    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["round"] for r in rows] == ["1", "2", "3"]
    assert float(rows[-1]["global_test_acc"]) == pytest.approx(0.3)
    assert (run_dir / "config.yaml").exists()


def test_resolved_config_is_self_describing(tmp_path):
    config = smoke_config()
    dummy_run(config, tmp_path)
    dumped = load_config(tmp_path / "runs" / config_hash(config) / "config.yaml")
    assert dumped == config


def test_rta_is_first_round_reaching_target(tmp_path):
    dummy_run(smoke_config(), tmp_path, rounds=5, rta_target=0.25)
    assert read_jsonl(runs_path(tmp_path))[-1]["rta"] == 3


def test_rta_is_null_when_never_reached(tmp_path):
    dummy_run(smoke_config(), tmp_path, rounds=3, rta_target=0.99)
    assert read_jsonl(runs_path(tmp_path))[-1]["rta"] is None


def test_failure_is_recorded_and_reraised(tmp_path):
    with pytest.raises(RuntimeError):
        dummy_run(smoke_config(), tmp_path, fail_with=RuntimeError("boom"))

    rows = read_jsonl(runs_path(tmp_path))
    assert rows[-1]["status"] == "failed"
    assert completed_runs(tmp_path) == set()


def test_oom_gets_its_own_status(tmp_path):
    with pytest.raises(RuntimeError):
        dummy_run(
            smoke_config(),
            tmp_path,
            fail_with=RuntimeError("CUDA out of memory. Tried to allocate..."),
        )
    assert read_jsonl(runs_path(tmp_path))[-1]["status"] == "oom"


def test_resume_skip(tmp_path):
    config = smoke_config()
    assert not skip_if_exists(config, tmp_path)

    dummy_run(config, tmp_path)
    assert skip_if_exists(config, tmp_path)

    # A different cell in the same grid is not skipped.
    assert not skip_if_exists(smoke_config(trial=1), tmp_path)


def test_failed_run_is_not_skipped(tmp_path):
    """A crashed run must be retried, not treated as done."""
    config = smoke_config()
    with pytest.raises(RuntimeError):
        dummy_run(config, tmp_path, fail_with=RuntimeError("boom"))
    assert not skip_if_exists(config, tmp_path)

    dummy_run(config, tmp_path)
    assert skip_if_exists(config, tmp_path)


def test_killed_run_leaves_running_row_and_partial_history(tmp_path):
    """A hard kill mid-run stays diagnosable."""
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(tmp_path.cwd())!r})
        from tests.test_recording import smoke_config
        from experiments.recording import RunRecorder
        rec = RunRecorder(smoke_config(), "pid123", {str(tmp_path)!r}).start()
        rec.log_round(1, 0.1, 1.0, 2.0, 0.5)
        rec.log_round(2, 0.2, 0.5, 1.0, 0.5)
        os._exit(1)  # no unwinding: the recorder never gets to write an outcome
    """)
    assert subprocess.run([sys.executable, "-c", script]).returncode == 1

    rows = read_jsonl(runs_path(tmp_path))
    assert [row["status"] for row in rows] == ["running"]
    assert rows[-1]["finished_at"] is None

    history = tmp_path / "runs" / config_hash(smoke_config()) / "history.csv"
    with history.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert [r["round"] for r in parsed] == ["1", "2"]
    assert not skip_if_exists(smoke_config(), tmp_path)
