"""Tests for the run records: schema, resume, and crash behaviour."""

import csv
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from experiments.config import config_hash, from_dict, load_config, to_dict
from experiments.recording import (
    RTA_FRACTION,
    RunRecorder,
    completed_runs,
    read_jsonl,
    rounds_to_accuracy,
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


def dummy_run(config, out_dir, rounds=3, fail_with=None, accuracies=None):
    accuracies = accuracies or [0.1 * r for r in range(1, rounds + 1)]
    with RunRecorder(config, "pid123", out_dir) as rec:
        for r, acc in enumerate(accuracies, 1):
            rec.log_round(r, acc, 1.0 / r, 2.0 / r, 0.5)
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


def test_per_class_recorder_adds_its_files_and_balanced_accuracy(tmp_path):
    config = smoke_config()
    num_classes = config.dataset.num_classes
    untrained = np.full((num_classes, num_classes), 10)
    perfect = np.diag(np.full(num_classes, 100))
    skewed = perfect.copy()
    skewed[0, 0], skewed[0, 1] = 50, 50  # class 0 at 0.5, the others at 1
    with RunRecorder(config, "pid123", tmp_path, per_class=True) as rec:
        rec.log_initial(0.1, 2.3)
        rec.log_confusion(0, untrained)
        for r, cm in enumerate((perfect, skewed), 1):
            rec.log_round(r, np.trace(cm) / cm.sum(), 1.0, 1.0, 0.5)
            rec.log_confusion(r, cm)

    rows = read_jsonl(runs_path(tmp_path))
    for row in rows:
        assert set(row) == REQUIRED_RUN_FIELDS | {"final_balanced_acc", "best_balanced_acc"}
    assert rows[0]["final_balanced_acc"] is None
    final = rows[-1]
    expected = (0.5 + (num_classes - 1)) / num_classes
    assert final["final_balanced_acc"] == pytest.approx(expected)
    assert final["best_balanced_acc"] == 1.0  # round 0 (0.1) counts for nothing either way

    run_dir = tmp_path / "runs" / config_hash(config)
    with (run_dir / "per_class.csv").open(newline="", encoding="utf-8") as fh:
        per_class = list(csv.DictReader(fh))
    assert [row["round"] for row in per_class] == ["0", "1", "2"]
    assert float(per_class[0]["balanced_acc"]) == pytest.approx(0.1)
    assert float(per_class[2]["acc_class_0"]) == 0.5
    assert float(per_class[2][f"acc_class_{num_classes - 1}"]) == 1.0
    with (run_dir / "confusion.csv").open(newline="", encoding="utf-8") as fh:
        confusion = list(csv.DictReader(fh))
    assert len(confusion) == 3 * num_classes**2
    last = {(int(r["true_class"]), int(r["predicted_class"])): int(r["count"]) for r in confusion}
    assert last[(0, 1)] == 50 and last[(0, 0)] == 50 and last[(1, 1)] == 100


def test_a_recorder_without_per_class_refuses_a_confusion(tmp_path):
    with RunRecorder(smoke_config(), "pid123", tmp_path) as rec:
        with pytest.raises(RuntimeError, match="without per_class"):
            rec.log_confusion(0, np.eye(10))
    assert not (tmp_path / "runs" / config_hash(smoke_config()) / "per_class.csv").exists()


def test_a_confusion_of_the_wrong_size_is_refused(tmp_path):
    with RunRecorder(smoke_config(), "pid123", tmp_path, per_class=True) as rec:
        with pytest.raises(ValueError, match="10 x 10"):
            rec.log_confusion(0, np.eye(7))


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


def test_rta_is_first_round_reaching_90_percent_of_the_runs_best(tmp_path):
    # best 0.60, threshold 0.54: round 3 (0.53) misses, round 4 (0.55) reaches it
    dummy_run(smoke_config(), tmp_path, accuracies=[0.2, 0.4, 0.53, 0.55, 0.60, 0.58])
    assert read_jsonl(runs_path(tmp_path))[-1]["rta"] == 4


def test_rounds_to_accuracy():
    assert rounds_to_accuracy([0.2, 0.4, 0.53, 0.55, 0.60, 0.58]) == 4
    assert rounds_to_accuracy([0.5, 0.1, 0.52]) == 1  # the first round reaching it, dips or not
    assert rounds_to_accuracy([0.3, 0.5], fraction=1.0) == 2
    assert rounds_to_accuracy([0.3, 0.5], reference=0.8) is None  # an explicit max not reached
    assert rounds_to_accuracy([0.3, 0.5, 0.72], reference=0.8) == 3
    assert rounds_to_accuracy([]) is None
    assert rounds_to_accuracy(np.array([0.1, 0.2])) == 2
    for fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            rounds_to_accuracy([0.5], fraction=fraction)


def test_rta_counts_an_exact_tie_with_the_threshold():
    """0.9 * 6610/10000 is 5949/10000 exactly, but not in floating point."""
    best, tie = 6610 / 10000, 5949 / 10000
    assert tie < RTA_FRACTION * best  # the naive comparison would miss it
    assert rounds_to_accuracy([0.1, tie, best]) == 2
    below = 5948 / 10000  # one test sample short
    assert rounds_to_accuracy([0.1, below, best]) == 3


def test_round_zero_is_written_but_counts_for_nothing(tmp_path):
    with RunRecorder(smoke_config(), "pid123", tmp_path) as rec:
        rec.log_initial(0.95, 0.1)
        rec.log_round(1, 0.2, 2.0, 2.5, 0.5)
        rec.log_round(2, 0.3, 1.5, 2.0, 0.5)
        with pytest.raises(RuntimeError, match="round 0"):
            rec.log_initial(0.1, 2.3)
    final = read_jsonl(runs_path(tmp_path))[-1]
    assert (final["final_acc"], final["best_acc"], final["rta"]) == (0.3, 0.3, 2)

    history = tmp_path / "runs" / config_hash(smoke_config()) / "history.csv"
    with history.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["round"] for r in rows] == ["0", "1", "2"]
    assert rows[0] == {
        "round": "0",
        "global_test_acc": "0.95",
        "global_test_loss": "0.1",
        "train_loss": "",
        "round_wall_s": "",
    }


def test_clients_csv_has_a_row_per_trained_client(tmp_path):
    with RunRecorder(smoke_config(), "pid123", tmp_path) as rec:
        rec.log_round(1, 0.2, 2.0, 2.5, 0.5)
        rec.log_clients(1, (0, 2), (100, 40), (2.25, 3.0), (20, 8))
        with pytest.raises(ValueError):
            rec.log_clients(2, (0, 2), (100,), (2.0, 2.9), (20, 8))  # lengths differ
    clients = tmp_path / "runs" / config_hash(smoke_config()) / "clients.csv"
    with clients.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [
        {"round": "1", "client": "0", "num_samples": "100", "train_loss": "2.25", "steps": "20"},
        {"round": "1", "client": "2", "num_samples": "40", "train_loss": "3.0", "steps": "8"},
    ]


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
