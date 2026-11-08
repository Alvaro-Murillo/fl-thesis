"""Artifact store, atomic writes and provenance.

Partitions are content-addressed: `partition_id` hashes only the fields that
change the split, so FedAvg and FedProx at the same grid cell provably read the
same bytes, and changing `federated.rounds` does not regenerate anything.

Runs append to `runs.jsonl` twice - once at start with status `running`, once
at the end with the outcome - so a killed run is visible as `running` next to
a partial but parseable `history.csv` and `clients.csv`. Readers take the last
row per `run_id`.

Every write lands via a temp file plus `os.replace`, because a run killed
mid-write on Windows otherwise leaves half a JSON file behind.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.config import config_hash, dump_config
from experiments.seeding import seeds_for
from federated.confusion import balanced_accuracy, per_class_accuracy

TIMESTAMP_FMT = "%Y%m%dT%H%M%S"  # no colons; Windows rejects them in filenames


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FMT)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args), capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_provenance() -> dict:
    """Commit and dirty flag, so two numbers can be traced to the same code."""
    return {
        "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def file_lock(path: Path, poll_s: float = 0.01):
    """Hold an exclusive inter-process lock on `{path}.lock` for the block.

    Several training processes (`run_training --shard`) append to one runs.jsonl.
    Appending on Windows is a seek to the end and a write, not one atomic step, so two
    processes can interleave; every append and every read of a jsonl file takes this
    lock. It is a byte-range lock on the lock file (msvcrt on Windows, flock
    elsewhere), released by the operating system if the holder dies, so a killed run
    never leaves it held.
    """
    lock = Path(f"{path}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as fh:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(poll_s)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    with file_lock(path), path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with file_lock(path), path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def partition_id(config, partition_seed: int) -> str:
    """Hash of exactly the fields that determine the split.

    Fields added after the first schema enter the payload only when non-default, so every
    partition_id recorded before them stays valid.
    """
    p = config.partition
    fields = [
        config.dataset.name,
        p.K,
        p.mode.value,
        p.alpha,
        p.beta,
        p.min_size_policy,
        partition_seed,
    ]
    if p.size_cap != "none":
        fields.append({"size_cap": p.size_cap})
    if p.quantity_floor is not None:
        fields.append({"quantity_floor": p.quantity_floor})
    if p.class_order != "index":
        fields.append({"class_order": p.class_order})
    payload = json.dumps(fields, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def partition_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "partitions"


def partition_exists(out_dir: str | Path, pid: str) -> bool:
    return (partition_dir(out_dir) / f"{pid}.npz").exists()


def save_partition(
    out_dir: str | Path, pid: str, client_indices: list[np.ndarray], row: dict
) -> Path:
    """Write the .npz, its metrics sidecar, and append the partitions.jsonl row."""
    directory = partition_dir(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npz = directory / f"{pid}.npz"
    tmp = npz.with_name(npz.name + ".tmp")
    # Write through a handle: np.savez appends '.npz' to a path that lacks it.
    with tmp.open("wb") as fh:
        np.savez(fh, **{f"client_{i:03d}": idx for i, idx in enumerate(client_indices)})
    os.replace(tmp, npz)

    atomic_write_text(directory / f"{pid}.json", json.dumps(row, indent=2))
    append_jsonl(Path(out_dir) / "partitions.jsonl", row)
    return npz


def record_infeasible(out_dir: str | Path, pid: str, row: dict) -> None:
    """Record a cell that has no partition: sidecar JSON and jsonl row, no .npz."""
    atomic_write_text(partition_dir(out_dir) / f"{pid}.json", json.dumps(row, indent=2))
    append_jsonl(Path(out_dir) / "partitions.jsonl", row)


def infeasible_recorded(out_dir: str | Path, pid: str, max_redraws: int) -> bool:
    """True if this cell already exhausted the same redraw budget."""
    sidecar = partition_dir(out_dir) / f"{pid}.json"
    if not sidecar.exists():
        return False
    row = json.loads(sidecar.read_text(encoding="utf-8"))
    return row.get("status") == "infeasible_min_size" and row.get("max_redraws") == max_redraws


def load_partition(out_dir: str | Path, pid: str) -> list[np.ndarray]:
    with np.load(partition_dir(out_dir) / f"{pid}.npz") as data:
        return [data[key] for key in sorted(data.files)]


HISTORY_COLUMNS = (
    "round",
    "global_test_acc",
    "global_test_loss",
    "train_loss",
    "round_wall_s",
)
CLIENT_COLUMNS = ("round", "client", "num_samples", "train_loss", "steps")
CONFUSION_COLUMNS = ("round", "true_class", "predicted_class", "count")


def per_class_columns(num_classes: int) -> tuple[str, ...]:
    """per_class.csv: the round, balanced accuracy, then each class's accuracy."""
    return ("round", "balanced_acc", *(f"acc_class_{c}" for c in range(num_classes)))


RTA_FRACTION = 0.9
# Accuracies are counts over a test set, so a tie with the threshold is exact in theory but
# can miss by one float rounding (221 of the 1000 exact ties on a 10,000-sample test set).
# 1e-9 is far below one test sample (1e-4 here, about 1e-5 for a 20% test split of Covtype).
_RTA_TOLERANCE = 1e-9


def rounds_to_accuracy(
    accuracies, fraction: float = RTA_FRACTION, reference: float | None = None
) -> int | None:
    """Rounds-to-accuracy: the first round whose accuracy is at least fraction * reference.

    `accuracies` are the global model's test accuracies after rounds 1, 2, ...; the
    result is 1-indexed. `reference` defaults to the best of them, so with the
    default fraction this is the minimum number of communication rounds needed to
    reach 90% of the maximum accuracy the run achieved. None if `accuracies` is
    empty, or if an explicit reference is never reached.

    The metric is the source paper's (Jimenez-Gutierrez et al., arXiv:2503.17070 v2,
    Section 4.3). It cites Wu & Wang 2021 for it, but Wu & Wang count rounds to a fixed
    target accuracy (95% on MNIST, 80% on FashionMNIST; arXiv:2012.00661, Section V);
    the target relative to the maximum is the source paper's. The source paper's
    results text and tables take each algorithm's own maximum ("how many rounds each
    aggregation algorithm requires to achieve 90% of its maximum accuracy", Section
    5.3 in v2, V-A in v1), which is the default here. The definition added in v2's
    Section 4.3 says "the maximum accuracy among the aggregation algorithms"; pass
    that maximum as `reference` for this reading.
    """
    accuracies = list(accuracies)
    if not accuracies:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    threshold = fraction * (max(accuracies) if reference is None else reference)
    for round, accuracy in enumerate(accuracies, 1):
        if accuracy >= threshold - _RTA_TOLERANCE:
            return round
    return None


def torch_provenance() -> dict:
    """Which torch/GPU produced a number. Absent torch is recorded, not raised."""
    try:
        import torch
    except ImportError:
        return {"torch_version": None, "cuda_version": None, "gpu_name": None}
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def runs_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "runs.jsonl"


def completed_runs(out_dir: str | Path) -> set[str]:
    """config_hashes whose latest row says `completed` - the resume key."""
    latest: dict[str, str] = {}
    for row in read_jsonl(runs_path(out_dir)):
        latest[row["config_hash"]] = row["status"]
    return {h for h, status in latest.items() if status == "completed"}


def skip_if_exists(config, out_dir: str | Path | None = None) -> bool:
    """True when this exact config has already run to completion."""
    return config_hash(config) in completed_runs(out_dir or config.runtime.out_dir)


class RunRecorder:
    """Writes one run's rows, history, per-client log and resolved config.

    history.csv has one row per round. An optional round 0 (`log_initial`) is the
    initialised model before any training; it has no train loss or wall time and
    does not count towards final_acc, best_acc or rta. clients.csv has one row per
    client that trained in a round (`log_clients`).

    With `per_class=True`, each round's confusion matrix on the test set
    (`log_confusion`, round 0 included) goes to confusion.csv, one row per (true,
    predicted) class pair, and per_class.csv gets a row with each class's accuracy and
    the balanced accuracy (`federated.confusion`; balanced accuracy is the thesis's
    addition, not the source paper's). Both runs.jsonl rows then also carry
    final_balanced_acc and best_balanced_acc, which, like final_acc and best_acc, leave
    round 0 out.

    A `label` (a grid's own, e.g. that it is a post-hoc check) goes into both runs.jsonl
    rows as `label`; rows of runs without one have no such field. It is not part of the
    config, so it never moves a config_hash.

    Use as a context manager: an exception is recorded as `failed` (`oom` for a
    CUDA OOM) and re-raised, so a crashed run never looks completed.
    """

    def __init__(
        self,
        config,
        partition_id: str,
        out_dir: str | Path | None = None,
        per_class: bool = False,
        label: str | None = None,
    ):
        self.config = config
        self.partition_id = partition_id
        self.run_id = config_hash(config)
        self.out_dir = Path(out_dir or config.runtime.out_dir)
        self.dir = self.out_dir / "runs" / self.run_id
        self.accuracies: list[float] = []  # rounds 1, 2, ...; round 0 is not in it
        self.label = label
        self.per_class = per_class
        self.balanced_accuracies: list[float] = []  # rounds 1, 2, ..., as accuracies
        self._files = []
        self._history = None
        self._clients = None
        self._per_class = None
        self._confusion = None
        self._started = 0.0
        self._started_at = ""

    def _open_csv(self, name: str, columns) -> csv.DictWriter:
        handle = (self.dir / name).open("w", newline="", encoding="utf-8")
        self._files.append(handle)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        handle.flush()
        return writer

    def _flush(self) -> None:
        for handle in self._files:
            handle.flush()  # a kill -9 must still leave every finished round

    def start(self) -> RunRecorder:
        self.dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.config, self.dir / "config.yaml")
        self._started, self._started_at = time.perf_counter(), utc_stamp()
        self._history = self._open_csv("history.csv", HISTORY_COLUMNS)
        self._clients = self._open_csv("clients.csv", CLIENT_COLUMNS)
        if self.per_class:
            columns = per_class_columns(self.config.dataset.num_classes)
            self._per_class = self._open_csv("per_class.csv", columns)
            self._confusion = self._open_csv("confusion.csv", CONFUSION_COLUMNS)
        self._append_row("running")
        return self

    def log_initial(self, global_test_acc: float, global_test_loss: float) -> None:
        """Round 0: the initialised global model, evaluated before the first round."""
        if self.accuracies:
            raise RuntimeError("round 0 must be logged before round 1")
        self._history.writerow(
            {"round": 0, "global_test_acc": global_test_acc, "global_test_loss": global_test_loss}
        )
        self._flush()

    def log_round(
        self,
        round: int,
        global_test_acc: float,
        global_test_loss: float,
        train_loss: float,
        round_wall_s: float,
    ) -> None:
        self.accuracies.append(global_test_acc)
        self._history.writerow(
            {
                "round": round,
                "global_test_acc": global_test_acc,
                "global_test_loss": global_test_loss,
                "train_loss": train_loss,
                "round_wall_s": round_wall_s,
            }
        )
        self._flush()

    def log_clients(self, round: int, clients, num_samples, train_loss, steps) -> None:
        """One clients.csv row per client that trained this round, in the order given."""
        rows = list(zip(clients, num_samples, train_loss, steps, strict=True))
        self._clients.writerows(dict(zip(CLIENT_COLUMNS, (round, *row))) for row in rows)
        self._flush()

    def log_confusion(self, round: int, confusion) -> None:
        """The global model's test-set confusion matrix after `round` (0: before training).

        Rows are the true class, columns the predicted one. Writes the matrix to
        confusion.csv and the per-class and balanced accuracy to per_class.csv.
        """
        if not self.per_class:
            raise RuntimeError("this recorder was opened without per_class")
        confusion = np.asarray(confusion)
        num_classes = self.config.dataset.num_classes
        if confusion.shape != (num_classes, num_classes):
            raise ValueError(f"expected a {num_classes} x {num_classes} confusion matrix")
        balanced = balanced_accuracy(confusion)
        if round >= 1:
            self.balanced_accuracies.append(balanced)
        accuracy = per_class_accuracy(confusion)
        self._per_class.writerow(
            {
                "round": round,
                "balanced_acc": balanced,
                **{f"acc_class_{c}": float(a) for c, a in enumerate(accuracy)},
            }
        )
        self._confusion.writerows(
            {"round": round, "true_class": t, "predicted_class": p, "count": int(confusion[t, p])}
            for t in range(num_classes)
            for p in range(num_classes)
        )
        self._flush()

    def finish(self, status: str = "completed") -> None:
        for handle in self._files:
            handle.close()
        self._files = []
        self._append_row(status)

    def _append_row(self, status: str) -> None:
        done = status != "running"
        federated, runtime = self.config.federated, self.config.runtime
        seeds = seeds_for(self.config)
        append_jsonl(
            runs_path(self.out_dir),
            {
                "run_id": self.run_id,
                "partition_id": self.partition_id,
                "config_hash": self.run_id,
                "algorithm": federated.algorithm,
                "mu": federated.mu,
                "model": self.config.model.arch,
                "rounds": federated.rounds,
                "local_epochs": federated.local_epochs,
                "batch_size": federated.batch_size,
                "lr": federated.lr,
                "trial": runtime.trial,
                "init_seed": seeds.init,
                "train_seed": seeds.train,
                "deterministic": runtime.deterministic,
                "final_acc": self.accuracies[-1] if self.accuracies else None,
                "best_acc": max(self.accuracies) if self.accuracies else None,
                "rta": rounds_to_accuracy(self.accuracies),
                **self._balanced_fields(),
                **({} if self.label is None else {"label": self.label}),
                "wall_clock_s": round(time.perf_counter() - self._started, 3) if done else None,
                "status": status,
                **git_provenance(),
                **torch_provenance(),
                "started_at": self._started_at,
                "finished_at": utc_stamp() if done else None,
            },
        )

    def _balanced_fields(self) -> dict:
        """final_balanced_acc and best_balanced_acc for a per-class run; nothing otherwise."""
        if not self.per_class:
            return {}
        balanced = self.balanced_accuracies
        return {
            "final_balanced_acc": balanced[-1] if balanced else None,
            "best_balanced_acc": max(balanced) if balanced else None,
        }

    def __enter__(self) -> RunRecorder:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("completed")
        else:
            self.finish("oom" if _is_oom(exc) else "failed")
        return False


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
