"""Artifact store, atomic writes and provenance (work-plan tasks A4.3/A4.4).

Partitions are content-addressed: `partition_id` hashes only the fields that
change the split, so FedAvg and FedProx at the same grid cell provably read the
same bytes, and changing `federated.rounds` does not regenerate anything.

Runs append to `runs.jsonl` twice - once at start with status `running`, once
at the end with the outcome - so a killed run is visible as `running` next to
a partial but parseable `history.csv`. Readers take the last row per `run_id`.

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.config import config_hash, dump_config
from experiments.seeding import seeds_for

TIMESTAMP_FMT = "%Y%m%dT%H%M%S"  # no colons; Windows rejects them in filenames


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FMT)


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_provenance() -> dict:
    """Commit and dirty flag, so two numbers can be traced to the same code."""
    return {"git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
            "git_dirty": bool(_git("status", "--porcelain"))}


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def partition_id(config, partition_seed: int) -> str:
    """Hash of exactly the fields that determine the split.

    Fields added after A4 enter the payload only when non-default, so every
    partition_id recorded before them stays valid.
    """
    p = config.partition
    fields = [config.dataset.name, p.K, p.mode.value, p.alpha, p.beta,
              p.min_size_policy, partition_seed]
    if p.size_cap != "none":
        fields.append({"size_cap": p.size_cap})
    payload = json.dumps(fields, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def partition_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "partitions"


def partition_exists(out_dir: str | Path, pid: str) -> bool:
    return (partition_dir(out_dir) / f"{pid}.npz").exists()


def save_partition(out_dir: str | Path, pid: str,
                   client_indices: list[np.ndarray], row: dict) -> Path:
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


HISTORY_COLUMNS = ("round", "global_test_acc", "global_test_loss",
                   "mean_client_train_loss", "round_wall_s")


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
    """Writes one run's rows, history and resolved config.

    Use as a context manager: an exception is recorded as `failed` (`oom` for a
    CUDA OOM) and re-raised, so a crashed run never looks completed.
    """

    def __init__(self, config, partition_id: str, out_dir: str | Path | None = None,
                 rta_target: float | None = None):
        self.config = config
        self.partition_id = partition_id
        self.rta_target = rta_target
        self.run_id = config_hash(config)
        self.out_dir = Path(out_dir or config.runtime.out_dir)
        self.dir = self.out_dir / "runs" / self.run_id
        self.accuracies: list[float] = []
        self._history = None
        self._writer = None
        self._started = 0.0
        self._started_at = ""

    def start(self) -> RunRecorder:
        self.dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.config, self.dir / "config.yaml")
        self._started, self._started_at = time.perf_counter(), utc_stamp()
        self._history = (self.dir / "history.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._history, fieldnames=HISTORY_COLUMNS)
        self._writer.writeheader()
        self._history.flush()
        self._append_row("running")
        return self

    def log_round(self, round: int, global_test_acc: float, global_test_loss: float,
                  mean_client_train_loss: float, round_wall_s: float) -> None:
        self.accuracies.append(global_test_acc)
        self._writer.writerow({
            "round": round, "global_test_acc": global_test_acc,
            "global_test_loss": global_test_loss,
            "mean_client_train_loss": mean_client_train_loss,
            "round_wall_s": round_wall_s})
        self._history.flush()  # a kill -9 must still leave every finished round

    def finish(self, status: str = "completed") -> None:
        if self._history is not None:
            self._history.close()
            self._history = None
        self._append_row(status)

    def _rta(self) -> int | None:
        """First round reaching rta_target, 1-indexed; None if never reached."""
        if self.rta_target is None:
            return None
        hits = [i for i, acc in enumerate(self.accuracies, 1) if acc >= self.rta_target]
        return hits[0] if hits else None

    def _append_row(self, status: str) -> None:
        done = status != "running"
        federated, runtime = self.config.federated, self.config.runtime
        seeds = seeds_for(self.config)
        append_jsonl(runs_path(self.out_dir), {
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
            "rta": self._rta(),
            "wall_clock_s": round(time.perf_counter() - self._started, 3) if done else None,
            "status": status,
            **git_provenance(),
            **torch_provenance(),
            "started_at": self._started_at,
            "finished_at": utc_stamp() if done else None,
        })

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
