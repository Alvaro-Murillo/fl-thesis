"""Artifact store, atomic writes and provenance (work-plan task A4.3).

Partitions are content-addressed: `partition_id` hashes only the fields that
change the split, so FedAvg and FedProx at the same grid cell provably read the
same bytes, and changing `federated.rounds` does not regenerate anything.

Every write lands via a temp file plus `os.replace`, because a run killed
mid-write on Windows otherwise leaves half a JSON file behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    """Hash of exactly the fields that determine the split."""
    p = config.partition
    payload = json.dumps([config.dataset.name, p.K, p.mode.value, p.alpha, p.beta,
                          p.min_size_policy, partition_seed], separators=(",", ":"))
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


def load_partition(out_dir: str | Path, pid: str) -> list[np.ndarray]:
    with np.load(partition_dir(out_dir) / f"{pid}.npz") as data:
        return [data[key] for key in sorted(data.files)]
