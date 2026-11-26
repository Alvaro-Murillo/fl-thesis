"""Per-class test accuracy of recorded runs, by retraining them bit for bit.

    python -m experiments.per_class_accuracy --grid GRID [--where KEY=VALUE ...] [--list]
    python -m experiments.per_class_accuracy --import DIR [--out-dir DIR]

A run's history.csv holds the global model's overall test accuracy only. This module
retrains recorded federated runs with the calls `experiments.run_training.train` makes,
in the same order, minus its RunRecorder: the stored partition, `seeds_for`,
`configure_torch`, `build_model`, the round-0 `evaluate`, `FederatedData`, then
`run_federated` with `fedavg`, `optimizer_factory` and `proximal_mu` from run_training.
After each round, and for the untrained model, it keeps the confusion matrix of the
global model on the full test set (rows true class, columns predicted class). The
per-round model comes from `on_round`, after the round's aggregation and evaluation; the
pass is a forward pass in eval mode under no_grad, at `evaluate`'s batch size, so it
changes no state and draws no random number. The final model is `model` after
`run_federated` returns.

The confusion matrix and per-class accuracy are `federated.confusion`'s, which
`experiments.run_training` now also records every round (per_class.csv, confusion.csv), so
the reruns are for runs recorded before that. `recorded_per_class` reads a run's own
files, with the checks below adapted to them, and `per_class_result` reads either kind.

--grid selects the configs of a grid; --where narrows them by dotted-key equality, with
the grid file's own keys and YAML values (`partition.target_label_hd=0.7`). Each must
have a completed row in runs.jsonl. A centralized config has no clients and is refused.

A rerun is bit-exact when all of these hold:
    - its partition_id is the recorded run's;
    - its final accuracy equals the completed runs.jsonl row's final_acc;
    - its accuracy after every round, round 0 included, equals history.csv (read
      round-trip);
    - every round's confusion matrix gives that round's accuracy, and the final model's
      matrix is the last round's.
A rerun that is not bit-exact is saved with `bit_exact: false`, the sweep stops, and no
later sweep starts while such a file exists. `load_per_class` raises for it, so nothing
reads per-class results from a run that did not reproduce.

A run trained with per-class logging is read from its own run folder instead
(`recorded_per_class`). Its accuracies are history.csv's, and it is refused unless:
    - confusion.csv holds exactly one full matrix for each round of history.csv, round 0
      included, and per_class.csv one row for each;
    - every matrix gives that round's history.csv accuracy;
    - the last round's accuracy is the completed runs.jsonl row's final_acc, and its
      balanced accuracy the row's final_balanced_acc;
    - per_class.csv's per-class and balanced accuracies are the matrices', every round.
Where a run has both its own files and a rerun, the two must hold the same matrices.

A sweep refuses to start while a run is `running` in runs.jsonl, and stops if runs.jsonl
changes before a rerun: per-class results must come from the recorded runs, with nothing
else training. A run whose output exists is skipped, so a stopped sweep resumes.

--import copies per-class results made earlier by standalone scripts with the same code
path (a folder of JSON files, one per run, with `run_id`, `partition_id`, `accs` and
`confusions`). Every check above is made again against runs.jsonl and history.csv before
a file is copied; the file's own provenance (git commit, torch, GPU) is kept.

Writes {out_dir}/per_class/{run_id}.json, one file per run, atomically. Nothing else
under out_dir changes: no runs.jsonl row, no run folder.

CUBLAS_WORKSPACE_CONFIG must be set before torch is imported, so when this module runs as
a script it sets it before its other imports.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from experiments.seeding import enable_cublas_determinism

if __name__ == "__main__":
    enable_cublas_determinism()  # before any import below brings in torch

import argparse
import json
import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.config import ExperimentConfig, config_hash
from experiments.grid import expand
from experiments.recording import (
    atomic_write_text,
    git_provenance,
    read_jsonl,
    runs_path,
    torch_provenance,
    utc_stamp,
)
from experiments.run_training import describe, optimizer_factory, proximal_mu, stored_partition
from experiments.seeding import configure_torch, seeds_for
from experiments.tensors import TensorData, load_tensors
from federated.aggregation import fedavg
from federated.confusion import (
    EVAL_BATCH,
    balanced_accuracy,
    confusion_matrix,
    per_class_accuracy,
)
from federated.data import FederatedData
from federated.loop import RoundResult, evaluate, run_federated
from models.networks import build_model

__all__ = ["EVAL_BATCH", "confusion_matrix", "per_class_accuracy"]  # re-exported for callers
CHECKS = (
    "partition_id_equal",
    "final_acc_equal",
    "all_rounds_equal_history",
    "confusions_match_accuracy",
)


class NotBitExact(RuntimeError):
    """A rerun, an imported result or a run's own per-class files do not match its recorded run."""


def per_class_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "per_class"


def output_path(out_dir: str | Path, run_id: str) -> Path:
    return per_class_dir(out_dir) / f"{run_id}.json"


def load_per_class(out_dir: str | Path, run_id: str) -> dict:
    """The stored per-class result of `run_id`; raises unless it is bit-exact."""
    path = output_path(out_dir, run_id)
    if not path.exists():
        raise FileNotFoundError(f"no per-class result for run {run_id} in {path.parent}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("bit_exact") is not True:
        failed = [check for check in CHECKS if payload.get(check) is not True]
        raise NotBitExact(f"run {run_id}'s per-class result is not bit-exact: {failed}")
    return payload


def matches_accuracy(confusions, accs) -> bool:
    """Each matrix's share of correct predictions is the recorded accuracy, as a float."""
    return len(confusions) == len(accs) and all(
        int(np.trace(cm)) / int(np.sum(cm)) == acc
        for cm, acc in zip(map(np.asarray, confusions), accs)
    )


def rerun(config: ExperimentConfig, tensors: TensorData | None = None, on_round=None) -> dict:
    """Retrain a federated config as run_training.train does, keeping every round's confusion."""
    if config.partition is None:
        raise ValueError(
            "a centralized run has no clients; per-class reruns are for federated runs"
        )
    if config.dataset.val_split != 0.0:
        raise NotImplementedError("no validation split is implemented; set dataset.val_split: 0")
    device = config.runtime.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but no CUDA device is available")
    pid, client_indices = stored_partition(config)
    seeds = seeds_for(config)
    configure_torch(seeds, config.runtime.deterministic)
    if tensors is None:
        tensors = load_tensors(config.dataset, device)
    model = build_model(config.model, tensors.input_shape, seeds.init).to(device)
    federated, num_classes = config.federated, config.dataset.num_classes
    test_x, test_y = tensors.test_x, tensors.test_y

    accs = [evaluate(model, test_x, test_y)[0]]  # round 0, as train() logs it
    confusions = [confusion_matrix(model, test_x, test_y, num_classes)]
    data = FederatedData(tensors.train_x, tensors.train_y, client_indices)

    def record(result: RoundResult) -> None:
        accs.append(result.test_acc)
        confusions.append(confusion_matrix(model, test_x, test_y, num_classes))
        if on_round is not None:
            on_round(result)

    started, started_at = time.perf_counter(), utc_stamp()
    run_federated(
        model,
        data,
        test_x,
        test_y,
        rounds=federated.rounds,
        local_epochs=federated.local_epochs,
        batch_size=federated.batch_size,
        make_optimizer=optimizer_factory(federated),
        aggregate=fedavg,
        train_seed=seeds.train,
        client_fraction=federated.client_fraction,
        proximal_mu=proximal_mu(federated),
        on_round=record,
    )
    wall = time.perf_counter() - started
    final = confusion_matrix(model, test_x, test_y, num_classes)  # the returned global model
    return {
        "partition_id": pid,
        "trial": config.runtime.trial,
        "seeds": {"partition": seeds.partition, "init": seeds.init, "train": seeds.train},
        "accs": accs,
        "confusions": [cm.tolist() for cm in confusions],
        "final_model_is_last_round": bool(np.array_equal(final, confusions[-1])),
        "wall_clock_s": round(wall, 3),
        "started_at": started_at,
        "finished_at": utc_stamp(),
    }


def completed_rows(out_dir: str | Path) -> dict[str, dict]:
    """run_id -> latest runs.jsonl row, for runs whose latest row is `completed`."""
    latest = {row["run_id"]: row for row in read_jsonl(runs_path(out_dir))}
    return {run_id: row for run_id, row in latest.items() if row["status"] == "completed"}


def recorded_history(out_dir: str | Path, run_id: str) -> list[float]:
    """Test accuracy after rounds 0, 1, ..., as history.csv holds it."""
    history = pd.read_csv(
        Path(out_dir) / "runs" / run_id / "history.csv", float_precision="round_trip"
    )
    return history.sort_values("round")["global_test_acc"].tolist()


def _matrices(table: pd.DataFrame, rounds: list[int], num_classes: int) -> list | None:
    """One matrix per round from confusion.csv's rows, or None unless each round is one full matrix."""
    if sorted(table["round"].unique().tolist()) != rounds:
        return None
    pairs = [(t, p) for t in range(num_classes) for p in range(num_classes)]
    out = []
    for r in rounds:
        part = table[table["round"] == r]
        true, predicted = part["true_class"].to_numpy(), part["predicted_class"].to_numpy()
        if sorted(zip(true.tolist(), predicted.tolist())) != pairs:
            return None
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        matrix[true, predicted] = part["count"].to_numpy()
        out.append(matrix)
    return out


def recorded_per_class(out_dir: str | Path, run_id: str) -> dict:
    """The per-class result a run recorded itself, checked against its recorded run.

    Raises FileNotFoundError when the run folder has no confusion.csv, RuntimeError when
    the run has no completed runs.jsonl row, and NotBitExact when a check fails.
    """
    run_dir = Path(out_dir) / "runs" / run_id
    path = run_dir / "confusion.csv"
    if not path.exists():
        raise FileNotFoundError(f"run {run_id} recorded no confusion.csv in {run_dir}")
    row = completed_rows(out_dir).get(run_id)
    if row is None:
        raise RuntimeError(f"run {run_id} has no completed row in runs.jsonl")
    history = recorded_history(out_dir, run_id)
    rounds = list(range(len(history)))
    per_class = pd.read_csv(run_dir / "per_class.csv", float_precision="round_trip")
    per_class = per_class.sort_values("round")
    columns = [c for c in per_class.columns if c.startswith("acc_class_")]
    confusions = _matrices(pd.read_csv(path), rounds, len(columns))
    checks = {
        "one_matrix_per_round": confusions is not None and per_class["round"].tolist() == rounds
    }
    if checks["one_matrix_per_round"]:
        checks.update(
            {
                "confusions_match_accuracy": matches_accuracy(confusions, history),
                "final_acc_equal": history[-1] == row["final_acc"],
                "final_balanced_acc_equal": (
                    balanced_accuracy(confusions[-1]) == row.get("final_balanced_acc")
                ),
                "per_class_csv_matches": np.array_equal(
                    per_class[columns].to_numpy(),
                    np.array([per_class_accuracy(m) for m in confusions]),
                    equal_nan=True,
                )
                and per_class["balanced_acc"].tolist()
                == [balanced_accuracy(m) for m in confusions],
            }
        )
    failed = [check for check, ok in checks.items() if not ok]
    if failed:
        raise NotBitExact(f"run {run_id}'s own per-class files do not match its record: {failed}")
    return {
        "run_id": run_id,
        "source": "recorded",
        "partition_id": row["partition_id"],
        "trial": row["trial"],
        "accs": history,
        "confusions": [m.tolist() for m in confusions],
        "recorded_final_acc": row["final_acc"],
        "git_commit": row.get("git_commit"),
        "git_dirty": row.get("git_dirty"),
        **checks,
    }


def per_class_result(out_dir: str | Path, run_id: str) -> dict:
    """A run's checked per-class result: its own files if it recorded them, else its rerun.

    Raises FileNotFoundError when it has neither.
    """
    try:
        recorded = recorded_per_class(out_dir, run_id)
    except FileNotFoundError:
        return load_per_class(out_dir, run_id)
    if output_path(out_dir, run_id).exists():
        rerun_result = load_per_class(out_dir, run_id)
        if rerun_result["confusions"] != recorded["confusions"]:
            raise NotBitExact(f"run {run_id}: its rerun's matrices differ from its own records")
    return recorded


def with_bit_exact_checks(payload: dict, row: dict, history: list[float]) -> dict:
    """The payload with each bit-exactness check against its recorded run, and the verdict."""
    accs = payload["accs"]
    checks = {
        "partition_id_equal": payload["partition_id"] == row["partition_id"],
        "final_acc_equal": accs[-1] == row["final_acc"],
        "all_rounds_equal_history": accs == history,
        "confusions_match_accuracy": matches_accuracy(payload["confusions"], accs)
        and payload.get("final_model_is_last_round", True),
    }
    return {
        **payload,
        "recorded_final_acc": row["final_acc"],
        "recorded_wall_clock_s": row["wall_clock_s"],
        "recorded_git_commit": row.get("git_commit"),
        **checks,
        "bit_exact": all(checks.values()),
    }


def runs_state(out_dir: str | Path) -> tuple[int, list[str]]:
    """(rows in runs.jsonl, run_ids whose latest row is `running`)."""
    rows = read_jsonl(runs_path(out_dir))
    latest = {row["run_id"]: row["status"] for row in rows}
    return len(rows), sorted(run_id for run_id, status in latest.items() if status == "running")


def refuse_after_a_failure(out_dir: str | Path) -> None:
    for path in sorted(per_class_dir(out_dir).glob("*.json")):
        if json.loads(path.read_text(encoding="utf-8")).get("bit_exact") is not True:
            raise NotBitExact(f"{path} is not bit-exact; not continuing until it is resolved")


def sweep(
    configs: Sequence[ExperimentConfig], log: Callable[[str], None] = print
) -> tuple[int, int]:
    """Rerun every config not yet done. Returns (rerun, skipped because their output exists)."""
    out_dirs = {config.runtime.out_dir for config in configs}
    if len(out_dirs) != 1:
        raise ValueError(f"the configs write to {len(out_dirs)} out_dirs: {sorted(out_dirs)}")
    out_dir = out_dirs.pop()
    refuse_after_a_failure(out_dir)
    rows_at_start, running = runs_state(out_dir)
    if running:
        raise RuntimeError(
            f"runs.jsonl has runs in state running: {running}; is something training?"
        )
    recorded = completed_rows(out_dir)
    missing = [config_hash(c) for c in configs if config_hash(c) not in recorded]
    if missing:
        raise RuntimeError(f"no completed run for {missing}; train them first")
    tensors: dict[tuple, TensorData] = {}
    done = 0
    for i, config in enumerate(configs, 1):
        run_id = config_hash(config)
        path = output_path(out_dir, run_id)
        head = f"[{i}/{len(configs)}] {describe(config)}"
        if path.exists():
            log(f"{head}  per-class result exists")
            continue
        if runs_state(out_dir) != (rows_at_start, []):
            raise RuntimeError("runs.jsonl changed during the sweep; is something training?")
        log(head)
        key = (config.dataset, config.runtime.device)
        if key not in tensors:
            tensors[key] = load_tensors(*key)
        rounds = config.federated.rounds
        out = rerun(
            config,
            tensors[key],
            lambda r: log(f"  round {r.round:>3}/{rounds}  acc {r.test_acc:.4f}"),
        )
        payload = with_bit_exact_checks(
            {"run_id": run_id, "source": "rerun", **out, **git_provenance(), **torch_provenance()},
            recorded[run_id],
            recorded_history(out_dir, run_id),
        )
        atomic_write_text(path, json.dumps(payload, indent=1))
        done += 1
        log(
            f"  final {payload['accs'][-1]:.4f}  recorded {payload['recorded_final_acc']:.4f}  "
            f"bit-exact {payload['bit_exact']}  {payload['wall_clock_s'] / 60:.1f} min"
        )
        if not payload["bit_exact"]:
            failed = [check for check in CHECKS if not payload[check]]
            raise NotBitExact(f"run {run_id} did not reproduce ({failed}); stopped. {path}")
    return done, len(configs) - done


def import_results(
    folder: str | Path, out_dir: str | Path, log: Callable[[str], None] = print
) -> int:
    """Check and copy per-class results made elsewhere; refuses the whole folder on one failure."""
    recorded = completed_rows(out_dir)
    payloads = []
    for path in sorted(Path(folder).glob("*.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        run_id = source["run_id"]
        if run_id not in recorded:
            raise RuntimeError(f"{path.name}: run {run_id} has no completed row in runs.jsonl")
        keep = ("git_commit", "git_dirty", "torch_version", "cuda_version", "gpu_name")
        payload = with_bit_exact_checks(
            {
                "run_id": run_id,
                "source": f"imported from {path.name}",
                "partition_id": source["partition_id"],
                "trial": source["trial"],
                "seeds": source["seeds"],
                "accs": source["accs"],
                "confusions": source["confusions"],
                # the earlier scripts listed this comparison among their internal mismatches
                "final_model_is_last_round": not any(
                    "final model" in str(item) for item in source.get("internal_mismatch", [])
                ),
                "wall_clock_s": source["wall_clock_s"],
                "started_at": source["started_at"],
                "finished_at": source["finished_at"],
                **{field: source.get(field) for field in keep},
            },
            recorded[run_id],
            recorded_history(out_dir, run_id),
        )
        if not payload["bit_exact"]:
            failed = [check for check in CHECKS if not payload[check]]
            raise NotBitExact(
                f"{path.name} does not reproduce run {run_id} ({failed}); nothing imported"
            )
        payloads.append(payload)
    copied = 0
    for payload in payloads:
        path = output_path(out_dir, payload["run_id"])
        if path.exists():
            log(f"{payload['run_id']}  exists, not overwritten")
            continue
        atomic_write_text(path, json.dumps(payload, indent=1))
        copied += 1
        log(f"{payload['run_id']}  {payload['source']}")
    return copied


def _field(config: ExperimentConfig, key: str):
    section, name = key.split(".")
    value = getattr(getattr(config, section), name)
    return value.value if isinstance(value, Enum) else value


def filter_configs(
    configs: Sequence[ExperimentConfig], where: Sequence[str]
) -> list[ExperimentConfig]:
    """The configs whose dotted-key fields equal every KEY=VALUE given (values read as YAML)."""
    conditions = []
    for item in where:
        key, sep, value = item.partition("=")
        if not sep or key.count(".") != 1:
            raise ValueError(f"--where takes section.field=value, got {item!r}")
        conditions.append((key, yaml.safe_load(value)))
    chosen = [c for c in configs if all(_field(c, key) == value for key, value in conditions)]
    if not chosen:
        raise ValueError(f"no config matches {list(where)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", type=Path)
    source.add_argument("--import", dest="folder", type=Path, help="folder of per-class JSON files")
    parser.add_argument("--where", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="with --import")
    args = parser.parse_args()

    try:
        if args.folder is not None:
            print(f"{import_results(args.folder, args.out_dir)} files imported")
            return
        configs = filter_configs(expand(args.grid, for_training=False), args.where)
        if args.list:
            recorded = completed_rows(configs[0].runtime.out_dir)
            for i, config in enumerate(configs, 1):
                run_id = config_hash(config)
                row = recorded.get(run_id)
                done = output_path(config.runtime.out_dir, run_id).exists()
                print(
                    f"[{i}/{len(configs)}] {describe(config)}  recorded final_acc "
                    f"{row['final_acc'] if row else 'not trained'}  {'done' if done else 'pending'}"
                )
            return
        done, skipped = sweep(configs)
        print(f"{len(configs)} runs: {done} rerun, {skipped} already had a per-class result")
    except (NotBitExact, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
