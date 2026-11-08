"""Train one config, or every cell of a grid: the federated runs and the centralized baseline.

    python -m experiments.run_training --config config/smoke.yaml
    python -m experiments.run_training --grid config/grids/rq3.yaml
    python -m experiments.run_training --grid config/grids/centralized.yaml

Training reads partitions and never makes them. A cell's partition must already
be in the store (`experiments.run_partition`, `experiments.run_grid`), so every
partition is generated, and can be validated, before any GPU time is spent. A
grid is checked as a whole before its first run: one missing partition, or one
recorded as infeasible, stops the sweep before anything is trained or written.
A grid marked `partition_only`, or with `status: blocked`, is refused when it is
expanded (`experiments.grid`).

How a config maps onto `federated.loop.run_federated`:

    dataset    the training and test tensors (`experiments.tensors`), on runtime.device
    partition  the stored partition with this config's partition_id, one index array
               per client
    model      `models.networks.build_model`, Glorot weights drawn from the init seed
    federated  rounds, local_epochs, batch_size and client_fraction as they are; a
               fresh optimizer (adam or sgd at lr, library defaults otherwise) for every
               client in every round; the server rule is `federated.aggregation.fedavg`
               for both algorithms (`federated/proximal.py` says why), and fedprox adds
               the proximal term with mu (`proximal_mu`; fedavg passes None, so no term
               is applied)
    runtime    seeds derived from trial (`experiments.seeding`); strict determinism
               unless deterministic is false

The thesis's training settings are in config/base.yaml. They are the source
paper's: Jimenez-Gutierrez et al. (2026), "A thorough assessment of the non-IID
data impact in federated learning", Section 4, "Models" (checked against
arXiv:2503.17070 v1 and v2): K = 30 clients, Adam with learning rate 0.001,
batch size 64, 40 communication rounds of 10 local epochs. The paper ran 5 or 10
seeds; the grids here run trials 0-2. It gives no participation fraction for
FedAvg or FedProx: a fraction is tuned for Rand only (Section 4.1), and its
FedProx "aggregates local model updates from all devices" (Section 3.4; section
numbers as in v2). So client_fraction 1.0 has every client train in every round.

Each round goes to `experiments.recording.RunRecorder`: a history.csv row with
the global model's test accuracy and loss, the trained clients' final-epoch
training losses averaged by sample count (`train_loss`), and the round's wall
time, and a clients.csv row per trained client (size, final-epoch loss, steps).
history.csv starts with round 0, the initialised model's test accuracy and loss,
which no metric counts. Every round, round 0 included, the global model's confusion
matrix on the test set (`federated.confusion`) goes to confusion.csv, and each
class's accuracy and the balanced accuracy (the mean of the per-class accuracies;
the thesis's addition, the source paper reports accuracy) to per_class.csv. It is
an extra forward pass in eval mode that changes no state and draws no random
number, so it leaves the training itself as it was. runs.jsonl gets a row at the
start and one at the end, with final_acc, best_acc, final_balanced_acc,
best_balanced_acc and rta: rounds to reach 90% of the run's own best accuracy
(`experiments.recording.rounds_to_accuracy`). A run that already
completed (same config_hash in runs.jsonl) is skipped unless --force is given; a
failed or killed run is trained again. A sweep loads each dataset once, keeps it
on the device for every run, and stops at the first run that fails.

A centralized config (`federated.algorithm: centralized`, `partition: null`) reads no
partition. `federated.centralized.train_centralized` trains the model on the whole
training set for local_epochs epochs with one optimizer and the run's train seed, and
the run is recorded the same way, with an epoch in place of a round: history.csv has
one row per epoch (round 0 is again the untrained model), clients.csv one row per
epoch for the single client, and rta counts epochs. Its init seed is a federated run's
at the same trial, so trial t starts from the same weights in both.

CUBLAS_WORKSPACE_CONFIG must be set before torch is imported, so when this
module runs as a script it sets it before its other imports.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from experiments.seeding import enable_cublas_determinism

if __name__ == "__main__":
    enable_cublas_determinism()  # before any import below brings in torch

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from experiments.config import ExperimentConfig, config_hash, load_config
from experiments.grid import expand
from experiments.recording import (
    RunRecorder,
    load_partition,
    partition_dir,
    partition_exists,
    partition_id,
    skip_if_exists,
)
from experiments.seeding import configure_torch, seeds_for
from experiments.tensors import TensorData, load_tensors
from federated.aggregation import fedavg
from federated.centralized import EpochResult, train_centralized
from federated.confusion import confusion_matrix
from federated.data import FederatedData
from federated.loop import MakeOptimizer, RoundResult, evaluate, run_federated
from models.networks import build_model

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}  # mirrors config.OPTIMIZERS


class PartitionUnavailable(RuntimeError):
    """A cell's partition is not in the store, or was recorded as infeasible."""


def optimizer_factory(federated) -> MakeOptimizer:
    """Builds a new optimizer at `federated.lr` on every call; nothing else is set."""
    optimizer = OPTIMIZERS[federated.optimizer]
    return lambda params: optimizer(params, lr=federated.lr)


def proximal_mu(federated) -> float | None:
    """FedProx's mu; None for FedAvg, so its clients add no term at all."""
    return federated.mu if federated.algorithm == "fedprox" else None


def train_loss(result: RoundResult) -> float:
    """The trained clients' final-epoch losses averaged with weights n_k / sum n_j.

    This is the mean per-sample loss over every sample that was trained on, the
    same weighting as FedAvg's objective f(w) = sum_k (n_k / n) F_k(w) (McMahan et
    al. 2017, Section 1, "Federated Optimization") and as Flower 1.36.0's default
    aggregation of training metrics (weighted by "num-examples"). clients.csv keeps
    each client's own loss.
    """
    total = sum(result.client_sizes)
    return sum(n * loss for n, loss in zip(result.client_sizes, result.client_train_loss)) / total


def confusion_on_test_set(model: torch.nn.Module, tensors: TensorData, config) -> np.ndarray:
    """The model's confusion matrix on the test set (`federated.confusion`)."""
    return confusion_matrix(model, tensors.test_x, tensors.test_y, config.dataset.num_classes)


def partition_problem(config: ExperimentConfig) -> str | None:
    """Why this config's partition cannot be trained on; None if it is in the store.

    None for a centralized config, which reads no partition.
    """
    if config.partition is None:
        return None
    pid = partition_id(config, seeds_for(config).partition)
    out_dir = config.runtime.out_dir
    if partition_exists(out_dir, pid):
        return None
    sidecar = partition_dir(out_dir) / f"{pid}.json"
    if sidecar.exists():
        status = json.loads(sidecar.read_text(encoding="utf-8")).get("status")
        return f"partition {pid} is recorded as {status}, with no client data"
    return f"partition {pid} is not in {partition_dir(out_dir)}"


def stored_partition(config: ExperimentConfig) -> tuple[str, list[np.ndarray]]:
    """(partition_id, client index arrays) for this config, read from the store."""
    problem = partition_problem(config)
    if problem is not None:
        raise PartitionUnavailable(
            f"{problem}; generate it with experiments.run_partition or experiments.run_grid"
        )
    pid = partition_id(config, seeds_for(config).partition)
    return pid, load_partition(config.runtime.out_dir, pid)


def train(
    config: ExperimentConfig,
    tensors: TensorData | None = None,
    on_round: Callable[[RoundResult], None] | Callable[[EpochResult], None] | None = None,
) -> list[RoundResult] | list[EpochResult]:
    """Train one config and record it, whether or not it ran before.

    `tensors` is the config's dataset on runtime.device, loaded here if None.
    Nothing is recorded if the partition is unavailable; once training starts, an
    exception is recorded as a failed run and re-raised. A centralized config
    returns, and passes to `on_round`, one EpochResult per epoch.
    """
    if config.dataset.val_split != 0.0:
        raise NotImplementedError("no validation split is implemented; set dataset.val_split: 0")
    device = config.runtime.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but no CUDA device is available")
    centralized = config.partition is None
    pid, client_indices = (None, None) if centralized else stored_partition(config)
    seeds = seeds_for(config)
    configure_torch(seeds, config.runtime.deterministic)
    if tensors is None:
        tensors = load_tensors(config.dataset, device)
    model = build_model(config.model, tensors.input_shape, seeds.init).to(device)
    federated = config.federated

    with RunRecorder(config, pid, per_class=True) as recorder:
        recorder.log_initial(*evaluate(model, tensors.test_x, tensors.test_y))
        recorder.log_confusion(0, confusion_on_test_set(model, tensors, config))
        if centralized:
            return _train_centralized(model, tensors, federated, seeds.train, recorder, on_round)
        data = FederatedData(tensors.train_x, tensors.train_y, client_indices)

        def record(result: RoundResult) -> None:
            recorder.log_round(
                result.round, result.test_acc, result.test_loss, train_loss(result), result.wall_s
            )
            recorder.log_clients(
                result.round,
                result.clients,
                result.client_sizes,
                result.client_train_loss,
                result.client_steps,
            )
            recorder.log_confusion(result.round, confusion_on_test_set(model, tensors, config))
            if on_round is not None:
                on_round(result)

        return run_federated(
            model,
            data,
            tensors.test_x,
            tensors.test_y,
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


def _train_centralized(
    model: torch.nn.Module,
    tensors: TensorData,
    federated,
    train_seed: int,
    recorder: RunRecorder,
    on_epoch: Callable[[EpochResult], None] | None,
) -> list[EpochResult]:
    """The centralized baseline, each epoch recorded where a federated run records a round."""
    n_train = len(tensors.train_y)

    def record(result: EpochResult) -> None:
        recorder.log_round(
            result.epoch, result.test_acc, result.test_loss, result.train_loss, result.wall_s
        )
        recorder.log_clients(result.epoch, [0], [n_train], [result.train_loss], [result.steps])
        recorder.log_confusion(result.epoch, confusion_on_test_set(model, tensors, recorder.config))
        if on_epoch is not None:
            on_epoch(result)

    return train_centralized(
        model,
        tensors.train_x,
        tensors.train_y,
        tensors.test_x,
        tensors.test_y,
        epochs=federated.local_epochs,
        batch_size=federated.batch_size,
        make_optimizer=optimizer_factory(federated),
        train_seed=train_seed,
        on_epoch=record,
    )


def describe(config: ExperimentConfig) -> str:
    p, f = config.partition, config.federated
    if p is None:
        return (
            f"{config_hash(config)}  centralized {f.local_epochs} epochs  "
            f"trial={config.runtime.trial}"
        )
    knobs = " ".join(
        f"{name}={value:g}" for name, value in (("alpha", p.alpha), ("beta", p.beta)) if value
    )
    algorithm = f"fedprox mu={f.mu:g}" if f.algorithm == "fedprox" else f.algorithm
    pid = partition_id(config, seeds_for(config).partition)
    return (
        f"{config_hash(config)}  partition {pid}  {p.mode.value} K={p.K} {knobs}  "
        f"{algorithm}  trial={config.runtime.trial}"
    )


def describe_round(result: RoundResult, rounds: int) -> str:
    return (
        f"  round {result.round:>3}/{rounds}  acc {result.test_acc:.4f}  "
        f"loss {result.test_loss:.4f}  {result.wall_s:.1f} s"
    )


def describe_epoch(result: EpochResult, epochs: int) -> str:
    return (
        f"  epoch {result.epoch:>3}/{epochs}  acc {result.test_acc:.4f}  "
        f"loss {result.test_loss:.4f}  {result.wall_s:.1f} s"
    )


def train_all(
    configs: Sequence[ExperimentConfig],
    force: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Train every config in order. Returns (trained, skipped as already completed).

    Every partition the sweep needs is checked before the first run. Datasets are
    loaded once per (dataset, device) and reused across runs.
    """
    done = [not force and skip_if_exists(config) for config in configs]
    pending = [config for config, skip in zip(configs, done) if not skip]
    problems = [problem for problem in map(partition_problem, pending) if problem is not None]
    if problems:
        raise PartitionUnavailable(
            f"{len(problems)} of {len(pending)} runs have no partition to train on, e.g. "
            f"{problems[0]}; generate them with experiments.run_partition or experiments.run_grid"
        )
    tensors: dict[tuple, TensorData] = {}
    for i, (config, skip) in enumerate(zip(configs, done), 1):
        head = f"[{i}/{len(configs)}] {describe(config)}"
        if skip:
            log(f"{head}  already completed")
            continue
        log(head)
        key = (config.dataset, config.runtime.device)
        if key not in tensors:
            tensors[key] = load_tensors(*key)
        centralized, f = config.partition is None, config.federated
        describe_step = describe_epoch if centralized else describe_round
        steps = f.local_epochs if centralized else f.rounds
        train(config, tensors[key], lambda result: log(describe_step(result, steps)))
    return len(pending), len(configs) - len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--grid", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="train again even if the run already completed"
    )
    args = parser.parse_args()

    configs = [load_config(args.config)] if args.config else expand(args.grid, for_training=True)
    trained, skipped = train_all(configs, force=args.force)
    print(f"{len(configs)} runs: {trained} trained, {skipped} already completed")


if __name__ == "__main__":
    main()
