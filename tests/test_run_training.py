"""Tests for the training runner: config -> loop mapping, the partition gate, resume.

The fast tests train the real CNN on CPU on a toy dataset of random 32x32 images
(3 classes, 3 clients). The slow test runs the command line on real CIFAR-10 on
the GPU.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments import datasets, run_training, tensors
from experiments.config import config_hash, dump_config, from_dict, load_config, to_dict
from experiments.recording import (
    load_partition,
    partition_id,
    read_jsonl,
    record_infeasible,
    rounds_to_accuracy,
    runs_path,
    skip_if_exists,
)
from experiments.run_partition import generate_partition
from experiments.run_training import (
    PartitionUnavailable,
    optimizer_factory,
    proximal_mu,
    train,
    train_all,
    train_loss,
)
from experiments.seeding import seeds_for
from federated.aggregation import fedavg
from federated.centralized import train_centralized
from federated.data import FederatedData
from federated.loop import RoundResult, evaluate, run_federated
from models.networks import build_model

N_TRAIN, N_TEST, CLASSES = 90, 30, 3


@pytest.fixture(autouse=True)
def toy_dataset(monkeypatch):
    rng = np.random.default_rng(0)
    images = {
        True: rng.integers(0, 256, size=(N_TRAIN, 32, 32, 3), dtype=np.uint8),
        False: rng.integers(0, 256, size=(N_TEST, 32, 32, 3), dtype=np.uint8),
    }
    labels = {True: np.arange(N_TRAIN) % CLASSES, False: np.arange(N_TEST) % CLASSES}
    monkeypatch.setitem(datasets.LOADERS, "toy", lambda root: labels[True])
    monkeypatch.setitem(tensors.LOADERS, "toy", lambda root, train: (images[train], labels[train]))


@pytest.fixture(autouse=True)
def torch_state(monkeypatch):
    """configure_torch changes process-wide torch state; put it back after each test."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # its check; CPU runs use no cuBLAS
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    rng = torch.get_rng_state()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = benchmark
    torch.set_rng_state(rng)


def toy_config(out_dir, overrides=None, generate=True):
    """A CPU cell on the toy dataset; its partition is generated unless `generate` is off."""
    data = {
        "dataset": {"name": "toy", "num_classes": CLASSES},
        "partition": {"mode": "pure_label", "K": 3, "alpha": 1.0},
        "model": {"arch": "cnn", "in_channels": 3, "num_classes": CLASSES},
        "federated": {
            "algorithm": "fedavg",
            "rounds": 2,
            "local_epochs": 1,
            "batch_size": 16,
            "lr": 0.001,
        },
        "runtime": {"device": "cpu", "out_dir": str(out_dir)},
    }
    for key, value in (overrides or {}).items():
        section, field = key.split(".")
        data[section][field] = value
    config = from_dict(data)
    if generate:
        generate_partition(config)
    return config


def centralized_config(out_dir, overrides=None):
    """The toy dataset trained centrally on CPU: no partition, 3 epochs."""
    data = to_dict(toy_config(out_dir, generate=False))
    data["partition"] = None
    data["federated"].update(algorithm="centralized", rounds=1, local_epochs=3)
    for key, value in (overrides or {}).items():
        section, field = key.split(".")
        data[section][field] = value
    return from_dict(data)


def epoch_numbers(results):
    """Everything an EpochResult reports except its wall time."""
    return [(r.epoch, r.train_loss, r.test_acc, r.test_loss, r.steps) for r in results]


def numbers(results):
    """Everything a RoundResult reports except its wall time."""
    return [
        (r.round, r.test_acc, r.test_loss, r.clients, r.client_sizes, r.client_train_loss)
        for r in results
    ]


def history(config, name="history.csv"):
    path = Path(config.runtime.out_dir) / "runs" / config_hash(config) / name
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- one run ---------------------------------------------------------------------


def test_a_run_is_recorded_end_to_end(tmp_path):
    config = toy_config(tmp_path, {"federated.rounds": 3})
    results = train(config)

    rows = read_jsonl(runs_path(tmp_path))
    assert [row["status"] for row in rows] == ["running", "completed"]
    final = rows[-1]
    assert final["partition_id"] == partition_id(config, seeds_for(config).partition)
    assert final["final_acc"] == results[-1].test_acc
    assert final["best_acc"] == max(r.test_acc for r in results)
    assert final["rta"] == rounds_to_accuracy([r.test_acc for r in results])
    assert final["rta"] is not None
    assert skip_if_exists(config)

    initial, *rounds = history(config)
    assert [int(row["round"]) for row in rounds] == [1, 2, 3]
    for row, result in zip(rounds, results, strict=True):
        assert float(row["global_test_acc"]) == result.test_acc
        assert float(row["global_test_loss"]) == result.test_loss
        assert float(row["train_loss"]) == train_loss(result)
        assert float(row["round_wall_s"]) == result.wall_s

    clients = history(config, "clients.csv")
    expected = [
        (r.round, *client)
        for r in results
        for client in zip(r.clients, r.client_sizes, r.client_train_loss, r.client_steps)
    ]
    assert [
        (int(c["round"]), int(c["client"]), int(c["num_samples"]), float(c["train_loss"]))
        + (int(c["steps"]),)
        for c in clients
    ] == expected


def test_round_zero_is_the_untrained_model(tmp_path):
    """Round 0 evaluates the initialised model; it has no training columns."""
    config = toy_config(tmp_path)
    train(config)
    seeds = seeds_for(config)
    data = tensors.load_tensors(config.dataset)
    model = build_model(config.model, data.input_shape, seeds.init)
    acc, loss = evaluate(model, data.test_x, data.test_y)

    initial = history(config)[0]
    assert initial["round"] == "0"
    assert float(initial["global_test_acc"]) == acc
    assert float(initial["global_test_loss"]) == loss
    assert initial["train_loss"] == "" and initial["round_wall_s"] == ""


def test_round_zero_counts_for_no_metric(tmp_path, monkeypatch):
    """An untrained model better than every round is still not final, best or rta."""
    config = toy_config(tmp_path, {"federated.rounds": 3})
    monkeypatch.setattr(run_training, "evaluate", lambda model, x, y: (0.99, 0.01))
    results = train(config)
    final = read_jsonl(runs_path(tmp_path))[-1]
    assert final["best_acc"] == max(r.test_acc for r in results) < 0.99
    assert final["rta"] == rounds_to_accuracy([r.test_acc for r in results]) >= 1


@pytest.mark.parametrize("mu", [None, 0.1], ids=["fedavg", "fedprox"])
def test_matches_the_loop_wired_by_hand(tmp_path, mu):
    """The runner adds nothing: same numbers as run_federated called directly."""
    overrides = {} if mu is None else {"federated.algorithm": "fedprox", "federated.mu": mu}
    config = toy_config(tmp_path, overrides)
    results = train(config)

    seeds = seeds_for(config)
    data = tensors.load_tensors(config.dataset)
    clients = load_partition(tmp_path, partition_id(config, seeds.partition))
    model = build_model(config.model, data.input_shape, seeds.init)
    by_hand = run_federated(
        model,
        FederatedData(data.train_x, data.train_y, clients),
        data.test_x,
        data.test_y,
        rounds=2,
        local_epochs=1,
        batch_size=16,
        make_optimizer=lambda params: torch.optim.Adam(params, lr=0.001),
        aggregate=fedavg,
        train_seed=seeds.train,
        proximal_mu=mu,
    )
    assert numbers(results) == numbers(by_hand)


def test_reruns_are_identical_and_trials_differ(tmp_path):
    config = toy_config(tmp_path)
    assert numbers(train(config)) == numbers(train(config))
    other = toy_config(tmp_path, {"runtime.trial": 1})
    assert numbers(train(other)) != numbers(train(config))


@pytest.mark.parametrize(
    "overrides, mu",
    [
        ({}, None),
        ({"federated.algorithm": "fedprox", "federated.mu": 0.01}, 0.01),
    ],
    ids=["fedavg", "fedprox"],
)
def test_config_fields_reach_the_loop(tmp_path, monkeypatch, overrides, mu):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return run_federated(*args, **kwargs)

    monkeypatch.setattr(run_training, "run_federated", spy)
    overrides = {
        "federated.rounds": 3,
        "federated.local_epochs": 2,
        "federated.batch_size": 8,
        "federated.client_fraction": 0.5,
        "runtime.trial": 4,
        **overrides,
    }
    config = toy_config(tmp_path, overrides)
    train(config)
    assert seen["rounds"] == 3 and seen["local_epochs"] == 2 and seen["batch_size"] == 8
    assert seen["client_fraction"] == 0.5
    assert seen["train_seed"] == seeds_for(config).train
    assert seen["aggregate"] is fedavg
    assert seen["proximal_mu"] == mu


def test_optimizer_factory():
    params = [torch.nn.Parameter(torch.zeros(2))]
    federated = toy_config(None, generate=False).federated
    make = optimizer_factory(federated)
    first, second = make(params), make(params)
    assert type(first) is torch.optim.Adam and first is not second
    assert first.defaults["lr"] == 0.001

    data = to_dict(toy_config(None, generate=False))
    data["federated"].update(optimizer="sgd", lr=0.05)
    sgd = optimizer_factory(from_dict(data).federated)(params)
    assert type(sgd) is torch.optim.SGD
    assert sgd.defaults["lr"] == 0.05 and sgd.defaults["momentum"] == 0


def test_proximal_mu_is_none_for_fedavg():
    assert proximal_mu(toy_config(None, generate=False).federated) is None
    fedprox = toy_config(None, {"federated.algorithm": "fedprox", "federated.mu": 1.0}, False)
    assert proximal_mu(fedprox.federated) == 1.0


def test_train_loss_is_weighted_by_client_size():
    result = RoundResult(1, 0.5, 1.0, (0, 1), (10, 30), (1.0, 3.0), (1, 2), (), 0.1)
    assert train_loss(result) == (10 * 1.0 + 30 * 3.0) / 40


# --- per-class accuracy ------------------------------------------------------------


def confusions_by_round(config) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for row in history(config, "confusion.csv"):
        cm = matrices.setdefault(int(row["round"]), np.zeros((CLASSES, CLASSES), dtype=int))
        cm[int(row["true_class"]), int(row["predicted_class"])] = int(row["count"])
    return matrices


@pytest.mark.parametrize("make_config", [toy_config, centralized_config])
def test_every_round_records_the_confusion_and_per_class_accuracy(tmp_path, make_config):
    config = make_config(tmp_path)
    train(config)
    accuracy = {int(row["round"]): float(row["global_test_acc"]) for row in history(config)}
    confusions = confusions_by_round(config)
    per_class = {int(row["round"]): row for row in history(config, "per_class.csv")}

    assert sorted(confusions) == sorted(per_class) == sorted(accuracy)  # round 0 included
    for rnd, cm in confusions.items():
        assert cm.sum() == N_TEST
        assert np.trace(cm) / cm.sum() == accuracy[rnd]  # the same predictions evaluate() scored
        recalls = np.diag(cm) / cm.sum(axis=1)
        assert [float(per_class[rnd][f"acc_class_{c}"]) for c in range(CLASSES)] == list(recalls)
        assert float(per_class[rnd]["balanced_acc"]) == pytest.approx(recalls.mean(), abs=1e-15)

    final = read_jsonl(runs_path(tmp_path))[-1]
    trained = [float(per_class[rnd]["balanced_acc"]) for rnd in sorted(per_class) if rnd >= 1]
    assert final["final_balanced_acc"] == trained[-1]
    assert final["best_balanced_acc"] == max(trained)


@pytest.mark.parametrize("make_config", [toy_config, centralized_config])
def test_recording_the_confusion_leaves_training_as_it_was(tmp_path, monkeypatch, make_config):
    """The extra test-set pass changes no state: the run is the same without it."""
    with_pass = train(make_config(tmp_path / "with"))
    monkeypatch.setattr(
        run_training, "confusion_on_test_set", lambda model, tensors, config: np.eye(CLASSES)
    )
    without_pass = train(make_config(tmp_path / "without"))
    to_numbers = epoch_numbers if make_config is centralized_config else numbers
    assert to_numbers(with_pass) == to_numbers(without_pass)


# --- the partition gate ------------------------------------------------------------


def test_missing_partition_stops_before_anything_is_written(tmp_path):
    config = toy_config(tmp_path, generate=False)
    with pytest.raises(PartitionUnavailable, match="is not in .*run_partition"):
        train(config)
    assert not runs_path(tmp_path).exists()


def test_infeasible_partition_is_refused(tmp_path):
    config = toy_config(tmp_path, generate=False)
    pid = partition_id(config, seeds_for(config).partition)
    record_infeasible(tmp_path, pid, {"partition_id": pid, "status": "infeasible_min_size"})
    with pytest.raises(PartitionUnavailable, match="recorded as infeasible_min_size"):
        train(config)
    assert not runs_path(tmp_path).exists()


def test_validation_split_is_refused(tmp_path):
    config = toy_config(tmp_path, {"dataset.val_split": 0.1})
    with pytest.raises(NotImplementedError, match="val_split"):
        train(config)


# --- sweeps ---------------------------------------------------------------------------


def test_sweep_checks_every_partition_before_training(tmp_path):
    configs = [toy_config(tmp_path), toy_config(tmp_path, {"runtime.trial": 1}, generate=False)]
    with pytest.raises(PartitionUnavailable, match="1 of 2 runs"):
        train_all(configs, log=lambda line: None)
    assert not runs_path(tmp_path).exists()


def test_sweep_resumes_loads_data_once_and_force_retrains(tmp_path, monkeypatch):
    loads = []

    def counting_load(*args):
        loads.append(args)
        return tensors.load_tensors(*args)

    monkeypatch.setattr(run_training, "load_tensors", counting_load)
    configs = [toy_config(tmp_path, {"runtime.trial": t}) for t in (0, 1)]
    lines = []

    assert train_all(configs, log=lines.append) == (2, 0)
    assert len(loads) == 1
    assert train_all(configs, log=lines.append) == (0, 2)
    assert len(loads) == 1  # nothing to train, nothing loaded
    assert sum("already completed" in line for line in lines) == 2
    assert train_all(configs, force=True, log=lines.append) == (2, 0)

    statuses = [row["status"] for row in read_jsonl(runs_path(tmp_path))]
    assert statuses == ["running", "completed"] * 4


def test_failed_run_is_recorded_reraised_and_retried(tmp_path, monkeypatch):
    config = toy_config(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_training, "run_federated", boom)
    with pytest.raises(RuntimeError, match="boom"):
        train(config)
    assert read_jsonl(runs_path(tmp_path))[-1]["status"] == "failed"

    monkeypatch.setattr(run_training, "run_federated", run_federated)
    assert train_all([config], log=lambda line: None) == (1, 0)
    assert read_jsonl(runs_path(tmp_path))[-1]["status"] == "completed"


def test_command_line_trains_a_grid(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.yaml"
    dump_config(toy_config(tmp_path), base)
    toy_config(tmp_path, {"runtime.trial": 1})  # generates trial 1's partition
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"base": str(base), "sweep": {"runtime.trial": [0, 1]}}))

    monkeypatch.setattr(sys, "argv", ["run_training", "--grid", str(grid)])
    run_training.main()
    assert capsys.readouterr().out.splitlines()[-1] == "2 runs: 2 trained, 0 already completed"

    monkeypatch.setattr(sys, "argv", ["run_training", "--config", str(base)])
    run_training.main()
    assert capsys.readouterr().out.splitlines()[-1] == "1 runs: 0 trained, 1 already completed"


# --- the centralized baseline -------------------------------------------------------------


def test_a_centralized_run_is_recorded_end_to_end(tmp_path):
    config = centralized_config(tmp_path)
    results = train(config)
    assert not (tmp_path / "partitions").exists()  # no partition read or made

    rows = read_jsonl(runs_path(tmp_path))
    assert [row["status"] for row in rows] == ["running", "completed"]
    final = rows[-1]
    assert final["algorithm"] == "centralized" and final["partition_id"] is None
    assert (final["rounds"], final["local_epochs"]) == (1, 3)
    assert final["final_acc"] == results[-1].test_acc
    assert final["best_acc"] == max(r.test_acc for r in results)
    assert final["rta"] == rounds_to_accuracy([r.test_acc for r in results])
    assert skip_if_exists(config)

    initial, *epochs = history(config)
    assert initial["round"] == "0" and initial["train_loss"] == ""
    assert [int(row["round"]) for row in epochs] == [1, 2, 3]
    for row, result in zip(epochs, results, strict=True):
        assert float(row["global_test_acc"]) == result.test_acc
        assert float(row["global_test_loss"]) == result.test_loss
        assert float(row["train_loss"]) == result.train_loss
        assert float(row["round_wall_s"]) == result.wall_s

    clients = [
        (int(c["round"]), int(c["client"]), int(c["num_samples"]), int(c["steps"]))
        for c in history(config, "clients.csv")
    ]
    assert clients == [(epoch, 0, N_TRAIN, 6) for epoch in (1, 2, 3)]  # ceil(90 / 16) steps


def test_centralized_matches_train_centralized_wired_by_hand(tmp_path):
    """The runner adds nothing: same numbers as train_centralized called directly."""
    config = centralized_config(tmp_path, {"runtime.trial": 2})
    results = train(config)

    seeds = seeds_for(config)
    data = tensors.load_tensors(config.dataset)
    by_hand = train_centralized(
        build_model(config.model, data.input_shape, seeds.init),
        data.train_x,
        data.train_y,
        data.test_x,
        data.test_y,
        epochs=3,
        batch_size=16,
        make_optimizer=lambda params: torch.optim.Adam(params, lr=0.001),
        train_seed=seeds.train,
    )
    assert epoch_numbers(results) == epoch_numbers(by_hand)


def test_centralized_starts_from_the_federated_runs_weights_at_the_same_trial(tmp_path):
    centralized, federated = centralized_config(tmp_path), toy_config(tmp_path, generate=False)
    assert seeds_for(centralized).init == seeds_for(federated).init
    assert config_hash(centralized) != config_hash(federated)


def test_a_centralized_sweep_needs_no_partition_and_logs_epochs(tmp_path):
    configs = [centralized_config(tmp_path, {"runtime.trial": t}) for t in (0, 1)]
    lines = []
    assert train_all(configs, log=lines.append) == (2, 0)
    assert sum("centralized 3 epochs" in line for line in lines) == 2
    assert sum(line.startswith("  epoch   3/3") for line in lines) == 2
    assert train_all(configs, log=lines.append) == (0, 2)


# --- real CIFAR-10 -----------------------------------------------------------------------


def cifar10_cell(out_dir, trial):
    """smoke.yaml (CIFAR-10, K = 10) cut to 1 round of 1 epoch, trial `trial`, in `out_dir`."""
    data = to_dict(load_config("config/smoke.yaml"))
    data["federated"]["rounds"] = 1
    data["runtime"].update(trial=trial, out_dir=str(out_dir))
    return from_dict(data)


def run_cli(*args):
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-m", "experiments.run_training", *map(str, args)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def trained_numbers(config):
    """history.csv without wall times, clients.csv, and the final runs.jsonl row."""
    rounds = [{k: v for k, v in row.items() if k != "round_wall_s"} for row in history(config)]
    rounds.append(history(config, "clients.csv"))
    final = [r for r in read_jsonl(runs_path(config.runtime.out_dir)) if r["trial"] == 1][-1]
    return rounds, final


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cli_on_cifar10_is_bit_exact_whatever_ran_before(tmp_path):
    """Trial 1 trained second in a sweep = trial 1 trained alone in a fresh process."""
    sweep_dir, alone_dir = tmp_path / "sweep", tmp_path / "alone"
    for config in (cifar10_cell(sweep_dir, 0), cifar10_cell(sweep_dir, 1)):
        generate_partition(config)
    alone = cifar10_cell(alone_dir, 1)
    generate_partition(alone)

    base = tmp_path / "base.yaml"
    dump_config(cifar10_cell(sweep_dir, 0), base)
    grid = tmp_path / "grid.yaml"
    grid.write_text(yaml.safe_dump({"base": str(base), "sweep": {"runtime.trial": [0, 1]}}))
    assert run_cli("--grid", grid).splitlines()[-1] == "2 runs: 2 trained, 0 already completed"
    single = tmp_path / "alone.yaml"
    dump_config(alone, single)
    run_cli("--config", single)

    swept_rounds, swept = trained_numbers(cifar10_cell(sweep_dir, 1))
    alone_rounds, alone_row = trained_numbers(alone)
    assert swept_rounds == alone_rounds
    assert float(alone_rounds[1]["global_test_acc"]) > 0.15  # above chance (0.1): it learned
    for row in (swept, alone_row):
        assert row["status"] == "completed" and row["deterministic"] is True
        assert row["gpu_name"] and row["rta"] == 1
    assert swept["partition_id"] == alone_row["partition_id"]
    assert json.dumps(swept["final_acc"]) == json.dumps(alone_row["final_acc"])

    resumed = run_cli("--grid", grid).splitlines()[-1]
    assert resumed == "2 runs: 0 trained, 2 already completed"
