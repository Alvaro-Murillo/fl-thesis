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
    runs_path,
    skip_if_exists,
)
from experiments.run_partition import generate_partition
from experiments.run_training import (
    PartitionUnavailable,
    mean_client_train_loss,
    optimizer_factory,
    proximal_mu,
    train,
    train_all,
)
from experiments.seeding import seeds_for
from federated.aggregation import fedavg
from federated.data import FederatedData
from federated.loop import RoundResult, run_federated
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


def numbers(results):
    """Everything a RoundResult reports except its wall time."""
    return [
        (r.round, r.test_acc, r.test_loss, r.clients, r.client_sizes, r.client_train_loss)
        for r in results
    ]


def history(config):
    path = Path(config.runtime.out_dir) / "runs" / config_hash(config) / "history.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- one run ---------------------------------------------------------------------


def test_a_run_is_recorded_end_to_end(tmp_path):
    config = toy_config(tmp_path)
    results = train(config)

    rows = read_jsonl(runs_path(tmp_path))
    assert [row["status"] for row in rows] == ["running", "completed"]
    final = rows[-1]
    assert final["partition_id"] == partition_id(config, seeds_for(config).partition)
    assert final["final_acc"] == results[-1].test_acc
    assert final["best_acc"] == max(r.test_acc for r in results)
    assert final["rta"] is None
    assert skip_if_exists(config)

    rounds = history(config)
    assert [int(row["round"]) for row in rounds] == [1, 2]
    for row, result in zip(rounds, results):
        assert float(row["global_test_acc"]) == result.test_acc
        assert float(row["global_test_loss"]) == result.test_loss
        assert float(row["mean_client_train_loss"]) == mean_client_train_loss(result)
        assert float(row["round_wall_s"]) == result.wall_s


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


def test_mean_client_train_loss_is_unweighted():
    result = RoundResult(1, 0.5, 1.0, (0, 1), (10, 30), (1.0, 3.0), (1, 2), (), 0.1)
    assert mean_client_train_loss(result) == 2.0


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
    """history.csv without wall times, and the final runs.jsonl row."""
    rounds = [{k: v for k, v in row.items() if k != "round_wall_s"} for row in history(config)]
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
    assert float(alone_rounds[0]["global_test_acc"]) > 0.15  # above chance (0.1): it learned
    for row in (swept, alone_row):
        assert row["status"] == "completed" and row["deterministic"] is True
        assert row["gpu_name"] and row["rta"] is None
    assert swept["partition_id"] == alone_row["partition_id"]
    assert json.dumps(swept["final_acc"]) == json.dumps(alone_row["final_acc"])

    resumed = run_cli("--grid", grid).splitlines()[-1]
    assert resumed == "2 runs: 0 trained, 2 already completed"
