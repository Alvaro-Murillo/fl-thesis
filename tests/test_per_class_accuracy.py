"""Tests for per-class reruns: bit-exact against the recorded run, or refused.

Like test_run_training, these train the real CNN on CPU on a toy dataset of random
32x32 images (3 classes, 3 clients).
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import datasets, per_class_accuracy, tensors
from experiments.config import config_hash, from_dict, to_dict
from experiments.grid import expand
from experiments.per_class_accuracy import (
    NotBitExact,
    import_results,
    load_per_class,
    output_path,
    per_class_accuracy as recall,
    rerun,
    select,
    sweep,
)
from experiments.recording import append_jsonl, read_jsonl, runs_path
from experiments.run_partition import generate_partition
from experiments.run_training import train

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
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    rng = torch.get_rng_state()
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = benchmark
    torch.set_rng_state(rng)


def toy_config(out_dir, overrides=None):
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
    generate_partition(config)
    return config


def trained(out_dir, overrides=None):
    config = toy_config(out_dir, overrides)
    train(config)
    return config


def history_path(config):
    return Path(config.runtime.out_dir) / "runs" / config_hash(config) / "history.csv"


@pytest.mark.parametrize("mu", [None, 0.1], ids=["fedavg", "fedprox"])
def test_a_rerun_reproduces_its_recorded_run(tmp_path, mu):
    overrides = {} if mu is None else {"federated.algorithm": "fedprox", "federated.mu": mu}
    config = trained(tmp_path, overrides)
    rows_before = read_jsonl(runs_path(tmp_path))

    assert sweep([config], log=lambda _: None) == (1, 0)
    payload = load_per_class(tmp_path, config_hash(config))
    assert payload["bit_exact"] and payload["source"] == "rerun"
    with history_path(config).open(newline="", encoding="utf-8") as fh:
        recorded = [float(row["global_test_acc"]) for row in csv.DictReader(fh)]
    assert payload["accs"] == recorded  # round 0 first
    assert len(payload["confusions"]) == config.federated.rounds + 1
    final = np.array(payload["confusions"][-1])
    assert final.sum() == N_TEST and final.shape == (CLASSES, CLASSES)
    assert recall(final).mean() == pytest.approx(payload["accs"][-1])  # 10 images per class
    assert read_jsonl(runs_path(tmp_path)) == rows_before  # nothing recorded
    assert sweep([config], log=lambda _: None) == (0, 1)  # resumes: the output exists


def tamper_history(config):
    path = history_path(config)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[1]["global_test_acc"] = str(float(rows[1]["global_test_acc"]) + 1 / N_TEST)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tamper_final_acc(config):
    path = runs_path(config.runtime.out_dir)
    row = [r for r in read_jsonl(path) if r["run_id"] == config_hash(config)][-1]
    append_jsonl(path, {**row, "final_acc": row["final_acc"] + 0.01})


@pytest.mark.parametrize("tamper", [tamper_history, tamper_final_acc], ids=["history", "final_acc"])
def test_a_rerun_that_does_not_reproduce_is_saved_stops_the_sweep_and_is_refused(tmp_path, tamper):
    config, other = trained(tmp_path), trained(tmp_path, {"runtime.trial": 1})
    tamper(config)

    with pytest.raises(NotBitExact, match="did not reproduce"):
        sweep([config, other], log=lambda _: None)
    saved = json.loads(output_path(tmp_path, config_hash(config)).read_text(encoding="utf-8"))
    assert saved["bit_exact"] is False
    assert not output_path(tmp_path, config_hash(other)).exists()  # the sweep stopped
    with pytest.raises(NotBitExact, match="not bit-exact"):
        load_per_class(tmp_path, config_hash(config))
    with pytest.raises(NotBitExact, match="not continuing"):
        sweep([other], log=lambda _: None)


def test_a_centralized_config_is_refused(tmp_path):
    data = to_dict(toy_config(tmp_path))
    data["partition"] = None
    data["federated"].update(algorithm="centralized", rounds=1, local_epochs=3)
    with pytest.raises(ValueError, match="centralized"):
        rerun(from_dict(data))


def test_an_untrained_config_is_refused(tmp_path):
    trained(tmp_path)
    untrained = toy_config(tmp_path, {"runtime.trial": 1})
    with pytest.raises(RuntimeError, match="train them first"):
        sweep([untrained], log=lambda _: None)


def test_a_sweep_refuses_while_a_run_is_running(tmp_path):
    config = trained(tmp_path)
    row = read_jsonl(runs_path(tmp_path))[-1]
    append_jsonl(runs_path(tmp_path), {**row, "run_id": "0000beef", "status": "running"})
    with pytest.raises(RuntimeError, match="running"):
        sweep([config], log=lambda _: None)


def test_a_sweep_stops_if_runs_jsonl_changes(tmp_path, monkeypatch):
    first, second = trained(tmp_path), trained(tmp_path, {"runtime.trial": 1})
    real = per_class_accuracy.rerun

    def rerun_then_train(config, *args, **kwargs):
        out = real(config, *args, **kwargs)
        row = read_jsonl(runs_path(tmp_path))[-1]
        append_jsonl(runs_path(tmp_path), {**row, "run_id": "0000beef"})  # something trained
        return out

    monkeypatch.setattr(per_class_accuracy, "rerun", rerun_then_train)
    with pytest.raises(RuntimeError, match="changed during the sweep"):
        sweep([first, second], log=lambda _: None)
    assert output_path(tmp_path, config_hash(first)).exists()
    assert not output_path(tmp_path, config_hash(second)).exists()


def earlier_script_payload(config, out):
    """A per-class file as the standalone scripts of 2026-10-29 wrote it."""
    return {
        "run_id": config_hash(config),
        "arm": "capped",
        "level": 0.45,
        "partition_id": out["partition_id"],
        "trial": out["trial"],
        "seeds": out["seeds"],
        "accs": out["accs"],
        "confusions": out["confusions"],
        "internal_mismatch": [],
        "wall_clock_s": out["wall_clock_s"],
        "started_at": out["started_at"],
        "finished_at": out["finished_at"],
        "git_commit": "5985861",
        "git_dirty": False,
        "torch_version": "x",
        "cuda_version": None,
        "gpu_name": None,
    }


def test_import_checks_every_file_and_keeps_its_provenance(tmp_path):
    config = trained(tmp_path / "results")
    source = tmp_path / "earlier"
    source.mkdir()
    payload = earlier_script_payload(config, rerun(config))
    (source / "L045_capped_t0.json").write_text(json.dumps(payload), encoding="utf-8")

    assert import_results(source, tmp_path / "results", log=lambda _: None) == 1
    imported = load_per_class(tmp_path / "results", config_hash(config))
    assert imported["git_commit"] == "5985861" and imported["source"].startswith("imported")
    assert imported["accs"] == payload["accs"]
    assert import_results(source, tmp_path / "results", log=lambda _: None) == 0  # not overwritten


@pytest.mark.parametrize(
    "spoil",
    [
        lambda p: p["accs"].__setitem__(1, p["accs"][1] + 1 / N_TEST),
        lambda p: p.__setitem__("partition_id", "0123456789"),
        lambda p: p["internal_mismatch"].append("final model != last on_round model"),
        lambda p: p["confusions"][-1][0].__setitem__(0, p["confusions"][-1][0][0] + 1),
    ],
    ids=["a_round", "partition", "final_model", "confusion"],
)
def test_import_refuses_a_folder_with_one_file_that_does_not_reproduce(tmp_path, spoil):
    good, bad = trained(tmp_path / "results"), trained(tmp_path / "results", {"runtime.trial": 1})
    source = tmp_path / "earlier"
    source.mkdir()
    for name, config in (("a.json", good), ("b.json", bad)):
        payload = earlier_script_payload(config, rerun(config))
        if config is bad:
            spoil(payload)
        (source / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NotBitExact, match="nothing imported"):
        import_results(source, tmp_path / "results", log=lambda _: None)
    assert not output_path(tmp_path / "results", config_hash(good)).exists()


def test_select_filters_a_grid_by_dotted_keys():
    configs = expand("config/grids/rq2.yaml", for_training=False)
    chosen = select(
        configs,
        [
            "federated.algorithm=fedavg",
            "partition.mode=natural_per_class",
            "partition.target_label_hd=0.70",
        ],
    )
    assert len(chosen) == 10
    assert {c.partition.size_cap for c in chosen} == {"niid_bench", "none"}
    assert {c.runtime.trial for c in chosen} == {0, 1, 2, 5, 6}
    with pytest.raises(ValueError, match="no config matches"):
        select(configs, ["federated.algorithm=sgd"])
    with pytest.raises(ValueError, match="section.field=value"):
        select(configs, ["algorithm=fedavg"])
