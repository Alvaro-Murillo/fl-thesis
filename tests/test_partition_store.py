"""Tests for partition addressing, the artifact store, and the partitioner.

Covers A4.3 definition-of-done items 2 and 3 without touching a real dataset:
reruns are byte-identical, `trial` changes the partition, training knobs do not.
"""

import numpy as np
import pytest

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import (
    load_partition,
    partition_exists,
    partition_id,
    read_jsonl,
    save_partition,
)
from experiments.seeding import generator, seeds_for
from partitioning.dirichlet import natural_per_class

LABELS = np.tile(np.arange(10), 500)


def config_with(**runtime_or_section):
    data = to_dict(load_config("config/smoke.yaml"))
    for section, changes in runtime_or_section.items():
        data[section].update(changes)
    return from_dict(data)


def partition_for(config):
    seeds = seeds_for(config)
    return natural_per_class(
        LABELS,
        config.partition.K,
        config.partition.alpha,
        generator(seeds.partition),
        min_size=config.partition.min_size_policy,
    )


def test_same_config_gives_identical_indices():
    config = config_with()
    first, second = partition_for(config), partition_for(config)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(a.dtype == np.int32 for a in first)


def test_partition_covers_every_sample_exactly_once():
    combined = np.concatenate(partition_for(config_with()))
    assert np.array_equal(np.sort(combined), np.arange(len(LABELS)))


def test_min_size_policy_is_honoured():
    config = config_with(partition={"alpha": 0.1, "min_size_policy": 50})
    assert min(len(c) for c in partition_for(config)) >= 50


def test_trial_changes_the_partition():
    a, b = config_with(), config_with(runtime={"trial": 1})
    assert partition_id(a, seeds_for(a).partition) != partition_id(b, seeds_for(b).partition)
    assert not all(np.array_equal(x, y) for x, y in zip(partition_for(a), partition_for(b)))


def test_training_knobs_do_not_change_partition_id():
    """Changing rounds must not invalidate a cached partition (DoD item 3)."""
    base = config_with()
    for section, change in (
        ("federated", {"rounds": 99}),
        ("federated", {"algorithm": "fedprox", "mu": 0.01}),
        ("runtime", {"device": "cpu"}),
    ):
        other = config_with(**{section: change})
        assert partition_id(other, seeds_for(other).partition) == partition_id(
            base, seeds_for(base).partition
        )


def test_partition_fields_do_change_partition_id():
    base = config_with()
    for change in ({"K": 20}, {"alpha": 0.9}, {"min_size_policy": 25}):
        other = config_with(partition=change)
        assert partition_id(other, seeds_for(other).partition) != partition_id(
            base, seeds_for(base).partition
        )


def test_smoke_partition_id_is_stable():
    """Recorded IDs are cache keys; a schema change must not move them."""
    config = config_with()
    assert partition_id(config, seeds_for(config).partition) == "30fbc685a3"


@pytest.mark.slow
def test_smoke_partition_bytes_are_stable(tmp_path):
    """Real CIFAR-10: the smoke partition's indices hash as they did before B3."""
    import hashlib

    import experiments.run_partition as runner

    config = config_with(runtime={"out_dir": str(tmp_path)})
    row = runner.generate_partition(config)
    parts = load_partition(tmp_path, row["partition_id"])
    digest = hashlib.sha256(b"".join(np.ascontiguousarray(c).tobytes() for c in parts))
    assert digest.hexdigest() == "8c09716c79498d5ddfaf4471b84500ddd0ba2098edaf9ab76bb944500518023c"


def test_store_roundtrip_and_no_tmp_left(tmp_path):
    config = config_with()
    pid = partition_id(config, seeds_for(config).partition)
    client_indices = partition_for(config)
    row = {"partition_id": pid, "realized_label_hd": 0.5}

    assert not partition_exists(tmp_path, pid)
    save_partition(tmp_path, pid, client_indices, row)
    assert partition_exists(tmp_path, pid)

    loaded = load_partition(tmp_path, pid)
    assert all(np.array_equal(a, b) for a, b in zip(client_indices, loaded))
    assert read_jsonl(tmp_path / "partitions.jsonl") == [row]
    assert list((tmp_path / "partitions").glob("*.tmp")) == []


def test_jsonl_appends_rather_than_overwrites(tmp_path):
    config = config_with()
    pid = partition_id(config, seeds_for(config).partition)
    indices = partition_for(config)
    for i in range(3):
        save_partition(tmp_path, pid, indices, {"partition_id": pid, "n": i})
    assert [row["n"] for row in read_jsonl(tmp_path / "partitions.jsonl")] == [0, 1, 2]


def test_unimplemented_modes_raise():
    from experiments.run_partition import build_partition

    config = config_with(partition={"mode": "combined", "beta": 1.0})
    with pytest.raises(NotImplementedError, match="B5"):
        build_partition(config, generator(0))


def test_redraw_exhaustion_writes_an_infeasible_row_and_resumes(tmp_path, monkeypatch):
    import experiments.run_partition as runner

    config = config_with(
        partition={"alpha": 0.1, "min_size_policy": len(LABELS), "max_redraws": 3},
        runtime={"out_dir": str(tmp_path)},
    )
    row = runner.generate_partition(config, labels=LABELS)
    assert row["status"] == "infeasible_min_size"
    assert row["n_redraws"] == 3 and row["max_redraws"] == 3
    assert "realized_label_hd" not in row
    pid = row["partition_id"]
    assert not partition_exists(tmp_path, pid)
    assert read_jsonl(tmp_path / "partitions.jsonl") == [row]
    # Same budget: recorded, not retried. A bigger budget is a new attempt.
    assert runner.generate_partition(config, labels=LABELS) is None
    bigger = config_with(
        partition={"alpha": 0.1, "min_size_policy": len(LABELS), "max_redraws": 4},
        runtime={"out_dir": str(tmp_path)},
    )
    assert runner.generate_partition(bigger, labels=LABELS)["n_redraws"] == 4


def test_generated_row_records_redraws_and_settings(tmp_path):
    import experiments.run_partition as runner

    config = config_with(runtime={"out_dir": str(tmp_path)})
    row = runner.generate_partition(config, labels=LABELS)
    assert row["status"] == "generated"
    assert row["partition_id"] == "30fbc685a3"
    assert row["n_redraws"] >= 0
    for key in (
        "size_cap",
        "min_size_policy",
        "max_redraws",
        "trial",
        "n_empty_clients",
        "realized_quantity_hd_pairwise",
    ):
        assert key in row
    assert runner.generate_partition(config, labels=LABELS) is None
