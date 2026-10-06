"""Tests for quantity_only: MinSize-Dirichlet sizes, label-blind assignment."""

import numpy as np
import pytest

from experiments.config import config_hash, dump_config, from_dict, load_config, to_dict
from experiments.recording import load_partition, partition_id
from experiments.seeding import derive_seeds, seeds_for
from partitioning.metrics import (
    hellinger,
    max_quantity_hd,
    partition_metrics,
    quantity_hd_from_pairwise,
)
from partitioning.quantity import (
    default_quantity_floor,
    draw_client_sizes,
    min_size_bound,
    quantity_only,
)

C = 10
FLOOR = default_quantity_floor(C)  # 31, FedArtML's 3 * 10 + 1
BIG = np.repeat(np.arange(C), 5000)  # N = 50,000, balanced like CIFAR-10
N = len(BIG)


def split(labels, K, beta, seed, floor=FLOOR, stats=None):
    return quantity_only(labels, K, beta, floor, np.random.default_rng(seed), stats=stats)


def drawn_shares(K, beta, seed):
    """The normalised Dirichlet draw quantity_only makes for this seed, before the floor."""
    rng = np.random.default_rng(seed)
    rng.permutation(N)
    p = rng.dirichlet(np.repeat(beta, K))
    return p / p.sum()


# --- assignment -------------------------------------------------------------


@pytest.mark.parametrize("K", [10, 30, 100])
@pytest.mark.parametrize("beta", [0.001, 0.1, 1.0, 1000.0])
def test_every_sample_assigned_exactly_once(K, beta):
    parts = split(BIG, K, beta, seed=3)
    assert len(parts) == K
    assert all(p.dtype == np.int32 for p in parts)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(N))


def test_sizes_are_integers_summing_to_N():
    for beta in (0.001, 0.5, 1000.0):
        sizes = draw_client_sizes(N, 30, beta, FLOOR, np.random.default_rng(0))
        assert sizes.dtype.kind == "i" and len(sizes) == 30
        assert sizes.sum() == N and sizes.min() > 0


def test_deterministic_and_global_rng_untouched():
    np.random.seed(123)
    before = np.random.get_state()
    a, b = split(BIG, 30, 0.3, seed=5), split(BIG, 30, 0.3, seed=5)
    after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_different_trial_gives_a_different_partition():
    a = split(BIG, 30, 0.3, derive_seeds(0).partition)
    b = split(BIG, 30, 0.3, derive_seeds(1).partition)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b))


def test_assignment_is_label_blind():
    """Same seed, same N and class count -> identical clients, whatever the labels say."""
    reference = split(BIG, 30, 0.1, seed=11)
    relabelled = np.random.default_rng(0).permutation(C)[BIG]
    shuffled = np.random.default_rng(1).permutation(BIG)
    for labels in (relabelled, shuffled):
        assert all(np.array_equal(x, y) for x, y in zip(reference, split(labels, 30, 0.1, 11)))


# --- sizes and skew ---------------------------------------------------------


def test_large_beta_is_near_iid():
    """beta = 1000: quantity-HD ~ 0, label-HD within noise of an equal-size random split."""
    K, seeds = 30, range(10)
    ours, equal = [], []
    for seed in seeds:
        m = partition_metrics(BIG, split(BIG, K, 1000.0, seed), C)
        assert m["realized_quantity_hd"] < 0.02
        ours.append(m["realized_label_hd"])
        order = np.random.default_rng(10_000 + seed).permutation(N)
        equal.append(partition_metrics(BIG, np.array_split(order, K), C)["realized_label_hd"])
    se = np.sqrt(np.var(ours, ddof=1) / len(ours) + np.var(equal, ddof=1) / len(equal))
    assert abs(np.mean(ours) - np.mean(equal)) <= 4 * se


def test_mean_quantity_hd_decreases_in_beta():
    betas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    means = [
        np.mean(
            [
                partition_metrics(BIG, split(BIG, 30, b, s), C)["realized_quantity_hd"]
                for s in range(5)
            ]
        )
        for b in betas
    ]
    assert all(np.diff(means) < 0), means


@pytest.mark.parametrize("K", [10, 30, 100])
def test_floor_count_bound_and_ceiling(K):
    """n_floored_clients matches the draw; min size >= derived bound; HD <= ceiling."""
    below_nominal = False
    for beta in (0.001, 0.03, 0.3, 1.0, 10.0):
        for seed in range(8):
            stats = {}
            parts = split(BIG, K, beta, seed, stats=stats)
            m_floored = int((drawn_shares(K, beta, seed) < FLOOR / N).sum())
            assert stats["n_floored_clients"] == m_floored

            metrics = partition_metrics(BIG, parts, C)
            smallest = metrics["min_client_size"]
            assert smallest >= min_size_bound(N, FLOOR, m_floored)
            assert smallest >= min_size_bound(N, FLOOR, K - 1)
            below_nominal |= smallest < FLOOR

            assert metrics["realized_quantity_hd"] <= max_quantity_hd(N, K, smallest) + 1e-12
            assert metrics["realized_quantity_hd_pairwise"] <= np.sqrt(1 / K) + 1e-12
            assert quantity_hd_from_pairwise(
                metrics["realized_quantity_hd_pairwise"], K
            ) == pytest.approx(metrics["realized_quantity_hd"], abs=1e-9)
    # The floor is approximate by construction (FedArtML renormalises after it).
    assert below_nominal


def test_max_quantity_hd_is_the_extreme_vertex():
    K, floor = 30, 31
    sizes = np.array([floor] * (K - 1) + [N - (K - 1) * floor])
    direct = hellinger(sizes / N, np.full(K, 1 / K))
    assert max_quantity_hd(N, K, floor) == pytest.approx(direct, abs=1e-12)
    assert max_quantity_hd(N, K, floor) == pytest.approx(0.829, abs=5e-4)
    assert max_quantity_hd(N, K, 30) > max_quantity_hd(N, K, 31)


def test_min_size_bound_values():
    assert min_size_bound(50_000, 31, 0) == 31
    assert min_size_bound(50_000, 31, 1) == 30
    assert min_size_bound(50_000, 31, 29) == 30
    assert min_size_bound(50_000, 31, 99) == 29


def test_floor_is_derived_from_num_classes():
    assert default_quantity_floor(7) == 22
    assert default_quantity_floor(10) == 31


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"floor": 0}, "floor"),
        ({"floor": 2000}, "K \\* floor"),
        ({"beta": 0.0}, "beta"),
    ],
)
def test_draw_client_sizes_rejects_bad_arguments(kwargs, match):
    args = {"N": N, "K": 30, "beta": 1.0, "floor": FLOOR, "rng": np.random.default_rng(0)}
    with pytest.raises(ValueError, match=match):
        draw_client_sizes(**{**args, **kwargs})


# --- exact replication of FedArtML on a shared RNG stream ---------------------


@pytest.mark.parametrize(
    "num_classes, K, beta",
    [
        (10, 5, 0.01),
        (10, 5, 1.0),
        (10, 30, 0.01),
        (10, 30, 0.3),
        (10, 30, 1.0),
        (10, 30, 10.0),
        (10, 30, 1000.0),
        (7, 20, 0.1),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_replicates_fedartml_minsize_dirichlet_exactly(
    shimmed_fedartml, num_classes, K, beta, seed
):
    """Byte-identical client index sets. FedArtML derives its floor from the labels
    (3 * #classes + 1); ours comes from default_quantity_floor(num_classes)."""
    labels = np.repeat(np.arange(num_classes), 300)
    stats = {}
    ours = quantity_only(
        labels,
        K,
        beta,
        default_quantity_floor(num_classes),
        np.random.default_rng(seed),
        stats=stats,
    )
    splitter = shimmed_fedartml(np.random.default_rng(seed))
    _, _, theirs, _ = splitter.dirichlet_method_quant_skew(
        labels, K, alpha=beta, random_state=0, method="minsize-dirichlet"
    )
    assert len(theirs) == K
    assert all(
        np.array_equal(a, np.sort(np.asarray(b, dtype=np.int64))) for a, b in zip(ours, theirs)
    )
    if beta <= 0.01:
        assert stats["n_floored_clients"] > 0  # the floor branch was exercised


# --- config and runner --------------------------------------------------------


def quantity_config(**sections):
    data = to_dict(load_config("config/smoke_quantity_only.yaml"))
    for section, changes in sections.items():
        data[section].update(changes)
    return from_dict(data)


def test_quantity_only_requires_beta_and_rejects_alpha():
    with pytest.raises(ValueError, match="requires beta"):
        quantity_config(partition={"beta": None})
    with pytest.raises(ValueError, match="does not use alpha"):
        quantity_config(partition={"alpha": 0.5})


def test_quantity_only_rejects_redraw_settings():
    with pytest.raises(ValueError, match="min_size_policy"):
        quantity_config(partition={"min_size_policy": 10})
    with pytest.raises(ValueError, match="max_redraws"):
        quantity_config(partition={"max_redraws": 5})


def test_quantity_floor_validation():
    with pytest.raises(ValueError, match=">= 1"):
        quantity_config(partition={"quantity_floor": 0})
    data = to_dict(load_config("config/smoke.yaml"))
    data["partition"]["quantity_floor"] = 31
    with pytest.raises(ValueError, match="quantity_only and combined only"):
        from_dict(data)


def test_quantity_floor_roundtrip_and_identity(tmp_path):
    default, explicit = quantity_config(), quantity_config(partition={"quantity_floor": 40})
    out = tmp_path / "dumped.yaml"
    dump_config(explicit, out)
    assert load_config(out) == explicit and load_config(out).partition.quantity_floor == 40

    pid = lambda c: partition_id(c, seeds_for(c).partition)
    omitted = to_dict(default)
    del omitted["partition"]["quantity_floor"]
    assert config_hash(from_dict(omitted)) == config_hash(default)
    assert pid(from_dict(omitted)) == pid(default)
    assert config_hash(explicit) != config_hash(default)
    assert pid(explicit) != pid(default)


def test_runner_writes_quantity_rows(tmp_path):
    import experiments.run_partition as runner

    config = quantity_config(runtime={"out_dir": str(tmp_path)})
    row = runner.generate_partition(config, labels=BIG)
    assert row["status"] == "generated" and row["mode"] == "quantity_only"
    assert row["quantity_floor"] == 31 and row["n_redraws"] == 0
    assert row["alpha"] is None and row["beta"] == 0.5
    assert isinstance(row["n_floored_clients"], int)
    assert row["min_client_size"] >= min_size_bound(N, 31, row["n_floored_clients"])
    assert row["label_drift"] is None and row["max_label_drift"] is None  # label-blind

    parts = load_partition(tmp_path, row["partition_id"])
    rng = np.random.default_rng(seeds_for(config).partition)
    expected = quantity_only(BIG, config.partition.K, 0.5, 31, rng)
    assert all(np.array_equal(a, b) for a, b in zip(parts, expected))
    assert "floored=" in runner.describe(row)
    assert runner.generate_partition(config, labels=BIG) is None


def test_runner_resolves_floor_from_num_classes(tmp_path):
    import experiments.run_partition as runner

    labels = np.repeat(np.arange(7), 400)
    config = quantity_config(
        dataset={"num_classes": 7}, model={"num_classes": 7}, runtime={"out_dir": str(tmp_path)}
    )
    row = runner.generate_partition(config, labels=labels)
    assert row["quantity_floor"] == 22


def test_natural_rows_carry_null_quantity_fields(tmp_path):
    import experiments.run_partition as runner

    data = to_dict(load_config("config/smoke.yaml"))
    data["runtime"]["out_dir"] = str(tmp_path)
    row = runner.generate_partition(from_dict(data), labels=np.tile(np.arange(10), 500))
    assert row["partition_id"] == "30fbc685a3"
    assert row["quantity_floor"] is None and row["n_floored_clients"] is None
    assert row["label_drift"] is None and row["mean_label_drift"] is None
