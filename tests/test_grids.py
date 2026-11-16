"""Tests for the grid files and their expansion."""

import sys
from dataclasses import replace

import numpy as np
import pytest
import yaml

from experiments.config import PartitionMode, config_hash, to_dict
from experiments.grid import GridBlocked, GridIncomplete, expand
from experiments.recording import partition_id, read_jsonl
from experiments.seeding import seeds_for

GRIDS = "config/grids"
PARTITION_ONLY_GRIDS = [
    "rq1.yaml",
    "quantity_skew.yaml",
    "rq3_edge.yaml",
    "covtype_rq1.yaml",
    "covtype_quantity_only.yaml",
]


def pid(config):
    return partition_id(config, seeds_for(config).partition)


def write_grid(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_rq1_expands_to_valid_configs():
    configs = expand(f"{GRIDS}/rq1.yaml", for_training=False)
    assert len(configs) == 3 * 10 * 3 * 3
    assert all(c.partition.min_size_policy == 10 for c in configs)
    cases = {(c.partition.mode, c.partition.size_cap) for c in configs}
    assert cases == {
        (PartitionMode.NATURAL_PER_CLASS, "none"),
        (PartitionMode.NATURAL_PER_CLASS, "niid_bench"),
        (PartitionMode.NATURAL_PER_CLIENT, "none"),
    }


def test_rq1_cells_are_distinct_partitions():
    assert len({pid(c) for c in expand(f"{GRIDS}/rq1.yaml", for_training=False)}) == 270


def test_rq1_keeps_every_partition_of_the_original_63_cell_grid(tmp_path):
    """The original 63-cell RQ1 grid must survive the extension with unchanged IDs."""
    original = tmp_path / "original.yaml"
    original.write_text(
        yaml.safe_dump(
            {
                "base": "config/base.yaml",
                "constant": {"partition.min_size_policy": 10},
                "sweep": {
                    "partition.mode": ["natural_per_class"],
                    "partition.alpha": [0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 100.0],
                    "partition.K": [10, 30, 100],
                    "runtime.trial": [0, 1, 2],
                },
            }
        ),
        encoding="utf-8",
    )
    old = {pid(c) for c in expand(original, for_training=False)}
    assert len(old) == 63
    assert old <= {pid(c) for c in expand(f"{GRIDS}/rq1.yaml", for_training=False)}


def test_rq1_sweeps_only_partition_and_trial():
    """RQ1 is partition-only: no training knob may vary across its cells."""
    federated = {
        tuple(sorted(to_dict(c)["federated"].items()))
        for c in expand(f"{GRIDS}/rq1.yaml", for_training=False)
    }
    assert len(federated) == 1


def test_quantity_skew_grid_expands_to_quantity_only_cells():
    from partitioning.calibration import DEFAULT_BETAS

    configs = expand(f"{GRIDS}/quantity_skew.yaml", for_training=False)
    assert len(configs) == 30 * 3 * 3
    assert len({pid(c) for c in configs}) == 270
    assert all(c.partition.mode is PartitionMode.QUANTITY_ONLY for c in configs)
    assert all(c.partition.alpha is None and c.partition.min_size_policy == 0 for c in configs)
    assert all(c.partition.quantity_floor is None for c in configs)
    # Exactly the calibration grid, so K = 30 cells are the curve's partitions.
    assert sorted({c.partition.beta for c in configs}) == list(DEFAULT_BETAS)
    assert {c.partition.K for c in configs} == {10, 30, 100}


LABEL_LEVELS = [0.0, 0.25, 0.45, 0.70]
QUANTITY_LEVELS = [0.0, 0.35, 0.70]
EDGE_LEVEL = 0.80
FIRST_TRIALS = {0, 1, 2}  # trained first
REPORTED_TRIALS = {0, 1, 2, 5, 6}  # 5 and 6 added 2026-11-11; 3 and 4 tuned mu
RQ2_CURVES = {
    (PartitionMode.NATURAL_PER_CLASS, "none"): "147e58d391",
    (PartitionMode.NATURAL_PER_CLASS, "niid_bench"): "146de82d74",
    (PartitionMode.PURE_LABEL, "none"): "90f8f1dd9b",
}
# alpha per label level and arm, and beta per quantity level, as measured on the
# 20-trial curves before the grids were written (4 significant figures).
MEASURED_ALPHAS = {
    (PartitionMode.NATURAL_PER_CLASS, "none"): {0.25: 1.629, 0.45: 0.4185, 0.70: 0.06852},
    (PartitionMode.NATURAL_PER_CLASS, "niid_bench"): {0.25: 2.579, 0.45: 0.6051, 0.70: 0.09733},
    (PartitionMode.PURE_LABEL, "none"): {0.25: 1.577, 0.45: 0.4116, 0.70: 0.06039},
}
MEASURED_BETAS = {0.35: 0.8634, 0.70: 0.07397, EDGE_LEVEL: 0.007623}


def rq2():
    return expand(f"{GRIDS}/rq2.yaml", for_training=False)


def rq3(name="rq3"):
    return expand(f"{GRIDS}/{name}.yaml", for_training=False)


def test_rq2_resolves_each_arm_on_its_own_20_trial_curve():
    from partitioning.calibration import NO_SKEW, alpha_for_target, load_calibration

    configs = rq2()
    assert len(configs) == 3 * 4 * 5 * 2 and len({pid(c) for c in configs}) == 60
    for c in configs:
        p = c.partition
        assert p.K == 30 and p.target_quantity_hd is None and p.beta is None
        assert p.calibration_id == RQ2_CURVES[(p.mode, p.size_cap)]
        assert p.min_size_policy == (0 if p.mode is PartitionMode.PURE_LABEL else 10)
        expected = (
            NO_SKEW
            if p.target_label_hd == 0
            else alpha_for_target(load_calibration(p.calibration_id), p.target_label_hd)
        )
        assert p.alpha == expected
        if p.target_label_hd:
            measured = MEASURED_ALPHAS[(p.mode, p.size_cap)][p.target_label_hd]
            assert p.alpha == pytest.approx(measured, rel=1e-3)
    assert sorted({c.partition.target_label_hd for c in configs}) == LABEL_LEVELS
    assert {c.runtime.trial for c in configs} == REPORTED_TRIALS


def test_rq2_natural_cells_at_target_zero_are_the_rq1_alpha_1000_partitions():
    zero = {
        (c.partition.size_cap, c.runtime.trial): pid(c)
        for c in rq2()
        if c.partition.target_label_hd == 0
        and c.partition.mode is PartitionMode.NATURAL_PER_CLASS
        and c.runtime.trial in FIRST_TRIALS  # RQ1 runs trials 0-2 only
    }
    assert zero == {
        ("none", 0): "268c0acde3",
        ("none", 1): "ef7bf61e06",
        ("none", 2): "d1cbf3debc",
        ("niid_bench", 0): "1f1683dcb2",
        ("niid_bench", 1): "29c3e9f22c",
        ("niid_bench", 2): "547ae83745",
    }
    rq1 = {pid(c) for c in expand(f"{GRIDS}/rq1.yaml", for_training=False)}
    assert set(zero.values()) <= rq1


def test_rq3_is_combined_with_one_alpha_per_label_level_and_one_beta_per_quantity_level():
    configs = rq3()
    assert len(configs) == 4 * 3 * 5 * 2 and len({pid(c) for c in configs}) == 60
    assert all(c.partition.mode is PartitionMode.COMBINED for c in configs)
    assert all(c.partition.min_size_policy == 0 and c.partition.K == 30 for c in configs)
    alphas, betas = {}, {}
    for c in configs:
        p = c.partition
        alphas.setdefault(p.target_label_hd, set()).add(p.alpha)
        betas.setdefault(p.target_quantity_hd, set()).add(p.beta)
    assert all(len(v) == 1 for v in (*alphas.values(), *betas.values()))
    assert sorted(alphas) == LABEL_LEVELS and sorted(betas) == QUANTITY_LEVELS
    for level, (beta,) in betas.items():
        assert beta == (1000.0 if level == 0 else pytest.approx(MEASURED_BETAS[level], rel=1e-3))
    # Same curve, same alpha as the pure arm: a label level means the same thing in RQ2 and RQ3.
    pure = {
        c.partition.target_label_hd: c.partition.alpha
        for c in rq2()
        if c.partition.mode is PartitionMode.PURE_LABEL
    }
    assert {t: a.pop() for t, a in alphas.items()} == pure


def test_the_edge_level_is_only_in_the_partition_only_edge_grid():
    trained, edge = rq3(), rq3("rq3_edge")
    assert EDGE_LEVEL not in {c.partition.target_quantity_hd for c in trained}
    assert len(edge) == 4 * 3 and len({pid(c) for c in edge}) == 12
    assert {c.partition.target_quantity_hd for c in edge} == {EDGE_LEVEL}
    (beta,) = {c.partition.beta for c in edge}
    assert beta == pytest.approx(MEASURED_BETAS[EDGE_LEVEL], rel=1e-3)
    # The same label levels, alphas and trials as the trained grid.
    alpha = lambda configs: {(c.partition.target_label_hd, c.partition.alpha) for c in configs}
    assert alpha(edge) == alpha(trained)
    assert {c.runtime.trial for c in edge} == {0, 1, 2}
    assert not {pid(c) for c in edge} & {pid(c) for c in trained}


TUNED_MU = 0.001  # chosen on mu_tuning.yaml by the rule fixed there
# sha256 (first 16 hex) of the grid's fedavg config hashes, in expansion order, as they
# were before the fedprox case was added: adding it must not move a fedavg cell. The
# first digest is over trials 0-2 and predates trials 5 and 6, which must not move it;
# the second is over all five trials, pinned before any trial 5-6 run.
FEDAVG_HASHES = {"rq2.yaml": "f76969d1ec263096", "rq3.yaml": "f08f2173ce5df40d"}
FEDAVG_HASHES_5_TRIALS = {"rq2.yaml": "da0b61ad92f2f189", "rq3.yaml": "495aa172cb2279e3"}


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_rq2_and_rq3_pair_every_fedavg_cell_with_fedprox_at_the_tuned_mu(name):
    """Both algorithms on every partition; FedProx's mu is the one tuned on other trials."""
    pairs = {}
    for c in expand(f"{GRIDS}/{name}", for_training=False):
        pairs.setdefault((pid(c), c.runtime.trial), set()).add(
            (c.federated.algorithm, c.federated.mu)
        )
    assert len(pairs) == 60
    assert all(v == {("fedavg", 0.0), ("fedprox", TUNED_MU)} for v in pairs.values())


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_the_fedavg_cells_keep_their_config_hashes(name):
    import hashlib

    from experiments.config import config_hash

    fedavg = [
        c
        for c in expand(f"{GRIDS}/{name}", for_training=False)
        if c.federated.algorithm == "fedavg"
    ]
    digest = lambda hashes: hashlib.sha256(",".join(hashes).encode()).hexdigest()[:16]
    first = [config_hash(c) for c in fedavg if c.runtime.trial in FIRST_TRIALS]
    assert len(first) == 36 and digest(first) == FEDAVG_HASHES[name]
    every = [config_hash(c) for c in fedavg]
    assert len(every) == 60 and digest(every) == FEDAVG_HASHES_5_TRIALS[name]


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_rq2_and_rq3_are_open_for_training(name):
    grid = yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))
    assert "status" not in grid and "blocked_on" not in grid
    assert "partition_only" not in grid
    assert len(expand(f"{GRIDS}/{name}", for_training=True)) == 120


def test_the_centralized_grid_is_the_reported_trials_with_no_partition():
    configs = expand(f"{GRIDS}/centralized.yaml", for_training=True)
    assert [c.runtime.trial for c in configs] == sorted(REPORTED_TRIALS)
    assert all(c.partition is None and c.federated.algorithm == "centralized" for c in configs)
    assert len({config_hash(c) for c in configs}) == 5
    # the three runs trained before trials 5 and 6 were added keep their ids
    assert [config_hash(c) for c in configs[:3]] == ["68edceb2", "e997bce9", "55712819"]


# --- what a grid may be expanded for ----------------------------------------


def test_for_training_must_be_stated_by_keyword():
    with pytest.raises(TypeError):
        expand(f"{GRIDS}/rq1.yaml")
    with pytest.raises(TypeError):
        expand(f"{GRIDS}/rq1.yaml", False)


@pytest.mark.parametrize("name", PARTITION_ONLY_GRIDS)
def test_training_refuses_partition_only_grids(name):
    assert yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))["partition_only"] is True
    with pytest.raises(GridBlocked, match=rf"{name} is partition_only"):
        expand(f"{GRIDS}/{name}", for_training=True)
    assert expand(f"{GRIDS}/{name}", for_training=False)


def blocked_grid(tmp_path):
    return write_grid(
        tmp_path / "blocked.yaml",
        "base: config/smoke.yaml\n"
        "status: blocked\n"
        "blocked_on: a training setting\n"
        "constant:\n"
        f"  runtime.out_dir: {tmp_path.as_posix()}\n"
        "sweep:\n"
        "  runtime.trial: [0, 1]\n",
    )


def test_blocked_refuses_training_only(tmp_path):
    grid = blocked_grid(tmp_path)
    with pytest.raises(GridBlocked, match="blocked, on: a training setting"):
        expand(grid, for_training=True)
    assert len(expand(grid, for_training=False)) == 2


def test_the_training_runner_refuses_a_partition_only_grid(monkeypatch):
    import experiments.run_training as run_training

    monkeypatch.setattr(sys, "argv", ["run_training", "--grid", f"{GRIDS}/rq1.yaml"])
    monkeypatch.setattr(run_training, "train_all", lambda *a, **k: pytest.fail("trained"))
    with pytest.raises(GridBlocked, match="partition_only"):
        run_training.main()


def test_run_grid_generates_a_blocked_grids_partitions(tmp_path, monkeypatch):
    import experiments.run_grid as run_grid

    monkeypatch.setattr(run_grid, "load_labels", lambda dataset: np.repeat(np.arange(10), 600))
    monkeypatch.setattr(sys, "argv", ["run_grid", "--grid", str(blocked_grid(tmp_path))])
    run_grid.main()
    assert [r["status"] for r in read_jsonl(tmp_path / "partitions.jsonl")] == ["generated"] * 2


def test_each_caller_states_what_it_expands_for(tmp_path, monkeypatch):
    """Only the training runner expands for training; the partition and analysis tools never do."""
    import analysis.partition_validation as partition_validation
    import analysis.quantity_skew_measurements as quantity_skew_measurements
    import analysis.training_summary as training_summary
    import experiments.quantity_levels as quantity_levels
    import experiments.run_grid as run_grid
    import experiments.run_training as run_training

    class Stop(Exception):
        pass

    seen = {}

    def spy(name):
        def fake(path, **kwargs):
            seen[name] = kwargs
            raise Stop

        return fake

    grid = str(blocked_grid(tmp_path))
    calls = {
        run_training: (lambda: run_training.main(), ["run_training", "--grid", grid]),
        run_grid: (lambda: run_grid.main(), ["run_grid", "--grid", grid]),
        quantity_levels: (lambda: quantity_levels.main(), ["quantity_levels", "--grid", grid]),
        partition_validation: (lambda: partition_validation.load_cells([grid]), None),
        training_summary: (lambda: training_summary.grid_runs([grid]), None),
        quantity_skew_measurements: (lambda: quantity_skew_measurements.grid_rows(), None),
    }
    for module, (call, argv) in calls.items():
        monkeypatch.setattr(module, "expand", spy(module.__name__))
        if argv:
            monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(Stop):
            call()
    assert seen == {module.__name__: {"for_training": module is run_training} for module in calls}


# --- an incomplete grid still fails, though blocked no longer stops it ------


@pytest.mark.parametrize(
    "spec, error, match",
    [
        (  # alpha null with no target to resolve it from
            "constant:\n  partition.alpha: null\nsweep:\n  runtime.trial: [0]\n",
            ValueError,
            "requires alpha > 0, got None",
        ),
        (  # a target with no curve to read it off
            "constant:\n  partition.alpha: null\nsweep:\n  partition.target_label_hd: [0.25]\n",
            ValueError,
            "target_label_hd=0.25 needs calibration_id",
        ),
        (  # a level still a placeholder, on a real curve
            "constant:\n  partition.alpha: null\n  partition.calibration_id: 147e58d391\n"
            'sweep:\n  partition.target_label_hd: ["<mid>"]\n',
            ValueError,
            "target_label_hd must be a number, got '<mid>'",
        ),
        (  # no sweep: a stub
            "constant:\n  partition.alpha: null\n",
            GridIncomplete,
            "has no sweep",
        ),
    ],
)
def test_an_incomplete_blocked_grid_fails_on_expansion(tmp_path, spec, error, match):
    grid = write_grid(
        tmp_path / "incomplete.yaml",
        "base: config/base.yaml\nstatus: blocked\nblocked_on: its levels\n" + spec,
    )
    with pytest.raises(error, match=match):
        expand(grid, for_training=False)


def test_mu_tuning_uses_the_rq3_mid_cell_at_trials_no_reported_grid_trains():
    configs = expand(f"{GRIDS}/mu_tuning.yaml", for_training=True)
    assert len(configs) == 7 * 2 and len({pid(c) for c in configs}) == 2
    (cell,) = [
        c
        for c in rq3()
        if (c.partition.target_label_hd, c.partition.target_quantity_hd) == (0.45, 0.35)
        and c.runtime.trial == 0
        and c.federated.algorithm == "fedavg"
    ]
    assert {(c.partition.alpha, c.partition.beta) for c in configs} == {
        (cell.partition.alpha, cell.partition.beta)
    }
    reported = {c.runtime.trial for c in rq2() + rq3() + rq3("rq3_edge")}
    assert {c.runtime.trial for c in configs} == {3, 4} and not reported & {3, 4}
    settings = sorted({(c.federated.algorithm, c.federated.mu) for c in configs})
    assert settings == [("fedavg", 0.0)] + [
        ("fedprox", mu) for mu in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    ]


def test_the_class_order_grid_is_rq2s_capped_fedavg_cells_with_classes_reversed():
    configs = expand(f"{GRIDS}/rq2_class_order.yaml", for_training=True)
    capped = {
        (c.partition.alpha, c.runtime.trial): c
        for c in rq2()
        if c.partition.size_cap == "niid_bench"
        and c.partition.target_label_hd in (0.25, 0.45, 0.70)
        and c.federated.algorithm == "fedavg"
    }
    assert len(configs) == 15 and {(c.partition.alpha, c.runtime.trial) for c in configs} == set(
        capped
    )
    # the three runs trained at 0.45 before the other levels were added keep their ids
    at_045 = [
        config_hash(c)
        for c in configs
        if c.partition.alpha == 0.6050899750391434 and c.runtime.trial in FIRST_TRIALS
    ]
    assert at_045 == ["e9cb46cd", "3305baed", "52f05aeb"]
    for config in configs:
        reference = capped[(config.partition.alpha, config.runtime.trial)]
        assert config.partition.class_order == "reversed"
        assert config.partition.alpha == reference.partition.alpha  # the same float
        assert config.partition.target_label_hd is None
        assert config.partition.calibration_id is None
        as_reference = replace(
            config,
            partition=replace(
                config.partition,
                class_order="index",
                target_label_hd=reference.partition.target_label_hd,
                calibration_id=reference.partition.calibration_id,
            ),
        )
        assert as_reference == reference
        assert seeds_for(config) == seeds_for(reference)
        assert pid(config) != pid(reference)


# --- Covtype ------------------------------------------------------------------------------

COVTYPE_CURVES = {
    ("natural_per_class", "none"): "c517442ecf",
    ("natural_per_class", "niid_bench"): "ea8e56e2d6",
    ("pure_label", "none"): "a9a59164f5",
}


@pytest.mark.parametrize(
    "name, runs, partitions, open_for_training",
    [
        ("covtype_rq2", 120, 60, True),
        ("covtype_centralized", 5, 0, True),
        ("covtype_mu_tuning", 14, 2, True),
        ("covtype_centralized_steps", 5, 0, False),
    ],
)
def test_the_covtype_grids_train_the_dnn_on_covtype(name, runs, partitions, open_for_training):
    path = f"{GRIDS}/{name}.yaml"
    if open_for_training:
        assert len(expand(path, for_training=True)) == runs
    else:
        with pytest.raises(GridBlocked, match="blocked"):
            expand(path, for_training=True)
    configs = expand(path, for_training=False)
    assert len(configs) == len({config_hash(c) for c in configs}) == runs
    assert len({pid(c) for c in configs if c.partition is not None}) == partitions
    for c in configs:
        assert (c.dataset.name, c.dataset.num_classes) == ("covtype", 7)
        assert (c.model.arch, c.model.in_channels, c.model.num_classes) == ("dnn", 54, 7)


COVTYPE_LEVELS = {0.0, 0.15, 0.35, 0.50}  # the level rule on Covtype's 20-seed ceilings
COVTYPE_FEDAVG_HASHES = "b3918f6082faab77"
COVTYPE_TUNED_MU = 0.001  # chosen on covtype_mu_tuning.yaml by the rule fixed there


def test_covtype_rq2_is_every_arm_at_every_rule_level_on_its_covtype_curve():
    configs = expand(f"{GRIDS}/covtype_rq2.yaml", for_training=False)
    cells = {}
    for c in configs:
        p = c.partition
        assert p.calibration_id == COVTYPE_CURVES[(p.mode.value, p.size_cap)]
        cells.setdefault((p.mode.value, p.size_cap), set()).add(p.target_label_hd)
        assert (p.alpha == 1000.0) == (p.target_label_hd == 0.0)
    assert cells == {arm: COVTYPE_LEVELS for arm in COVTYPE_CURVES}
    assert {c.runtime.trial for c in configs} == {0, 1, 2, 3, 4}


def test_covtype_rq2_pairs_every_fedavg_cell_with_fedprox_at_the_tuned_mu():
    """Both algorithms on every partition; FedProx's mu is the one tuned on trials 5-6."""
    pairs = {}
    for c in expand(f"{GRIDS}/covtype_rq2.yaml", for_training=False):
        pairs.setdefault((pid(c), c.runtime.trial), set()).add(
            (c.federated.algorithm, c.federated.mu)
        )
    assert len(pairs) == 60
    assert all(v == {("fedavg", 0.0), ("fedprox", COVTYPE_TUNED_MU)} for v in pairs.values())


def test_the_covtype_fedavg_cells_keep_their_config_hashes():
    """Pinned before any FedProx case is added: adding one must not move these."""
    import hashlib

    hashes = [
        config_hash(c)
        for c in expand(f"{GRIDS}/covtype_rq2.yaml", for_training=False)
        if c.federated.algorithm == "fedavg"
    ]
    assert len(hashes) == 60
    assert hashlib.sha256(",".join(hashes).encode()).hexdigest()[:16] == COVTYPE_FEDAVG_HASHES


def test_covtype_mu_tuning_is_the_third_level_on_trials_covtype_rq2_does_not_train():
    tuning = expand(f"{GRIDS}/covtype_mu_tuning.yaml", for_training=False)
    reported = expand(f"{GRIDS}/covtype_rq2.yaml", for_training=False)
    assert {c.runtime.trial for c in tuning} == {5, 6}
    assert not {pid(c) for c in tuning} & {pid(c) for c in reported}
    assert {(c.federated.algorithm, c.federated.mu) for c in tuning} == {
        ("fedavg", 0.0),
        *(("fedprox", mu) for mu in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)),
    }
    pure_035 = {
        c.partition.alpha
        for c in reported
        if c.partition.mode.value == "pure_label" and c.partition.target_label_hd == 0.35
    }
    assert {c.partition.target_label_hd for c in tuning} == {0.35}
    assert {c.partition.alpha for c in tuning} == pure_035  # the reported cell's own alpha


def test_the_400_epoch_centralized_grid_is_the_baseline_trained_longer_and_labelled_post_hoc():
    steps = expand(f"{GRIDS}/covtype_centralized_steps.yaml", for_training=False)
    baseline = expand(f"{GRIDS}/covtype_centralized.yaml", for_training=False)
    assert [c.federated.local_epochs for c in steps] == [400] * 5
    assert [replace(c, federated=replace(c.federated, local_epochs=10)) for c in steps] == baseline
    assert not {config_hash(c) for c in steps} & {config_hash(c) for c in baseline}
    label = yaml.safe_load(open(f"{GRIDS}/covtype_centralized_steps.yaml", encoding="utf-8"))[
        "label"
    ]
    assert "post-hoc" in label and "2026-11-08" in label


def test_covtype_rq1_is_rq1s_cases_and_alphas_on_covtype_at_k30_over_20_trials():
    configs = expand(f"{GRIDS}/covtype_rq1.yaml", for_training=False)
    cifar = expand(f"{GRIDS}/rq1.yaml", for_training=False)
    assert len(configs) == len({pid(c) for c in configs}) == 3 * 10 * 20
    assert {(c.dataset.name, c.partition.K) for c in configs} == {("covtype", 30)}
    assert {c.runtime.trial for c in configs} == set(range(20))
    assert {c.partition.alpha for c in configs} == {c.partition.alpha for c in cifar}
    cases = lambda cs: {(c.partition.mode, c.partition.size_cap) for c in cs}  # noqa: E731
    assert cases(configs) == cases(cifar)
    assert all(c.partition.min_size_policy == 10 for c in configs)


def test_covtype_rq1_reuses_covtype_rq2s_level_zero_natural_partitions():
    rq1 = {pid(c) for c in expand(f"{GRIDS}/covtype_rq1.yaml", for_training=False)}
    level0 = {
        pid(c)
        for c in expand(f"{GRIDS}/covtype_rq2.yaml", for_training=False)
        if c.partition.target_label_hd == 0.0
        and c.partition.mode is PartitionMode.NATURAL_PER_CLASS
    }
    assert len(level0) == 10 and level0 <= rq1


def test_covtype_quantity_only_is_the_beta_curve_grid_on_covtype_at_k30_over_20_trials():
    from partitioning.calibration import DEFAULT_BETAS

    configs = expand(f"{GRIDS}/covtype_quantity_only.yaml", for_training=False)
    assert len(configs) == len({pid(c) for c in configs}) == 30 * 20
    assert all(c.partition.mode is PartitionMode.QUANTITY_ONLY for c in configs)
    assert all(c.partition.alpha is None and c.partition.min_size_policy == 0 for c in configs)
    assert all(c.partition.quantity_floor is None for c in configs)
    assert sorted({c.partition.beta for c in configs}) == list(DEFAULT_BETAS)
    assert {(c.dataset.name, c.partition.K) for c in configs} == {("covtype", 30)}
    assert {c.runtime.trial for c in configs} == set(range(20))
