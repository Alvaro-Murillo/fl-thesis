"""Tests for the grid files and their expansion."""

import sys

import numpy as np
import pytest
import yaml

from experiments.config import PartitionMode, to_dict
from experiments.grid import GridBlocked, GridIncomplete, expand
from experiments.recording import partition_id, read_jsonl
from experiments.seeding import seeds_for

GRIDS = "config/grids"
PARTITION_ONLY_GRIDS = ["rq1.yaml", "quantity_skew.yaml"]


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


def test_rq1_keeps_every_partition_of_the_a4_grid(tmp_path):
    """The original 63-cell RQ1 grid must survive the extension with unchanged IDs."""
    a4 = tmp_path / "a4.yaml"
    a4.write_text(
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
    old = {pid(c) for c in expand(a4, for_training=False)}
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


CANDIDATE_LABEL_TARGETS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]


def test_rq2_candidates_resolve_each_arm_on_its_own_curve():
    from partitioning.calibration import NO_SKEW, alpha_for_target, load_calibration

    configs = expand(f"{GRIDS}/rq2_candidates.yaml", for_training=False)
    assert len(configs) == 3 * 10 * 3 and len({pid(c) for c in configs}) == 90
    curves = {
        (PartitionMode.NATURAL_PER_CLASS, "none"): "96fb06debf",
        (PartitionMode.NATURAL_PER_CLASS, "niid_bench"): "d85a67a9c5",
        (PartitionMode.PURE_LABEL, "none"): "da8b82c519",
    }
    for c in configs:
        p = c.partition
        assert p.K == 30 and p.target_quantity_hd is None and p.beta is None
        assert p.calibration_id == curves[(p.mode, p.size_cap)]
        assert p.min_size_policy == (0 if p.mode is PartitionMode.PURE_LABEL else 10)
        expected = (
            NO_SKEW
            if p.target_label_hd == 0
            else alpha_for_target(load_calibration(p.calibration_id), p.target_label_hd)
        )
        assert p.alpha == expected
    assert sorted({c.partition.target_label_hd for c in configs}) == CANDIDATE_LABEL_TARGETS
    assert {c.runtime.trial for c in configs} == {0, 1, 2}


def test_rq3_candidates_are_combined_with_shared_draws_per_axis():
    configs = expand(f"{GRIDS}/rq3_candidates.yaml", for_training=False)
    assert len(configs) == 10 * 9 * 3 and len({pid(c) for c in configs}) == 270
    assert all(c.partition.mode is PartitionMode.COMBINED for c in configs)
    assert all(c.partition.min_size_policy == 0 and c.partition.K == 30 for c in configs)
    alphas, betas = {}, {}
    for c in configs:
        p = c.partition
        alphas.setdefault(p.target_label_hd, set()).add(p.alpha)
        betas.setdefault(p.target_quantity_hd, set()).add(p.beta)
    assert all(len(v) == 1 for v in (*alphas.values(), *betas.values()))
    assert sorted(alphas) == CANDIDATE_LABEL_TARGETS
    assert sorted(betas) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # Same curve, same alpha as the pure arm: a label level means the same thing in RQ2 and RQ3.
    pure = {
        c.partition.target_label_hd: c.partition.alpha
        for c in expand(f"{GRIDS}/rq2_candidates.yaml", for_training=False)
        if c.partition.mode is PartitionMode.PURE_LABEL
    }
    assert {t: a.pop() for t, a in alphas.items()} == pure


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
@pytest.mark.parametrize("for_training", [True, False])
def test_stub_grids_are_refused_by_every_tool(name, for_training):
    """A stub has no sweep, so no tool may expand it to its base config's cell."""
    grid = yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))
    assert grid["status"] == "blocked" and grid["blocked_on"]
    assert "sweep" not in grid
    with pytest.raises(GridIncomplete, match="no sweep"):
        expand(f"{GRIDS}/{name}", for_training=for_training)


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
