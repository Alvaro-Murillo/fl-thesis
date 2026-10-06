"""Tests for the grid files and their expansion."""

import pytest
import yaml

from experiments.config import PartitionMode, to_dict
from experiments.grid import GridBlocked, expand
from experiments.recording import partition_id
from experiments.seeding import seeds_for

GRIDS = "config/grids"


def pid(config):
    return partition_id(config, seeds_for(config).partition)


def test_rq1_expands_to_valid_configs():
    configs = expand(f"{GRIDS}/rq1.yaml")
    assert len(configs) == 3 * 10 * 3 * 3
    assert all(c.partition.min_size_policy == 10 for c in configs)
    cases = {(c.partition.mode, c.partition.size_cap) for c in configs}
    assert cases == {
        (PartitionMode.NATURAL_PER_CLASS, "none"),
        (PartitionMode.NATURAL_PER_CLASS, "niid_bench"),
        (PartitionMode.NATURAL_PER_CLIENT, "none"),
    }


def test_rq1_cells_are_distinct_partitions():
    assert len({pid(c) for c in expand(f"{GRIDS}/rq1.yaml")}) == 270


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
    old = {pid(c) for c in expand(a4)}
    assert len(old) == 63
    assert old <= {pid(c) for c in expand(f"{GRIDS}/rq1.yaml")}


def test_rq1_sweeps_only_partition_and_trial():
    """RQ1 is partition-only: no training knob may vary across its cells."""
    federated = {
        tuple(sorted(to_dict(c)["federated"].items())) for c in expand(f"{GRIDS}/rq1.yaml")
    }
    assert len(federated) == 1


def test_quantity_skew_grid_expands_to_quantity_only_cells():
    from partitioning.calibration import DEFAULT_BETAS

    configs = expand(f"{GRIDS}/quantity_skew.yaml")
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

    configs = expand(f"{GRIDS}/rq2_candidates.yaml")
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
    configs = expand(f"{GRIDS}/rq3_candidates.yaml")
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
        for c in expand(f"{GRIDS}/rq2_candidates.yaml")
        if c.partition.mode is PartitionMode.PURE_LABEL
    }
    assert {t: a.pop() for t, a in alphas.items()} == pure


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_blocked_grids_declare_themselves(name):
    """A stub must be refusable by the driver, not silently sweepable."""
    grid = yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))
    assert grid["status"] == "blocked"
    assert grid["blocked_on"]
    assert "sweep" not in grid
    with pytest.raises(GridBlocked):
        expand(f"{GRIDS}/{name}")
