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
    """The A4 grid's 63 cells must survive the extension with unchanged IDs."""
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


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_blocked_grids_declare_themselves(name):
    """A stub must be refusable by the driver, not silently sweepable."""
    grid = yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))
    assert grid["status"] == "blocked"
    assert grid["blocked_on"]
    assert "sweep" not in grid
    with pytest.raises(GridBlocked):
        expand(f"{GRIDS}/{name}")
