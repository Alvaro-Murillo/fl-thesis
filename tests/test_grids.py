"""Tests for the grid files.

Expansion lives here rather than in `experiments/` on purpose: the sweep driver
is not written yet, and this keeps the grid data verifiable without inventing
half of it. When the driver lands, it takes over these five lines.
"""

import itertools

import pytest
import yaml

from experiments.config import from_dict, load_config, to_dict
from experiments.recording import partition_id
from experiments.seeding import seeds_for

GRIDS = "config/grids"


def expand(grid_path):
    grid = yaml.safe_load(open(grid_path, encoding="utf-8"))
    base = to_dict(load_config(grid["base"]))
    for key, value in (grid.get("constant") or {}).items():
        section, field = key.split(".")
        base[section][field] = value

    sweep = grid["sweep"]
    for combination in itertools.product(*sweep.values()):
        cell = {section: dict(values) for section, values in base.items()}
        for key, value in zip(sweep, combination):
            section, field = key.split(".")
            cell[section][field] = value
        yield from_dict(cell)


def test_rq1_expands_to_valid_configs():
    configs = list(expand(f"{GRIDS}/rq1.yaml"))
    assert len(configs) == 7 * 3 * 3
    assert all(c.partition.min_size_policy == 10 for c in configs)


def test_rq1_cells_are_distinct_partitions():
    ids = {partition_id(c, seeds_for(c).partition) for c in expand(f"{GRIDS}/rq1.yaml")}
    assert len(ids) == 63


def test_rq1_sweeps_only_partition_and_trial():
    """RQ1 is partition-only: no training knob may vary across its cells."""
    federated = {tuple(sorted(to_dict(c)["federated"].items()))
                 for c in expand(f"{GRIDS}/rq1.yaml")}
    assert len(federated) == 1


@pytest.mark.parametrize("name", ["rq2.yaml", "rq3.yaml"])
def test_blocked_grids_declare_themselves(name):
    """A stub must be refusable by the driver, not silently sweepable."""
    grid = yaml.safe_load(open(f"{GRIDS}/{name}", encoding="utf-8"))
    assert grid["status"] == "blocked"
    assert grid["blocked_on"]
    assert "sweep" not in grid
