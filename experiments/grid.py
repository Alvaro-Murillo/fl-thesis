"""Grid files -> ExperimentConfigs.

Grid contract:
    base            path to a complete config; every field not swept comes from it
    constant        dotted overrides applied to every cell
    cases           optional list of dotted-override dicts; cells are cases x sweep.
                    For settings that only make sense together (size_cap applies to
                    natural_per_class only), which a cartesian sweep cannot express
    sweep           dotted key -> list of values; cells are the cartesian product.
                    Required: a grid without one is refused, so a stub can never
                    expand to its base config's cell
    partition_only  `true`: the grid's partitions are made and checked, never trained
                    (rq1.yaml, quantity_skew.yaml, the edge-level grid)
    status          `blocked`, with `blocked_on`: the grid's partitions can be made and
                    checked, but it cannot be trained until what it names is settled
    label           optional text the training runner writes into every run's
                    runs.jsonl rows, e.g. that the grid is a post-hoc check; it is not
                    part of any config, so it moves no config_hash

Every caller says what it expands for. `expand(path, for_training=True)` (the
training runner) refuses a partition_only or blocked grid; the partition tools
expand with `for_training=False`.

A cell may give target_label_hd / target_quantity_hd with alpha / beta null:
the parameter is then resolved from the curve named by calibration_id /
quantity_calibration_id, and target 0 means no skew on that axis
(`partitioning.calibration.resolve_targets`). A centralized cell has no
partition, so nothing is resolved for it.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from experiments.config import ExperimentConfig, from_dict, load_config, to_dict
from partitioning.calibration import CALIBRATION_DIR, resolve_targets


class GridBlocked(RuntimeError):
    """The grid may not be trained: it is partition_only, or blocked."""


class GridIncomplete(ValueError):
    """The grid does not specify its cells."""


def _apply(cell: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        section, field = key.split(".")
        cell[section][field] = value


def load_grid(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def check_trainable(path: str | Path, grid: dict) -> None:
    """Raise GridBlocked unless the grid may be trained."""
    if grid.get("partition_only"):
        raise GridBlocked(f"{path} is partition_only: its partitions are never trained")
    if grid.get("status") == "blocked":
        raise GridBlocked(f"{path} has status blocked, on: {grid.get('blocked_on')}")


def expand(
    path: str | Path, *, for_training: bool, calibration_dir: str | Path = CALIBRATION_DIR
) -> list[ExperimentConfig]:
    grid = load_grid(path)
    sweep = grid.get("sweep")
    if not sweep:
        raise GridIncomplete(f"{path} has no sweep, so it does not specify its cells")
    if for_training:
        check_trainable(path, grid)
    base = to_dict(load_config(grid["base"]))
    _apply(base, grid.get("constant") or {})

    configs = []
    for case in grid.get("cases") or [{}]:
        for combination in itertools.product(*sweep.values()):
            cell = {s: None if values is None else dict(values) for s, values in base.items()}
            _apply(cell, case)
            _apply(cell, dict(zip(sweep, combination)))
            if cell["partition"] is not None:
                resolve_targets(cell, calibration_dir)
            configs.append(from_dict(cell))
    return configs
