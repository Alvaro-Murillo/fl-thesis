"""Grid files -> ExperimentConfigs.

Grid contract:
    base      path to a complete config; every field not swept comes from it
    constant  dotted overrides applied to every cell
    cases     optional list of dotted-override dicts; cells are cases x sweep.
              For settings that only make sense together (size_cap applies to
              natural_per_class only), which a cartesian sweep cannot express
    sweep     dotted key -> list of values; cells are the cartesian product
    status    `blocked` makes the grid unrunnable (see rq2.yaml)
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from experiments.config import ExperimentConfig, from_dict, load_config, to_dict


class GridBlocked(RuntimeError):
    pass


def _apply(cell: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        section, field = key.split(".")
        cell[section][field] = value


def load_grid(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expand(path: str | Path, allow_blocked: bool = False) -> list[ExperimentConfig]:
    grid = load_grid(path)
    if grid.get("status") == "blocked" and not allow_blocked:
        raise GridBlocked(f"{path} is blocked on: {grid.get('blocked_on')}")
    base = to_dict(load_config(grid["base"]))
    _apply(base, grid.get("constant") or {})

    sweep = grid.get("sweep") or {}
    configs = []
    for case in grid.get("cases") or [{}]:
        for combination in itertools.product(*sweep.values()):
            cell = {section: dict(values) for section, values in base.items()}
            _apply(cell, case)
            _apply(cell, dict(zip(sweep, combination)))
            configs.append(from_dict(cell))
    return configs
