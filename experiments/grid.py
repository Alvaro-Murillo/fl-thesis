"""Grid files -> ExperimentConfigs.

Grid contract:
    base      path to a complete config; every field not swept comes from it
    constant  dotted overrides applied to every cell
    cases     optional list of dotted-override dicts; cells are cases x sweep.
              For settings that only make sense together (size_cap applies to
              natural_per_class only), which a cartesian sweep cannot express
    sweep     dotted key -> list of values; cells are the cartesian product
    status    `blocked` makes the grid unrunnable (see rq2.yaml)

A cell may give target_label_hd / target_quantity_hd with alpha / beta null:
the parameter is then resolved from the curve named by calibration_id /
quantity_calibration_id, and target 0 means no skew on that axis
(`partitioning.calibration.resolve_targets`).

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from experiments.config import ExperimentConfig, from_dict, load_config, to_dict
from partitioning.calibration import CALIBRATION_DIR, resolve_targets


class GridBlocked(RuntimeError):
    pass


def _apply(cell: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        section, field = key.split(".")
        cell[section][field] = value


def load_grid(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expand(
    path: str | Path, allow_blocked: bool = False, calibration_dir: str | Path = CALIBRATION_DIR
) -> list[ExperimentConfig]:
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
            configs.append(from_dict(resolve_targets(cell, calibration_dir)))
    return configs
