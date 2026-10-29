"""Experiment configuration schema.

Frozen dataclasses are the schema; YAML is the input format. A config
canonicalises to a key-sorted dict, so `config_hash` identifies a grid cell
reproducibly across machines and runs. No Hydra: grids stay diffable in git.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

ALGORITHMS = frozenset({"fedavg", "fedprox"})
ARCHS = frozenset({"cnn", "dnn"})  # mirrors models.networks.ARCHS
OPTIMIZERS = frozenset({"adam", "sgd"})
DEVICES = frozenset({"cuda", "cpu"})
SIZE_CAPS = ("none", "niid_bench")  # mirrors partitioning.dirichlet.SIZE_CAPS
CLASS_ORDERS = ("index", "reversed")  # mirrors partitioning.dirichlet.CLASS_ORDERS
DEFAULT_MAX_REDRAWS = 1000


class PartitionMode(str, Enum):
    NATURAL_PER_CLASS = "natural_per_class"  # FedArtML per-class Dirichlet draw
    NATURAL_PER_CLIENT = "natural_per_client"  # per-client draw, from scratch
    PURE_LABEL = "pure_label"  # label skew, quantity-corrected (equal client sizes)
    QUANTITY_ONLY = "quantity_only"  # MinSize-Dirichlet quantity skew
    COMBINED = "combined"  # two-stage nested: quantity_only sizes, then label skew within them


# Modes that never redraw, and what fixes their client sizes instead.
NO_REDRAW = {
    PartitionMode.QUANTITY_ONLY: "its only size mechanism is quantity_floor",
    PartitionMode.PURE_LABEL: "every client holds N // K or N // K + 1 samples",
    PartitionMode.COMBINED: "its client sizes come from the quantity draw and quantity_floor",
}
# Modes whose client sizes come from partitioning.quantity.draw_client_sizes.
FLOORED_MODES = (PartitionMode.QUANTITY_ONLY, PartitionMode.COMBINED)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    num_classes: int
    root: str = "data"
    val_split: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")


@dataclass(frozen=True)
class PartitionConfig:
    mode: PartitionMode
    K: int
    alpha: float | None = None  # label-Dirichlet concentration
    beta: float | None = None  # quantity-Dirichlet concentration
    target_label_hd: float | None = None
    target_quantity_hd: float | None = None
    min_size_policy: int = 0  # min samples per client; redraw below it, 0 = off
    calibration_id: str | None = None  # calibration/{id}.json: target_label_hd -> alpha
    size_cap: str = "none"  # "niid_bench" = FedArtML's N/K balancing term
    max_redraws: int = DEFAULT_MAX_REDRAWS  # min-size redraw budget; exhausted = infeasible row
    quantity_floor: int | None = None  # size floor in samples; None = 3 * num_classes + 1
    quantity_calibration_id: str | None = None  # calibration/{id}.json: target_quantity_hd -> beta
    class_order: str = "index"  # natural_per_class: order the classes are split in

    def __post_init__(self):
        if self.size_cap not in SIZE_CAPS:
            raise ValueError(f"size_cap must be one of {SIZE_CAPS}, got {self.size_cap!r}")
        if self.size_cap != "none" and self.mode is not PartitionMode.NATURAL_PER_CLASS:
            raise ValueError(f"size_cap applies to natural_per_class only, not {self.mode.value}")
        if self.class_order not in CLASS_ORDERS:
            raise ValueError(f"class_order must be one of {CLASS_ORDERS}, got {self.class_order!r}")
        if self.class_order != "index" and self.mode is not PartitionMode.NATURAL_PER_CLASS:
            raise ValueError(
                f"class_order applies to natural_per_class only, not {self.mode.value}"
            )
        if self.max_redraws < 1:
            raise ValueError(f"max_redraws must be >= 1, got {self.max_redraws}")
        if self.quantity_floor is not None:
            if self.mode not in FLOORED_MODES:
                raise ValueError(
                    f"quantity_floor applies to quantity_only and combined only, "
                    f"not {self.mode.value}"
                )
            if self.quantity_floor < 1:
                raise ValueError(f"quantity_floor must be >= 1, got {self.quantity_floor}")
        if self.quantity_calibration_id is not None and self.mode not in FLOORED_MODES:
            raise ValueError(
                f"quantity_calibration_id applies to quantity_only and combined only, "
                f"not {self.mode.value}"
            )
        if self.mode in NO_REDRAW:
            # No redraw happens in these modes, so no row may claim one did.
            if self.min_size_policy != 0:
                raise ValueError(
                    f"{self.mode.value} never redraws; {NO_REDRAW[self.mode]}. "
                    f"Set min_size_policy to 0, got {self.min_size_policy}"
                )
            if self.max_redraws != DEFAULT_MAX_REDRAWS:
                raise ValueError(
                    f"{self.mode.value} never redraws; leave max_redraws at {DEFAULT_MAX_REDRAWS}"
                )
        if self.K < 2:
            raise ValueError(f"K must be >= 2, got {self.K}")
        needs_alpha = self.mode is not PartitionMode.QUANTITY_ONLY
        needs_beta = self.mode in (PartitionMode.QUANTITY_ONLY, PartitionMode.COMBINED)
        for name, value, needed in (
            ("alpha", self.alpha, needs_alpha),
            ("beta", self.beta, needs_beta),
        ):
            if needed and (value is None or value <= 0):
                raise ValueError(f"mode {self.mode.value} requires {name} > 0, got {value}")
            if not needed and value is not None:
                raise ValueError(f"mode {self.mode.value} does not use {name}")
        for name, value in (
            ("target_label_hd", self.target_label_hd),
            ("target_quantity_hd", self.target_quantity_hd),
        ):
            if value is not None and not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    in_channels: int  # image channels for the cnn, feature count for the dnn
    num_classes: int

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {sorted(ARCHS)}, got {self.arch!r}")


@dataclass(frozen=True)
class FederatedConfig:
    algorithm: str
    rounds: int
    local_epochs: int
    batch_size: int
    lr: float
    mu: float = 0.0  # FedProx proximal term
    optimizer: str = "adam"
    client_fraction: float = 1.0

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
        if self.algorithm == "fedavg" and self.mu != 0.0:
            raise ValueError("mu is a FedProx parameter; fedavg requires mu = 0")
        if self.algorithm == "fedprox" and self.mu <= 0.0:
            raise ValueError("fedprox requires mu > 0")
        if min(self.rounds, self.local_epochs, self.batch_size) < 1:
            raise ValueError("rounds, local_epochs and batch_size must be >= 1")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0.0 < self.client_fraction <= 1.0:
            raise ValueError(f"client_fraction must be in (0, 1], got {self.client_fraction}")


@dataclass(frozen=True)
class RuntimeConfig:
    trial: int = 0
    deterministic: bool = True
    device: str = "cuda"
    out_dir: str = "results"
    partition_seed: int | None = None  # None = derive from trial (experiments.seeding)
    init_seed: int | None = None
    train_seed: int | None = None

    def __post_init__(self):
        if self.trial < 0:
            raise ValueError(f"trial must be >= 0, got {self.trial}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {sorted(DEVICES)}")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    partition: PartitionConfig
    model: ModelConfig
    federated: FederatedConfig
    runtime: RuntimeConfig = RuntimeConfig()

    def __post_init__(self):
        if self.model.num_classes != self.dataset.num_classes:
            raise ValueError(
                f"model.num_classes ({self.model.num_classes}) != "
                f"dataset.num_classes ({self.dataset.num_classes})"
            )


_SECTIONS = {f.name: f.type for f in fields(ExperimentConfig)}
_SECTION_TYPES = {
    "dataset": DatasetConfig,
    "partition": PartitionConfig,
    "model": ModelConfig,
    "federated": FederatedConfig,
    "runtime": RuntimeConfig,
}


def _build(section: str, values: dict[str, Any]):
    cls = _SECTION_TYPES[section]
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{section}': {sorted(unknown)}")
    if section == "partition" and "mode" in values:
        values = {**values, "mode": PartitionMode(values["mode"])}
    return cls(**values)


def from_dict(data: dict[str, Any]) -> ExperimentConfig:
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")
    return ExperimentConfig(**{s: _build(s, data.get(s, {})) for s in _SECTIONS})


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Plain nested dict with enums flattened to their values (YAML/JSON safe)."""
    return asdict(
        config, dict_factory=lambda kv: {k: (v.value if isinstance(v, Enum) else v) for k, v in kv}
    )


# Fields added after the first schema, with their defaults. They enter config_hash only when
# set to something else, so every run recorded before them still resumes.
_ADDED_FIELD_DEFAULTS = {
    "partition": {
        "size_cap": "none",
        "max_redraws": DEFAULT_MAX_REDRAWS,
        "quantity_floor": None,
        "quantity_calibration_id": None,
        "class_order": "index",
    }
}


def config_hash(config: ExperimentConfig) -> str:
    """First 8 hex chars of the sha256 of the canonical (key-sorted) config."""
    data = to_dict(config)
    for section, defaults in _ADDED_FIELD_DEFAULTS.items():
        for key, default in defaults.items():
            if data[section][key] == default:
                del data[section][key]
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh) or {})


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    text = yaml.safe_dump(to_dict(config), sort_keys=True, default_flow_style=False)
    Path(path).write_text(text, encoding="utf-8")
