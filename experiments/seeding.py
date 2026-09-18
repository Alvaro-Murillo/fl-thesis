"""Seed derivation and torch determinism (work-plan task A4.2).

One `trial` index derives three independent streams, so it is always known
whether spread across seeds came from partitioning or from training:

    partition_seed -> Dirichlet draws, sample assignment
    init_seed      -> model weight initialisation
    train_seed     -> batch shuffling, augmentation order

Partitioners take an explicit `np.random.Generator`; the global `np.random`
state is never touched. Seeds fit in uint32 so they can also be handed to
FedArtML's `random_state`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Changing this invalidates every seed recorded so far.
MASTER_SEED = 20260912

_UINT32 = 2 ** 32


@dataclass(frozen=True)
class Seeds:
    partition: int
    init: int
    train: int


def derive_seeds(trial: int, partition: int | None = None,
                 init: int | None = None, train: int | None = None) -> Seeds:
    """Three independent streams for `trial`; explicit values override."""
    children = np.random.SeedSequence([MASTER_SEED, trial]).spawn(3)
    derived = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    seeds = [override if override is not None else default
             for override, default in zip((partition, init, train), derived)]
    for name, seed in zip(("partition", "init", "train"), seeds):
        if not 0 <= seed < _UINT32:
            raise ValueError(f"{name}_seed must fit in uint32, got {seed}")
    return Seeds(*seeds)


def seeds_for(config) -> Seeds:
    """Seeds for an ExperimentConfig, honouring `runtime` overrides."""
    runtime = config.runtime
    return derive_seeds(runtime.trial, runtime.partition_seed,
                        runtime.init_seed, runtime.train_seed)


def generator(seed: int) -> np.random.Generator:
    """The only sanctioned source of randomness for partitioners."""
    return np.random.default_rng(seed)


def enable_cublas_determinism() -> None:
    """Must run before `import torch`; call it first in every entry script."""
    if "torch" in sys.modules:
        raise RuntimeError(
            "torch is already imported - CUBLAS_WORKSPACE_CONFIG has no effect. "
            "Call enable_cublas_determinism() before importing torch.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def configure_torch(seeds: Seeds, deterministic: bool = True):
    """Seed torch and set the determinism mode. Returns the torch module."""
    import torch

    torch.manual_seed(seeds.init)
    torch.cuda.manual_seed_all(seeds.init)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)
    if deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "deterministic mode needs CUBLAS_WORKSPACE_CONFIG=:4096:8 set before "
            "torch was imported; call enable_cublas_determinism() first.")
    return torch


def torch_generator(seed: int):
    """Generator for DataLoader shuffling, so batch order follows train_seed."""
    import torch

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
