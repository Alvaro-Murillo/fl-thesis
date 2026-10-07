"""Centralized training: the whole training set as one client, one optimizer.

This is the plain, non-federated setting: it checks that a model trains before
it goes into the federated loop, and gives the centralized baseline. It reuses
the loop's `local_train` and `evaluate`, so a centralized epoch takes exactly
the steps a federated client would take on the same data.

Differences from a federated client: one optimizer is built for the whole run,
so Adam's moments carry across epochs, and one batch-order stream seeded with
`train_seed` runs through all epochs. The test set is evaluated after every
epoch.

This module imports torch: entry scripts call
`experiments.seeding.enable_cublas_determinism()` before importing it.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from federated.data import FederatedData
from federated.loop import MakeOptimizer, evaluate, local_train


@dataclass(frozen=True)
class EpochResult:
    epoch: int  # 1-indexed
    train_loss: float  # mean per-sample loss over the epoch, taken during training
    test_acc: float
    test_loss: float
    steps: int
    wall_s: float


def train_centralized(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    make_optimizer: MakeOptimizer,
    train_seed: int,
    on_epoch: Callable[[EpochResult], None] | None = None,
    eval_batch_size: int = 1000,
) -> list[EpochResult]:
    """Train `model` in place on all of (x, y) for `epochs` epochs."""
    devices = {p.device for p in model.parameters()} | {x.device, test_x.device}
    if len(devices) != 1:
        raise ValueError(f"model, training data and test data must share a device, got {devices}")
    test_y = test_y.long()
    data = FederatedData(x, y, [np.arange(len(y))])
    optimizer = make_optimizer(model.parameters())
    generator = torch.Generator()
    generator.manual_seed(train_seed)
    results = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, steps = local_train(
            model,
            data,
            0,
            epochs=1,
            batch_size=batch_size,
            make_optimizer=lambda _: optimizer,  # the run's optimizer, not a fresh one
            generator=generator,
        )
        test_acc, test_loss = evaluate(model, test_x, test_y, eval_batch_size)
        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            test_acc=test_acc,
            test_loss=test_loss,
            steps=steps,
            wall_s=time.perf_counter() - started,  # evaluate() syncs the device first
        )
        results.append(result)
        if on_epoch is not None:
            on_epoch(result)
    return results
