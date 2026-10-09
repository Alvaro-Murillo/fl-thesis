"""The federated simulation loop: a custom loop, not Flower.

Every round:

    select clients -> broadcast the global weights -> train each client
    locally -> aggregate the client weights -> evaluate the global model

This is the structure of FederatedAveraging (McMahan et al. 2017, Algorithm 1),
with the server rule and the client optimizer left to the caller. All clients
run in one process, one after another, on one device. There is no networking,
so "broadcast" is loading the global state into a worker model.

What the loop leaves to its caller:
    model           already built and initialised (the init seed is the caller's)
    aggregate       the server rule, (global_state, updates) -> new global state
    make_optimizer  called once per client per round, so no optimizer state
                    carries over between rounds
    proximal_mu     None for FedAvg; for FedProx (Li et al. 2020), the mu of the
                    proximal term every client adds to its local objective
                    (`federated/proximal.py`)
    on_round        receives each round's RoundResult, e.g. to write it to disk

Randomness. Each client's batch order comes from its own stream, keyed by
(train_seed, round, client), and client selection from a stream keyed by
(train_seed, round). A client's update therefore depends only on the global
state, its data and that key, not on which clients trained before it.

Training needs CUBLAS_WORKSPACE_CONFIG set before torch is imported, and this
module imports torch: entry scripts call
`experiments.seeding.enable_cublas_determinism()` before importing it.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from federated.data import FederatedData
from federated.proximal import add_proximal_gradient, check_mu, global_parameters

State = dict[str, torch.Tensor]

# spawn_key tags, so the selection and shuffling streams never coincide
_SELECT, _SHUFFLE = 0, 1


@dataclass(frozen=True)
class ClientUpdate:
    """What one client returns to the server after local training."""

    client: int
    num_samples: int
    state: State
    train_loss: float  # mean per-sample loss over the client's final local epoch
    steps: int  # optimizer steps taken this round


@dataclass(frozen=True)
class RoundResult:
    round: int  # 1-indexed
    test_acc: float
    test_loss: float
    clients: tuple[int, ...]  # clients that trained, in client order
    client_sizes: tuple[int, ...]
    client_train_loss: tuple[float, ...]
    client_steps: tuple[int, ...]
    skipped_empty: tuple[int, ...]  # selected but holding no samples
    wall_s: float


Aggregate = Callable[[State, Sequence[ClientUpdate]], State]
MakeOptimizer = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


def _stream_seed(train_seed: int, *key: int) -> int:
    sequence = np.random.SeedSequence(train_seed, spawn_key=key)
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def client_generator(train_seed: int, round: int, client: int) -> torch.Generator:
    """The batch-order stream for one client in one round."""
    gen = torch.Generator()
    gen.manual_seed(_stream_seed(train_seed, _SHUFFLE, round, client))
    return gen


def select_clients(num_clients: int, fraction: float, train_seed: int, round: int) -> list[int]:
    """max(floor(fraction * K), 1) clients without replacement, in client order.

    The rule m = max(C * K, 1) is McMahan et al. 2017, Algorithm 1; the floor for
    non-integer C * K is this module's choice. With fraction 1 every client is
    selected and no random number is drawn.
    """
    m = max(int(np.floor(fraction * num_clients + 1e-9)), 1)
    if m >= num_clients:
        return list(range(num_clients))
    rng = np.random.default_rng(_stream_seed(train_seed, _SELECT, round))
    return sorted(int(k) for k in rng.choice(num_clients, size=m, replace=False))


def snapshot(model: nn.Module) -> State:
    """A detached copy of the model's state, safe to keep across training."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def local_train(
    model: nn.Module,
    data: FederatedData,
    client: int,
    *,
    epochs: int,
    batch_size: int,
    make_optimizer: MakeOptimizer,
    generator: torch.Generator,
    proximal_mu: float | None = None,
) -> tuple[float, int]:
    """Train `model` in place on one client's data. Returns (final-epoch loss, steps).

    The optimizer is built here, so every call starts from a fresh optimizer state.

    With `proximal_mu` set, every step follows FedProx's objective, the batch's
    cross-entropy plus (mu / 2) * ||w - w_global||^2, where w_global is the model's
    parameters on entry: the term's gradient is added after backward()
    (`federated/proximal.py`). The returned loss is the cross-entropy alone, so
    FedAvg and FedProx clients report the same quantity. With `proximal_mu` None
    no term is applied.
    """
    if proximal_mu is not None:
        check_mu(proximal_mu)
    model.train()
    optimizer = make_optimizer(model.parameters())
    anchor = None if proximal_mu is None else global_parameters(model)
    steps = 0
    epoch_loss = torch.zeros((), device=data.device)
    for _ in range(epochs):
        epoch_loss = torch.zeros((), device=data.device)
        for x, y in data.batches(client, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if anchor is not None:
                add_proximal_gradient(model, anchor, proximal_mu)
            optimizer.step()
            epoch_loss += loss.detach() * len(y)
            steps += 1
    return float(epoch_loss) / data.size(client), steps


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1000
) -> tuple[float, float]:
    """(accuracy, mean cross-entropy) of `model` on (x, y)."""
    model.eval()
    correct = torch.zeros((), device=x.device)
    loss_sum = torch.zeros((), device=x.device)
    for start in range(0, len(y), batch_size):
        logits = model(x[start : start + batch_size])
        target = y[start : start + batch_size]
        loss_sum += F.cross_entropy(logits, target, reduction="sum")
        correct += (logits.argmax(dim=1) == target).sum()
    return float(correct) / len(y), float(loss_sum) / len(y)


def run_federated(
    model: nn.Module,
    data: FederatedData,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    make_optimizer: MakeOptimizer,
    aggregate: Aggregate,
    train_seed: int,
    client_fraction: float = 1.0,
    proximal_mu: float | None = None,
    on_round: Callable[[RoundResult], None] | None = None,
    eval_batch_size: int = 1000,
) -> list[RoundResult]:
    """Run `rounds` rounds; `model` ends holding the final global weights.

    `proximal_mu` None runs FedAvg's clients; a value runs FedProx's, each client
    adding the proximal term around the global weights it received that round.
    """
    devices = {p.device for p in model.parameters()} | {data.device, test_x.device}
    if len(devices) != 1:
        raise ValueError(f"model, training data and test data must share a device, got {devices}")
    if proximal_mu is not None:
        check_mu(proximal_mu)
    test_y = test_y.long()
    worker = copy.deepcopy(model)
    results = []
    for rnd in range(1, rounds + 1):
        started = time.perf_counter()
        global_state = snapshot(model)
        updates, skipped = [], []
        for k in select_clients(data.num_clients, client_fraction, train_seed, rnd):
            if data.size(k) == 0:
                skipped.append(k)
                continue
            worker.load_state_dict(global_state)
            loss, steps = local_train(
                worker,
                data,
                k,
                epochs=local_epochs,
                batch_size=batch_size,
                make_optimizer=make_optimizer,
                generator=client_generator(train_seed, rnd, k),
                proximal_mu=proximal_mu,
            )
            updates.append(ClientUpdate(k, data.size(k), snapshot(worker), loss, steps))
        if not updates:
            raise RuntimeError(f"round {rnd}: every selected client is empty")

        model.load_state_dict(aggregate(global_state, updates))
        test_acc, test_loss = evaluate(model, test_x, test_y, eval_batch_size)
        result = RoundResult(
            round=rnd,
            test_acc=test_acc,
            test_loss=test_loss,
            clients=tuple(u.client for u in updates),
            client_sizes=tuple(u.num_samples for u in updates),
            client_train_loss=tuple(u.train_loss for u in updates),
            client_steps=tuple(u.steps for u in updates),
            skipped_empty=tuple(skipped),
            wall_s=time.perf_counter() - started,  # evaluate() syncs the device first
        )
        results.append(result)
        if on_round is not None:
            on_round(result)
    return results
