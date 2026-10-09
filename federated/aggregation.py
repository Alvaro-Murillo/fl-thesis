"""FedAvg's server rule: the sample-weighted average of the client models.

After a round, the server replaces the global model with

    w_global <- sum_k (n_k / m_t) * w_k,    m_t = sum of n_j over the clients that trained

where w_k is client k's model after local training and n_k its number of
training samples. This is the aggregation step of FederatedAveraging, McMahan
et al. (2017), "Communication-Efficient Learning of Deep Networks from
Decentralized Data", Algorithm 1, checked against arXiv:1602.05629v4 (26 Jan
2023). There the sum and m_t run over S_t, the clients selected that round;
the paper's footnote 4 says earlier versions summed over all K clients, which
was an error. Here the sum runs over the clients `run_federated` passes in:
the selected clients that hold data. An empty client would have weight 0, so
skipping it changes nothing. With every client selected, m_t is the size of
the training set.

The weights n_k / m_t are what make FedAvg sensitive to client sizes: a
client's pull on the global model grows with its share of the data, so
quantity skew changes the aggregate even when the label distributions do not.

`global_state` is not used in the average. It fixes the keys, shapes and
dtypes every client state must have. Only floating-point entries are averaged;
the thesis's CNN and DNN have no buffers, so every entry is a weight or a bias.

The function is adapted from Flower's FedAvg (see the header below), the
framework the source paper used, and keeps its arithmetic: the factors are
Python floats, and each client's state is scaled and added in the order the
updates arrive, in the state's own dtype (float32). `run_federated` passes
the updates in client order, so the sum is deterministic.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from collections.abc import Sequence

from federated.loop import ClientUpdate, State


def _check(global_state: State, updates: Sequence[ClientUpdate]) -> None:
    if not updates:
        raise ValueError("fedavg needs at least one client update")
    for name, value in global_state.items():
        if not value.is_floating_point():
            raise ValueError(
                f"fedavg averages floating-point entries only; '{name}' is {value.dtype}"
            )
    for update in updates:
        if update.num_samples <= 0:
            raise ValueError(f"client {update.client}: num_samples must be positive")
        if update.state.keys() != global_state.keys():
            raise ValueError(f"client {update.client}: state keys differ from the global state")
        for name, value in update.state.items():
            reference = global_state[name]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(
                    f"client {update.client}: '{name}' is {value.dtype} {tuple(value.shape)}, "
                    f"the global state has {reference.dtype} {tuple(reference.shape)}"
                )


# Adapted from Flower 1.36.0, flwr/serverapp/strategy/strategy_utils.py, aggregate_arrayrecords
#   https://github.com/flwrlabs/flower/blob/framework-1.36.0/framework/py/flwr/serverapp/strategy/strategy_utils.py
#   (tag framework-1.36.0, commit 79163b48021c8113cf3cb538e173240b69eb03cd; the
#   installed flwr 1.36.0 file is identical)
#   Apache License 2.0, Copyright 2025 Flower Labs GmbH - see LICENSES/Flower-Apache-2.0.txt
#   Changes: client states are torch state dicts weighted by ClientUpdate.num_samples,
#   not RecordDicts weighted by a MetricRecord entry; returns a state dict, not an
#   ArrayRecord; input checks added (_check); renamed locals. The arithmetic is Flower's:
#   factor n_k / sum(n_j) as a Python float, value * factor, summed in update order.
def fedavg(global_state: State, updates: Sequence[ClientUpdate]) -> State:
    """Sample-weighted average of the client states, sum_k (n_k / sum_j n_j) * w_k.

    Has the signature `run_federated` expects of `aggregate`. The returned
    tensors are new; neither the client states nor `global_state` are changed.
    """
    _check(global_state, updates)
    total = sum(update.num_samples for update in updates)
    factors = [update.num_samples / total for update in updates]

    averaged: State = {}
    for update, factor in zip(updates, factors, strict=True):
        for name, value in update.state.items():
            if name not in averaged:
                averaged[name] = value * factor
            else:
                averaged[name] += value * factor
    return averaged
