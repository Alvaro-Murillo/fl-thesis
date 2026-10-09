"""FedProx's proximal term, (mu / 2) * ||w - w_global||^2, on a client's local objective.

FedProx (Li et al. 2020, "Federated Optimization in Heterogeneous Networks",
MLSys 2020; Section 3.2, Eq. 2 and Algorithm 2, checked against the ar5iv
rendering of arXiv:1812.06127) has each selected client k approximately minimise

    h_k(w; w^t) = F_k(w) + (mu / 2) * ||w - w^t||^2

instead of F_k(w) alone, where w^t is the global model the client received
that round. The term pulls the client's model back towards w^t, so its local
epochs carry it less far from the global model. The paper names FedAvg the
special case mu = 0 (with SGD as the local solver).

How the term enters training. Its gradient is mu * (w - w^t). After the
batch's cross-entropy has been backpropagated, `add_proximal_gradient` adds
mu * (w - w^t) to every parameter's gradient, and the optimizer then steps on
the sum. This is how Li et al.'s own implementation applies the term (see the
header below): their PerturbedGradientDescent steps w - lr * (grad + mu * (w - w*)).
The thesis's optimizer is Adam, not SGD, so Adam receives the same total
gradient as it would if (mu / 2) * ||w - w^t||^2 had been added to the loss
before backward(), which is how Flower's FedProx baseline writes it. The two
forms agree up to float rounding. In the federated loop on CIFAR-10 the loss
form costs about twice FedAvg's step time, this form about 15% more.

w^t is a detached copy of the model's parameters taken when local training
starts, right after the broadcast, and stays fixed for the whole round
(`global_parameters`). Only parameters take part, not buffers; the thesis's
CNN and DNN have none.

At mu = 0 the term is still applied. It adds exactly zero to every gradient,
so the client's update is bit-identical to FedAvg's. This is the sanity check
for the implementation, and it is tested. `local_train` skips
the term altogether when `proximal_mu` is None, which is FedAvg's setting.

The server side of FedProx is the caller's server rule. The thesis uses
FedAvg's sample-weighted average (`federated.aggregation.fedavg`) for both
algorithms, so FedAvg and FedProx runs differ only in this term. Li et al.'s
code averages by sample count too (`aggregate` in flearn/trainers/fedbase.py),
and so does Flower's FedProx strategy, which subclasses FedAvg and only sends
mu to the clients. Algorithm 2 in the paper instead selects clients with
probability p_k and takes the plain mean of their models.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def check_mu(mu: float) -> None:
    """Refuse a mu that is negative, infinite or NaN; 0 is allowed."""
    if not math.isfinite(mu) or mu < 0:
        raise ValueError(f"proximal_mu must be finite and >= 0, got {mu}")


def global_parameters(model: nn.Module) -> list[torch.Tensor]:
    """w^t: detached copies of the model's parameters, taken before local training starts."""
    return [value.detach().clone() for value in model.parameters()]


# Adapted from Li et al.'s FedProx code, flearn/optimizer/pgd.py,
# PerturbedGradientDescent._apply_dense (TensorFlow 1)
#   https://github.com/litian96/FedProx/blob/d2a4501f319f1594b732d88315c5ca1a72855f50/flearn/optimizer/pgd.py
#   (the repository's latest commit, 2021-04-14; the file last changed in b00d206, 2019-06-24)
#   MIT License, Copyright (c) 2019. Tian Li - see LICENSES/FedProx-MIT.txt
#   Changes: torch instead of a TensorFlow optimizer. Only the perturbed gradient
#   grad + mu * (var - vstar) is kept; it is written into each parameter's .grad in place,
#   and the caller's optimizer (Adam here) takes the step instead of var -= lr * (...).
#   vstar, the optimizer slot that `set_params` loads with the round's global model, is the
#   `global_params` argument. A parameter without a gradient is skipped, as TensorFlow's
#   apply_gradients skips it in the original; every parameter of the thesis's models has one.
@torch.no_grad()
def add_proximal_gradient(
    model: nn.Module, global_params: Sequence[torch.Tensor], mu: float
) -> None:
    """grad += mu * (w - w^t) for every parameter: the gradient of (mu / 2) * ||w - w^t||^2.

    Call it after backward() and before the optimizer step.
    """
    for var, vstar in zip(model.parameters(), global_params, strict=True):
        if var.grad is not None:
            var.grad.add_(var - vstar, alpha=mu)
