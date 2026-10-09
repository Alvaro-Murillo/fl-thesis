"""Tests for FedProx's proximal term and its use in local training and the loop.

The central check: with mu = 0 FedProx must reduce exactly to FedAvg.
The term is still applied at mu = 0, so these tests compare it bit for bit with
`proximal_mu=None`, where no term is applied at all.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from federated.aggregation import fedavg
from federated.data import FederatedData
from federated.loop import client_generator, local_train, run_federated, snapshot
from federated.proximal import add_proximal_gradient, check_mu, global_parameters

N_TRAIN, N_TEST, K = 300, 200, 5


def separable(n, seed):
    """2-D points labelled by the sign of x0 + x1 (as in test_federated_loop.py)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    return x, (x.sum(dim=1) > 0).long()


def make_data(sizes=(37, 60, 80, 50, 73), seed=0):
    x, y = separable(sum(sizes), seed)
    perm = np.random.default_rng(seed).permutation(sum(sizes))
    return FederatedData(x, y, np.split(perm, np.cumsum(sizes)[:-1]))


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))


def adam(params):
    return torch.optim.Adam(params, lr=0.05)


def train_client(model, data, mu, client=0, epochs=3, make_optimizer=adam, seed=0):
    return local_train(
        model,
        data,
        client,
        epochs=epochs,
        batch_size=16,
        make_optimizer=make_optimizer,
        generator=torch.Generator().manual_seed(seed),
        proximal_mu=mu,
    )


def run(mu, data=None, rounds=3, **overrides):
    test_x, test_y = separable(N_TEST, seed=99)
    model = make_model()
    kwargs = dict(
        rounds=rounds,
        local_epochs=2,
        batch_size=16,
        make_optimizer=adam,
        aggregate=fedavg,
        train_seed=123,
        proximal_mu=mu,
    )
    kwargs.update(overrides)
    results = run_federated(model, make_data() if data is None else data, test_x, test_y, **kwargs)
    return model, results


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def distance(a, b):
    return math.sqrt(sum(float(((a[k] - b[k]).double() ** 2).sum()) for k in a))


# --- the term on its own ------------------------------------------------------


def test_proximal_gradient_by_hand():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 3.0]]))
        model.bias.copy_(torch.tensor([2.0]))
    anchor = [torch.tensor([[0.0, 1.0]]), torch.tensor([2.5])]
    model.weight.grad = torch.tensor([[0.5, 0.0]])
    model.bias.grad = torch.tensor([1.0])
    add_proximal_gradient(model, anchor, mu=2.0)
    assert torch.equal(model.weight.grad, torch.tensor([[2.5, 4.0]]))  # g + 2 * (w - w_global)
    assert torch.equal(model.bias.grad, torch.tensor([0.0]))


def test_matches_autograd_of_the_proximal_loss():
    """The gradient form equals backward() through CE + (mu / 2) * ||w - w_global||^2."""
    x, y = separable(64, seed=1)
    model = make_model(seed=1)
    anchor = global_parameters(make_model(seed=2))
    for mu in [0.001, 0.1, 10.0]:
        model.zero_grad()
        term = sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
        (F.cross_entropy(model(x), y) + (mu / 2) * term).backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        add_proximal_gradient(model, anchor, mu)
        for p, g in zip(model.parameters(), expected):
            assert torch.allclose(p.grad, g, rtol=1e-6, atol=1e-7)


def test_mu_zero_leaves_gradients_unchanged():
    x, y = separable(64, seed=1)
    model = make_model(seed=1)
    anchor = global_parameters(make_model(seed=2))
    F.cross_entropy(model(x), y).backward()
    before = [p.grad.clone() for p in model.parameters()]
    add_proximal_gradient(model, anchor, 0.0)
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), before))


def test_parameter_without_gradient_is_skipped():
    model = nn.Linear(2, 1)
    anchor = global_parameters(nn.Linear(2, 1))
    model.weight.grad = torch.zeros(1, 2)
    add_proximal_gradient(model, anchor, 1.0)
    assert model.bias.grad is None


def test_global_parameters_are_frozen_copies():
    model = make_model()
    anchor = global_parameters(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert all(not a.requires_grad for a in anchor)
    assert all(not torch.equal(p, a) for p, a in zip(model.parameters(), anchor))


@pytest.mark.parametrize("mu", [-0.1, math.nan, math.inf])
def test_rejects_bad_mu(mu):
    with pytest.raises(ValueError, match="proximal_mu"):
        check_mu(mu)
    with pytest.raises(ValueError, match="proximal_mu"):
        train_client(make_model(), make_data(), mu)
    with pytest.raises(ValueError, match="proximal_mu"):
        run(mu)


def test_accepts_mu_zero():
    check_mu(0.0)


# --- local training -----------------------------------------------------------


def test_mu_zero_is_fedavg_bit_for_bit_in_local_training():
    data = make_data()
    fedavg_client, fedprox_client = make_model(), make_model()
    loss_a, steps_a = train_client(fedavg_client, data, None)
    loss_b, steps_b = train_client(fedprox_client, data, 0.0)
    assert states_equal(snapshot(fedavg_client), snapshot(fedprox_client))
    assert (loss_a, steps_a) == (loss_b, steps_b)


def test_with_sgd_a_step_is_the_perturbed_gradient_step():
    """Two SGD steps by hand: w <- w - lr * (grad + mu * (w - w_global)), w_global fixed at entry."""
    lr, mu = 0.1, 0.5
    data = make_data(sizes=(32,))
    model = make_model()
    start = global_parameters(model)
    batches = list(data.batches(0, 16, torch.Generator().manual_seed(0)))
    weights = [w.clone() for w in start]
    for x, y in batches:
        params = [w.clone().requires_grad_() for w in weights]
        hidden = F.relu(F.linear(x, params[0], params[1]))
        loss = F.cross_entropy(F.linear(hidden, params[2], params[3]), y)
        grads = torch.autograd.grad(loss, params)
        weights = [w - lr * (g + mu * (w - w0)) for w, g, w0 in zip(weights, grads, start)]
    train_client(model, data, mu, epochs=1, make_optimizer=lambda p: torch.optim.SGD(p, lr=lr))
    for p, w in zip(model.parameters(), weights):
        assert torch.allclose(p, w, rtol=1e-6, atol=1e-7)


def test_larger_mu_keeps_the_client_closer_to_the_global_model():
    data = make_data()
    start = snapshot(make_model())
    drift = []
    for mu in [0.0, 0.1, 1.0, 10.0]:
        model = make_model()
        train_client(model, data, mu, epochs=5)
        drift.append(distance(snapshot(model), start))
    assert drift == sorted(drift, reverse=True) and drift[-1] < drift[0] / 5


def test_reported_loss_is_the_cross_entropy_alone():
    """One batch, one epoch: the loss is taken before any step, so mu must not change it."""
    data = make_data(sizes=(16,))
    reported = [train_client(make_model(), data, mu, epochs=1)[0] for mu in [None, 100.0]]
    assert reported[0] == reported[1]


# --- the loop -----------------------------------------------------------------


def test_mu_zero_is_fedavg_bit_for_bit_in_the_loop():
    model_a, results_a = run(None)
    model_b, results_b = run(0.0)
    assert states_equal(snapshot(model_a), snapshot(model_b))
    untimed = [[replace(r, wall_s=0.0) for r in results] for results in (results_a, results_b)]
    assert untimed[0] == untimed[1]  # accuracy, test loss, client losses and steps


def test_mu_changes_the_run():
    model_a, _ = run(None)
    model_b, _ = run(0.1)
    assert not states_equal(snapshot(model_a), snapshot(model_b))


def test_each_client_anchors_on_the_global_model_it_received():
    """A client trained alone from round 2's global state matches its update inside the round."""
    seen = []

    def record(global_state, updates):
        seen.append((global_state, updates))
        return fedavg(global_state, updates)

    data = make_data()
    run(0.5, data=data, rounds=2, aggregate=record)
    global_state, updates = seen[1]
    client = 3
    alone = make_model()
    alone.load_state_dict(global_state)
    local_train(
        alone,
        data,
        client,
        epochs=2,
        batch_size=16,
        make_optimizer=adam,
        generator=client_generator(123, 2, client),
        proximal_mu=0.5,
    )
    (update,) = [u for u in updates if u.client == client]
    assert states_equal(snapshot(alone), update.state)


def test_fedprox_reruns_are_identical():
    model_a, results_a = run(0.1)
    model_b, results_b = run(0.1)
    assert states_equal(snapshot(model_a), snapshot(model_b))
    assert [r.test_acc for r in results_a] == [r.test_acc for r in results_b]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_mu_zero_is_fedavg_bit_for_bit_on_cuda():
    x, y = separable(N_TRAIN, 0)
    parts = np.array_split(np.random.default_rng(0).permutation(N_TRAIN), K)
    test_x, test_y = separable(N_TEST, seed=99)
    states = []
    for mu in [None, 0.0]:
        model = make_model().cuda()
        run_federated(
            model,
            FederatedData(x.cuda(), y.cuda(), parts),
            test_x.cuda(),
            test_y.cuda(),
            rounds=2,
            local_epochs=2,
            batch_size=16,
            make_optimizer=adam,
            aggregate=fedavg,
            train_seed=0,
            proximal_mu=mu,
        )
        states.append(snapshot(model))
    assert states_equal(*states)
