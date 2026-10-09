"""Tests for FedAvg's server rule, the sample-weighted average of client states."""

import numpy as np
import pytest
import torch
from torch import nn

from federated.aggregation import fedavg
from federated.data import FederatedData
from federated.loop import ClientUpdate, run_federated, snapshot


def separable(n, seed):
    """2-D points labelled by the sign of x0 + x1 (as in test_federated_loop.py)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    return x, (x.sum(dim=1) > 0).long()


def run_toy(sizes, aggregate, rounds, local_epochs, train_seed):
    """`run_federated` on the separable task with clients of the given sizes."""
    x, y = separable(sum(sizes), seed=0)
    parts = np.split(np.random.default_rng(0).permutation(sum(sizes)), np.cumsum(sizes)[:-1])
    test_x, test_y = separable(200, seed=99)
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    results = run_federated(
        model, FederatedData(x, y, parts), test_x, test_y, rounds=rounds,
        local_epochs=local_epochs, batch_size=16,
        make_optimizer=lambda params: torch.optim.Adam(params, lr=0.05),
        aggregate=aggregate, train_seed=train_seed,
    )  # fmt: skip
    return model, results


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def update(client, num_samples, **state):
    return ClientUpdate(client, num_samples, state, train_loss=0.0, steps=1)


def reference(**state):
    return {name: torch.zeros_like(value) for name, value in state.items()}


def test_weighted_average_by_hand():
    a = {"w": torch.tensor([0.0, 4.0]), "b": torch.tensor([8.0])}
    b = {"w": torch.tensor([4.0, 0.0]), "b": torch.tensor([0.0])}
    out = fedavg(reference(**a), [update(0, 1, **a), update(1, 3, **b)])
    assert torch.equal(out["w"], torch.tensor([3.0, 1.0]))
    assert torch.equal(out["b"], torch.tensor([2.0]))


def test_larger_client_pulls_harder():
    small, large = torch.tensor([0.0]), torch.tensor([1.0])
    for n_large, expected in [(1, 0.5), (3, 0.75), (7, 0.875)]:
        out = fedavg(reference(w=small), [update(0, 1, w=small), update(1, n_large, w=large)])
        assert out["w"].item() == expected


def test_equal_sizes_give_the_unweighted_mean():
    gen = torch.Generator().manual_seed(0)
    states = [{"w": torch.randn(5, 3, generator=gen)} for _ in range(4)]
    out = fedavg(states[0], [update(k, 250, **s) for k, s in enumerate(states)])
    mean = torch.stack([s["w"] for s in states]).mean(dim=0)
    assert torch.allclose(out["w"], mean, rtol=0, atol=1e-6)


def test_only_relative_sizes_matter():
    gen = torch.Generator().manual_seed(1)
    states = [{"w": torch.randn(7, generator=gen)} for _ in range(3)]
    small = fedavg(
        states[0], [update(k, n, **s) for k, (n, s) in enumerate(zip([1, 2, 3], states))]
    )
    large = fedavg(
        states[0], [update(k, n, **s) for k, (n, s) in enumerate(zip([100, 200, 300], states))]
    )
    assert states_equal(small, large)  # n / total is the same float either way


def test_single_client_is_returned_unchanged():
    w = torch.randn(4, 4, generator=torch.Generator().manual_seed(2))
    out = fedavg(reference(w=w), [update(0, 37, w=w)])
    assert torch.equal(out["w"], w)


def test_identical_clients_are_a_fixed_point():
    w = torch.randn(6, generator=torch.Generator().manual_seed(3))
    out = fedavg(reference(w=w), [update(k, n, w=w.clone()) for k, n in enumerate([1, 1, 2])])
    assert torch.equal(out["w"], w)  # factors 1/4, 1/4, 1/2 are exact


def test_matches_the_formula_with_unequal_sizes():
    gen = torch.Generator().manual_seed(4)
    sizes = [1500, 31, 4012, 777, 60]
    states = [
        {"w": torch.randn(3, 2, generator=gen), "b": torch.randn(2, generator=gen)} for _ in sizes
    ]
    out = fedavg(states[0], [update(k, n, **s) for k, (n, s) in enumerate(zip(sizes, states))])
    total = sum(sizes)
    for name in ("w", "b"):
        expected = sum(n / total * s[name].double() for n, s in zip(sizes, states))
        assert torch.allclose(out[name].double(), expected, rtol=0, atol=1e-6)
        assert out[name].dtype == torch.float32


def test_inputs_are_left_untouched_and_outputs_are_new_tensors():
    a, b = torch.ones(3), torch.zeros(3)
    global_state = {"w": torch.full((3,), 5.0)}
    out = fedavg(global_state, [update(0, 2, w=a), update(1, 2, w=b)])
    out["w"] += 100
    assert torch.equal(a, torch.ones(3)) and torch.equal(b, torch.zeros(3))
    assert torch.equal(global_state["w"], torch.full((3,), 5.0))


def test_rejects_no_updates():
    with pytest.raises(ValueError, match="at least one"):
        fedavg(reference(w=torch.zeros(1)), [])


@pytest.mark.parametrize("n", [0, -3])
def test_rejects_non_positive_sizes(n):
    w = torch.zeros(2)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        fedavg(reference(w=w), [update(0, 5, w=w), update(1, n, w=w)])


def test_rejects_mismatched_keys():
    with pytest.raises(ValueError, match="keys differ"):
        fedavg(reference(w=torch.zeros(2)), [update(0, 1, v=torch.zeros(2))])


def test_rejects_a_shape_that_would_broadcast():
    with pytest.raises(ValueError, match="global state has"):
        fedavg(reference(w=torch.zeros(3)), [update(0, 1, w=torch.zeros(1))])


def test_rejects_a_dtype_mismatch():
    with pytest.raises(ValueError, match="global state has"):
        fedavg(reference(w=torch.zeros(2)), [update(0, 1, w=torch.zeros(2, dtype=torch.float64))])


def test_rejects_integer_entries():
    bn = nn.BatchNorm1d(2)  # num_batches_tracked is an integer buffer
    state = snapshot(bn)
    with pytest.raises(ValueError, match="floating-point entries only"):
        fedavg(state, [update(0, 1, **state)])


def test_is_the_global_model_after_a_round():
    seen = []

    def recording_fedavg(global_state, updates):
        seen.append((global_state, updates))
        return fedavg(global_state, updates)

    model, _ = run_toy(
        [10, 90, 35, 5, 60], recording_fedavg, rounds=2, local_epochs=1, train_seed=0
    )
    global_state, updates = seen[-1]
    assert [u.num_samples for u in updates] == [10, 90, 35, 5, 60]
    assert states_equal(snapshot(model), fedavg(global_state, updates))


def test_loop_with_fedavg_learns_with_unequal_clients():
    _, results = run_toy([5, 150, 20, 60, 65], fedavg, rounds=5, local_epochs=2, train_seed=123)
    assert results[-1].test_acc > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_bit_for_bit():
    gen = torch.Generator().manual_seed(5)
    sizes = [1666, 1, 523, 3000]
    states = [{"w": torch.randn(64, 1024, generator=gen)} for _ in sizes]
    cpu = fedavg(states[0], [update(k, n, **s) for k, (n, s) in enumerate(zip(sizes, states))])
    cuda_states = [{"w": s["w"].cuda()} for s in states]
    gpu = fedavg(
        cuda_states[0], [update(k, n, **s) for k, (n, s) in enumerate(zip(sizes, cuda_states))]
    )
    assert gpu["w"].is_cuda
    assert torch.equal(gpu["w"].cpu(), cpu["w"])
