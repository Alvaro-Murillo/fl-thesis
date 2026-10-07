"""Tests for the federated simulation loop and the client data it trains on.

The loop takes the server rule as a parameter. These tests use a plain unweighted
mean as a stand-in: it is not FedAvg's sample-weighted average, only something
that lets the loop run.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from federated.data import FederatedData
from federated.loop import (
    client_generator,
    evaluate,
    local_train,
    run_federated,
    select_clients,
    snapshot,
)

N_TRAIN, N_TEST, K = 300, 200, 5


def separable(n, seed):
    """2-D points labelled by the sign of x0 + x1."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    return x, (x.sum(dim=1) > 0).long()


def make_data(sizes=None, seed=0):
    x, y = separable(N_TRAIN, seed)
    perm = np.random.default_rng(seed).permutation(N_TRAIN)
    if sizes is None:
        parts = np.array_split(perm, K)
    else:
        parts = np.split(perm[: sum(sizes)], np.cumsum(sizes)[:-1])
    return FederatedData(x, y, parts)


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(2, 2)


def mean_of_states(global_state, updates):
    """Stand-in server rule: the unweighted mean of the client states."""
    return {
        name: torch.stack([u.state[name] for u in updates]).mean(dim=0) for name in global_state
    }


def adam(params):
    return torch.optim.Adam(params, lr=0.05)


def frozen(params):
    return torch.optim.SGD(params, lr=0.0)


def run(model=None, data=None, **overrides):
    test_x, test_y = separable(N_TEST, seed=99)
    kwargs = dict(
        rounds=3,
        local_epochs=2,
        batch_size=16,
        make_optimizer=adam,
        aggregate=mean_of_states,
        train_seed=123,
    )
    kwargs.update(overrides)
    model = make_model() if model is None else model
    data = make_data() if data is None else data
    return model, run_federated(model, data, test_x, test_y, **kwargs)


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- FederatedData ------------------------------------------------------------


def test_rejects_overlapping_clients():
    x, y = separable(10, 0)
    with pytest.raises(ValueError, match="more than one client"):
        FederatedData(x, y, [np.array([0, 1, 2]), np.array([2, 3])])


def test_rejects_out_of_range_index():
    x, y = separable(10, 0)
    with pytest.raises(ValueError, match="out of range"):
        FederatedData(x, y, [np.array([0, 10])])


def test_rejects_length_mismatch():
    x, y = separable(10, 0)
    with pytest.raises(ValueError, match="samples"):
        FederatedData(x, y[:9], [np.array([0])])


def test_epoch_covers_each_sample_once_and_keeps_last_partial_batch():
    data = make_data(sizes=[10, 50])
    batches = list(data.batches(0, 4, torch.Generator().manual_seed(0)))
    assert [len(y) for _, y in batches] == [4, 4, 2]
    seen = torch.cat([xb for xb, _ in batches])
    expected = data.x[data.indices[0]]
    assert sorted(map(tuple, seen.tolist())) == sorted(map(tuple, expected.tolist()))


def test_batch_order_follows_generator():
    data = make_data()

    def order(seed):
        return torch.cat([x for x, _ in data.batches(1, 7, torch.Generator().manual_seed(seed))])

    assert torch.equal(order(5), order(5))
    assert not torch.equal(order(5), order(6))


# --- seed streams and client selection ----------------------------------------


def test_client_streams_are_reproducible_and_distinct():
    def draw(*key):
        return torch.randperm(1000, generator=client_generator(*key))

    assert torch.equal(draw(1, 1, 0), draw(1, 1, 0))
    assert not torch.equal(draw(1, 1, 0), draw(1, 1, 1))  # another client
    assert not torch.equal(draw(1, 1, 0), draw(1, 2, 0))  # another round
    assert not torch.equal(draw(1, 1, 0), draw(2, 1, 0))  # another train seed


def test_full_participation_selects_every_client():
    assert select_clients(30, 1.0, train_seed=7, round=1) == list(range(30))


def test_partial_participation():
    picked = select_clients(30, 0.2, train_seed=7, round=1)
    assert len(picked) == 6 and picked == sorted(set(picked))
    assert picked == select_clients(30, 0.2, train_seed=7, round=1)
    rounds = {tuple(select_clients(30, 0.2, train_seed=7, round=r)) for r in range(1, 6)}
    assert len(rounds) > 1
    assert len(select_clients(30, 0.01, train_seed=7, round=1)) == 1  # max(floor(C * K), 1)


# --- local training and evaluation --------------------------------------------


def test_local_steps_and_final_epoch_loss():
    data = make_data(sizes=[37, 60])
    model = make_model()
    before = snapshot(model)
    loss, steps = local_train(
        model,
        data,
        0,
        epochs=3,
        batch_size=16,
        make_optimizer=frozen,
        generator=torch.Generator().manual_seed(0),
    )
    assert steps == 3 * 3  # ceil(37 / 16) batches per epoch
    assert states_equal(snapshot(model), before)  # lr 0 leaves the weights alone
    _, expected = evaluate(model, data.x[data.indices[0]], data.y[data.indices[0]])
    assert loss == pytest.approx(expected, rel=1e-6)


def test_evaluate_by_hand():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.bias.zero_()
    x = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    y = torch.tensor([0, 1, 1])
    acc, loss = evaluate(model, x, y, batch_size=2)
    right, wrong = np.log(1 + np.exp(-2.0)), np.log(1 + np.exp(2.0))
    assert acc == pytest.approx(2 / 3)
    assert loss == pytest.approx((2 * right + wrong) / 3, rel=1e-6)


# --- the round loop -----------------------------------------------------------


def test_every_client_starts_from_the_broadcast_weights():
    seen = []

    def record(global_state, updates):
        seen.append((global_state, updates))
        return mean_of_states(global_state, updates)

    run(rounds=1, make_optimizer=frozen, aggregate=record)
    global_state, updates = seen[0]
    assert len(updates) == K
    assert all(states_equal(u.state, global_state) for u in updates)


def test_client_update_does_not_depend_on_the_other_clients():
    """A client trained alone from the global state matches its update inside the round."""
    seen = []

    def record(global_state, updates):
        seen.append((global_state, updates))
        return mean_of_states(global_state, updates)

    data = make_data()
    run(data=data, rounds=2, aggregate=record)
    global_state, updates = seen[1]  # round 2: the global state is no longer the init
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
    )
    (update,) = [u for u in updates if u.client == client]
    assert states_equal(snapshot(alone), update.state)


def test_fresh_optimizer_per_client_per_round():
    built = []

    def counting(params):
        built.append(1)
        return adam(params)

    run(rounds=3, make_optimizer=counting)
    assert len(built) == 3 * K


def test_aggregate_output_becomes_the_global_model():
    def take_client_zero(global_state, updates):
        return updates[0].state

    seen = []

    def record(global_state, updates):
        seen.append(updates[0].state)
        return take_client_zero(global_state, updates)

    model, results = run(rounds=1, aggregate=record)
    assert states_equal(snapshot(model), seen[0])
    test_x, test_y = separable(N_TEST, seed=99)
    assert results[0].test_acc == evaluate(model, test_x, test_y)[0]


def test_round_results():
    calls = []
    _, results = run(rounds=3, on_round=calls.append)
    assert [r.round for r in results] == [1, 2, 3]
    assert calls == results
    sizes = tuple(len(p) for p in np.array_split(np.arange(N_TRAIN), K))
    for r in results:
        assert r.clients == tuple(range(K))
        assert r.client_sizes == sizes
        assert r.client_steps == tuple(2 * -(-n // 16) for n in sizes)
        assert r.skipped_empty == ()
        assert r.wall_s > 0


def test_empty_client_is_skipped_and_reported():
    seen = []

    def record(global_state, updates):
        seen.append([u.client for u in updates])
        return mean_of_states(global_state, updates)

    data = make_data(sizes=[100, 0, 100])
    _, results = run(data=data, rounds=2, aggregate=record)
    assert seen == [[0, 2], [0, 2]]
    assert all(r.skipped_empty == (1,) for r in results)


def test_reruns_are_identical():
    model_a, results_a = run()
    model_b, results_b = run()
    assert states_equal(snapshot(model_a), snapshot(model_b))
    assert [r.test_acc for r in results_a] == [r.test_acc for r in results_b]
    assert [r.client_train_loss for r in results_a] == [r.client_train_loss for r in results_b]


def test_train_seed_moves_batch_order():
    model_a, _ = run(train_seed=1)
    model_b, _ = run(train_seed=2)
    assert not states_equal(snapshot(model_a), snapshot(model_b))


def test_loop_learns_a_separable_task():
    test_x, test_y = separable(N_TEST, seed=99)
    model = make_model()
    before, _ = evaluate(model, test_x, test_y)
    _, results = run(model=model, rounds=5)
    assert results[-1].test_acc > 0.9
    assert results[-1].test_acc > before


def test_rejects_mixed_devices():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device")
    with pytest.raises(ValueError, match="share a device"):
        run(model=make_model().cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_cuda():
    x, y = separable(N_TRAIN, 0)
    parts = np.array_split(np.random.default_rng(0).permutation(N_TRAIN), K)
    data = FederatedData(x.cuda(), y.cuda(), parts)
    test_x, test_y = separable(N_TEST, seed=99)
    model = make_model().cuda()
    results = run_federated(
        model,
        data,
        test_x.cuda(),
        test_y.cuda(),
        rounds=2,
        local_epochs=1,
        batch_size=16,
        make_optimizer=adam,
        aggregate=mean_of_states,
        train_seed=0,
    )
    assert len(results) == 2
    assert all(p.is_cuda for p in model.parameters())


# A fresh interpreter per run: CUBLAS_WORKSPACE_CONFIG only takes effect if it is set
# before torch is imported, and pytest has imported torch already. The CNN is the
# benchmark copy in tests/gpu_check.py, a stand-in until the thesis models exist.
STRICT_GPU_RUN = textwrap.dedent("""
    import hashlib, sys
    sys.path.insert(0, {root!r})
    from experiments.seeding import configure_torch, derive_seeds, enable_cublas_determinism
    enable_cublas_determinism()

    import numpy as np
    import torch
    from torchvision.datasets import CIFAR10

    from federated.data import FederatedData
    from federated.loop import run_federated
    from tests.gpu_check import CNN

    def tensors(train):
        ds = CIFAR10("data", train=train, download=False)
        x = torch.as_tensor(np.asarray(ds.data)).permute(0, 3, 1, 2).float().div(255)
        return x.cuda(), torch.as_tensor(ds.targets).cuda()

    def mean_of_states(global_state, updates):
        return {{n: torch.stack([u.state[n] for u in updates]).mean(0) for n in global_state}}

    seeds = derive_seeds(0)
    configure_torch(seeds, deterministic=True)
    x, y = tensors(train=True)
    test_x, test_y = tensors(train=False)
    rng = np.random.default_rng(0)
    cuts = np.sort(rng.choice(np.arange(1, len(y)), size=29, replace=False))
    data = FederatedData(x, y, np.split(rng.permutation(len(y)), cuts))  # 30 unequal clients
    model = CNN().cuda()
    model(x[:2])  # materialise LazyLinear under the init seed
    results = run_federated(
        model, data, test_x, test_y, rounds=2, local_epochs=1, batch_size=64,
        make_optimizer=lambda p: torch.optim.Adam(p, lr=0.001),
        aggregate=mean_of_states, train_seed=seeds.train,
    )
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    print(torch.are_deterministic_algorithms_enabled())
    print(digest.hexdigest())
    print([r.test_acc for r in results], [r.client_train_loss for r in results])
""")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_strict_gpu_reruns_are_bit_exact_on_cifar10():
    """Two fresh processes, real CIFAR-10, K = 30 unequal clients, 2 rounds: same bytes."""
    root = Path(__file__).resolve().parents[1]
    script = STRICT_GPU_RUN.format(root=str(root))
    outputs = []
    for _ in range(2):
        done = subprocess.run(
            [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=600
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout.splitlines())
    first, second = outputs
    assert first[0] == "True"  # strict mode was really on
    assert first == second
