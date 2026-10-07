"""Tests for centralized training and the training tensors it reads."""

import numpy as np
import pytest
import torch

from experiments.config import DatasetConfig, ModelConfig
from experiments.tensors import LOADERS, images_to_unit_float, load_tensors
from federated.centralized import train_centralized
from federated.data import FederatedData
from federated.loop import local_train
from models.networks import build_model

DNN_CONFIG = ModelConfig(arch="dnn", in_channels=2, num_classes=2)


def separable(n, seed):
    """2-D points labelled by the sign of x0 + x1."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=gen)
    return x, (x.sum(dim=1) > 0).long()


def adam(params):
    return torch.optim.Adam(params, lr=0.01)


def train(model, epochs=3, train_seed=5, **kwargs):
    x, y = separable(300, seed=0)
    test_x, test_y = separable(200, seed=99)
    kwargs = dict(epochs=epochs, batch_size=32, make_optimizer=adam, train_seed=train_seed) | kwargs
    return train_centralized(model, x, y, test_x, test_y, **kwargs)


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- train_centralized ----------------------------------------------------------


def test_learns_a_separable_task():
    results = train(build_model(DNN_CONFIG, (2,), init_seed=0))
    assert results[-1].test_acc > 0.95
    assert results[-1].train_loss < results[0].train_loss


def test_epoch_results():
    seen = []
    results = train(build_model(DNN_CONFIG, (2,), init_seed=0), on_epoch=seen.append)
    assert seen == results
    assert [r.epoch for r in results] == [1, 2, 3]
    assert all(r.steps == 10 for r in results)  # ceil(300 / 32), last partial batch kept
    assert all(r.wall_s > 0 for r in results)


def test_one_optimizer_for_the_whole_run():
    built = []

    def counting_adam(params):
        built.append(adam(params))
        return built[-1]

    model = build_model(DNN_CONFIG, (2,), init_seed=0)
    train(model, make_optimizer=counting_adam)
    assert len(built) == 1
    state = built[0].state[next(model.parameters())]
    assert int(state["step"]) == 30  # Adam's moments carried across all 3 epochs


def test_matches_one_multi_epoch_local_train_bit_for_bit():
    """Epoch by epoch with one optimizer and one stream = a single E-epoch run."""
    a = build_model(DNN_CONFIG, (2,), init_seed=4)
    b = build_model(DNN_CONFIG, (2,), init_seed=4)
    train(a, epochs=3, train_seed=11)

    x, y = separable(300, seed=0)
    gen = torch.Generator()
    gen.manual_seed(11)
    data = FederatedData(x, y, [np.arange(300)])
    local_train(b, data, 0, epochs=3, batch_size=32, make_optimizer=adam, generator=gen)
    assert states_equal(a.state_dict(), b.state_dict())


def test_reruns_are_identical_and_train_seed_moves_batch_order():
    runs = []
    for seed in (5, 5, 6):
        model = build_model(DNN_CONFIG, (2,), init_seed=0)
        train(model, train_seed=seed)
        runs.append(model.state_dict())
    assert states_equal(runs[0], runs[1])
    assert not states_equal(runs[0], runs[2])


def test_rejects_mixed_devices():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device")
    with pytest.raises(ValueError, match="share a device"):
        train(build_model(DNN_CONFIG, (2,), init_seed=0).cuda())


# --- training tensors -----------------------------------------------------------


def test_images_become_unit_float_nchw():
    images = np.random.default_rng(0).integers(0, 256, size=(4, 5, 6, 3), dtype=np.uint8)
    images[0, 0, 0] = [0, 255, 128]
    x = images_to_unit_float(images)
    assert x.dtype == torch.float32 and x.shape == (4, 3, 5, 6) and x.is_contiguous()
    assert x[0, :, 0, 0].tolist() == [0.0, 1.0, float(np.float32(128) / np.float32(255))]
    restored = (x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    assert np.array_equal(restored, images)


def test_grayscale_images_get_one_channel():
    x = images_to_unit_float(np.zeros((2, 28, 28), dtype=np.uint8))
    assert x.shape == (2, 1, 28, 28)


def test_images_must_be_uint8():
    with pytest.raises(ValueError, match="uint8"):
        images_to_unit_float(np.zeros((2, 4, 4, 3), dtype=np.float32))


@pytest.fixture
def toy_dataset(monkeypatch):
    rng = np.random.default_rng(0)

    def loader(root, train):
        n = 12 if train else 6
        images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
        return images, np.arange(n) % 3

    monkeypatch.setitem(LOADERS, "toy", loader)
    return DatasetConfig(name="toy", num_classes=3)


def test_load_tensors(toy_dataset):
    data = load_tensors(toy_dataset)
    assert data.train_x.shape == (12, 3, 32, 32) and data.test_x.shape == (6, 3, 32, 32)
    assert data.train_y.dtype == torch.long and data.train_y.tolist() == [0, 1, 2] * 4
    assert data.input_shape == (3, 32, 32)
    assert 0.0 <= data.train_x.min() and data.train_x.max() <= 1.0


def test_load_tensors_checks_the_class_count(toy_dataset):
    with pytest.raises(ValueError, match="3 classes, config says 4"):
        load_tensors(DatasetConfig(name="toy", num_classes=4))


def test_load_tensors_rejects_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        load_tensors(DatasetConfig(name="nope", num_classes=10))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_load_tensors_onto_cuda(toy_dataset):
    data = load_tensors(toy_dataset, "cuda")
    cpu = load_tensors(toy_dataset)
    assert data.train_x.is_cuda and data.test_y.is_cuda
    assert data.train_x.shape == cpu.train_x.shape


# --- real CIFAR-10 --------------------------------------------------------------

CIFAR10 = DatasetConfig(name="cifar10", num_classes=10)


@pytest.mark.slow
def test_cifar10_tensors_line_up_with_the_partition_labels():
    from experiments.datasets import load_labels

    data = load_tensors(CIFAR10)
    assert data.train_x.shape == (50_000, 3, 32, 32) and data.test_x.shape == (10_000, 3, 32, 32)
    assert np.array_equal(data.train_y.numpy(), load_labels(CIFAR10))
    assert data.train_x.min() == 0.0 and data.train_x.max() == 1.0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cnn_trains_centrally_on_cifar10():
    """3 epochs at the federated settings (Adam 0.001, batch 64) reach about 0.61."""
    data = load_tensors(CIFAR10, "cuda")
    model = build_model(ModelConfig("cnn", 3, 10), data.input_shape, init_seed=0).cuda()
    results = train_centralized(
        model,
        data.train_x,
        data.train_y,
        data.test_x,
        data.test_y,
        epochs=3,
        batch_size=64,
        make_optimizer=lambda p: torch.optim.Adam(p, lr=0.001),
        train_seed=0,
    )
    accs = [r.test_acc for r in results]
    assert accs[-1] > 0.55
    assert accs[0] < accs[-1]
