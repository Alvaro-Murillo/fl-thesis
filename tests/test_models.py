"""Tests for the CNN and DNN and their Glorot-uniform initialisation."""

import math

import pytest
import torch
from torch import nn

from experiments.config import ARCHS, ModelConfig
from models.networks import CNN, DNN, build_model, glorot_init_
from models.networks import ARCHS as BUILDERS

CIFAR = ModelConfig(arch="cnn", in_channels=3, num_classes=10)
COVTYPE = ModelConfig(arch="dnn", in_channels=54, num_classes=7)


def layers(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# --- architecture ---------------------------------------------------------------


def test_cnn_layers_on_cifar10():
    model = build_model(CIFAR, (3, 32, 32), init_seed=0)
    shapes = [tuple(m.weight.shape) for m in layers(model)]
    assert shapes == [(32, 3, 3, 3), (64, 32, 3, 3), (64, 64, 3, 3), (64, 1024), (10, 64)]
    # by hand: 896 + 18,496 + 36,928 + 65,600 + 650
    assert sum(p.numel() for p in model.parameters()) == 122_570
    assert [type(m) for m in model.modules() if isinstance(m, nn.MaxPool2d)] == [nn.MaxPool2d] * 2


def test_cnn_output_and_intermediate_shapes():
    model = CNN(3, 10, (32, 32))
    x = torch.rand(5, 3, 32, 32)
    assert model.features(x).shape == (5, 1024)
    assert model(x).shape == (5, 10)


def test_cnn_on_grayscale_28x28():
    model = build_model(ModelConfig("cnn", 1, 10), (1, 28, 28), init_seed=0)
    assert layers(model)[3].in_features == 3 * 3 * 64  # 28 -> 26 -> 13 -> 11 -> 5 -> 3
    assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)


def test_cnn_rejects_too_small_images():
    with pytest.raises(ValueError, match="too small"):
        CNN(3, 10, (16, 16))


def test_dnn_layers():
    model = build_model(COVTYPE, (54,), init_seed=0)
    shapes = [tuple(m.weight.shape) for m in layers(model)]
    assert shapes == [(500, 54), (500, 500), (500, 500), (7, 500)]
    assert sum(isinstance(m, nn.ReLU) for m in model.modules()) == 3
    assert model(torch.rand(4, 54)).shape == (4, 7)


def test_models_have_no_layers_that_depend_on_train_or_eval_mode():
    for model in (CNN(3, 10, (32, 32)), DNN(54, 7)):
        allowed = (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten, nn.Sequential)
        assert all(isinstance(m, allowed) for m in model.modules() if m is not model)


# --- initialisation -------------------------------------------------------------


@pytest.mark.parametrize("config, shape", [(CIFAR, (3, 32, 32)), (COVTYPE, (54,))])
def test_glorot_uniform_bounds_and_zero_biases(config, shape):
    model = build_model(config, shape, init_seed=1)
    for layer in layers(model):
        w = layer.weight
        receptive = math.prod(w.shape[2:])  # 1 for a dense layer
        fan_in, fan_out = w.shape[1] * receptive, w.shape[0] * receptive
        limit = math.sqrt(6 / (fan_in + fan_out))
        assert w.abs().max() <= limit
        assert w.abs().max() > 0.9 * limit  # the whole interval is used
        assert torch.count_nonzero(layer.bias) == 0


def test_glorot_uniform_spread():
    """A uniform on [-a, a] has standard deviation a / sqrt(3)."""
    model = build_model(CIFAR, (3, 32, 32), init_seed=2)
    dense = layers(model)[3].weight  # 64 x 1024
    limit = math.sqrt(6 / (1024 + 64))
    assert dense.std().item() == pytest.approx(limit / math.sqrt(3), rel=0.02)
    assert dense.mean().item() == pytest.approx(0.0, abs=0.01 * limit)


def test_init_depends_only_on_the_seed():
    torch.manual_seed(0)
    a = build_model(CIFAR, (3, 32, 32), init_seed=7)
    torch.manual_seed(12345)  # the global RNG is not used for the weights
    b = build_model(CIFAR, (3, 32, 32), init_seed=7)
    c = build_model(CIFAR, (3, 32, 32), init_seed=8)
    assert states_equal(a.state_dict(), b.state_dict())
    assert not torch.equal(layers(a)[0].weight, layers(c)[0].weight)


def test_glorot_init_refuses_a_model_off_the_cpu():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    with pytest.raises(ValueError, match="on the CPU"):
        glorot_init_(DNN(4, 2).cuda(), seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_same_weights_after_moving_to_cuda():
    cpu = build_model(CIFAR, (3, 32, 32), init_seed=3)
    gpu = build_model(CIFAR, (3, 32, 32), init_seed=3).cuda()
    assert all(torch.equal(a, b.cpu()) for a, b in zip(cpu.parameters(), gpu.parameters()))
    assert gpu(torch.rand(2, 3, 32, 32, device="cuda")).shape == (2, 10)


# --- build_model and the config -------------------------------------------------


def test_builders_match_the_config_schema():
    assert set(BUILDERS) == ARCHS


def test_config_rejects_unknown_arch():
    with pytest.raises(ValueError, match="arch must be one of"):
        ModelConfig(arch="resnet", in_channels=3, num_classes=10)


@pytest.mark.parametrize(
    "config, shape, match",
    [
        (CIFAR, (1, 32, 32), "in_channels is 3"),
        (CIFAR, (3 * 32 * 32,), r"\(channels, height, width\)"),
        (COVTYPE, (55,), "feature count"),
        (COVTYPE, (1, 54), r"\(features,\)"),
    ],
)
def test_build_model_checks_the_data_shape(config, shape, match):
    with pytest.raises(ValueError, match=match):
        build_model(config, shape, init_seed=0)
