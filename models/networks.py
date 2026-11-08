"""The source paper's two models: a small CNN for images and a DNN for tabular data.

Both follow the descriptions in Jimenez-Gutierrez et al. (2026), "A thorough
assessment of the non-IID data impact in federated learning" (JII 50, 101052),
Section 4, "Models". The text was checked against arXiv:2503.17070 v1 and v2;
the journal version was not checked.

    CNN  conv 32 3x3 ReLU -> max pool -> conv 64 3x3 ReLU -> max pool
         -> conv 64 3x3 ReLU -> flatten -> dense 64 ReLU -> dense num_classes
    DNN  dense 500 ReLU -> dense 500 ReLU -> dense 500 ReLU -> dense num_classes

The paper gives neither padding, pooling size, initialisation nor the CNN's
output layer, and names no library besides Flower. Those are this module's
choices, all Keras defaults: unpadded ("valid") 3x3 convolutions with stride 1,
2x2 max pooling with stride 2, Glorot-uniform weights with zero biases, and a
num_classes output layer as in the DNN. With 32x32x3 inputs the flatten has
4 * 4 * 64 = 1024 features and the CNN 122,570 parameters.

The DNN is the paper's model for its tabular datasets, Covtype among them. The
paper (v1 and v2) describes an input layer with one unit per feature, three hidden
layers of 500 ReLU units and a softmax output layer with one unit per class. The
input layer is read as the input itself, as a Keras `Input`, not as a trained
dense layer of width num_features. The paper gives no initialisation, bias,
dropout or normalisation for it either; the choices are the CNN's, Keras
defaults: Glorot-uniform weights, biases in every layer and zero at the start, no
dropout, no normalisation. On Covtype (54 features, 7 classes) it is
54 -> 500 -> 500 -> 500 -> 7 with 532,007 parameters.

Both models return logits. The DNN's softmax output layer is the softmax inside
`F.cross_entropy`, and argmax is the same before and after it. The flatten runs
over (channel, row, column), where Keras flattens (row, column, channel); this
only permutes the dense layer's inputs, which are initialised i.i.d., so the
initial distribution of the model is unchanged.

Initialisation is the normalized initialization of Glorot & Bengio (2010),
Section 4.2.1, eq. (16), with the biases set to 0 as in their Section 2.3:

    W ~ U[-sqrt(6 / (fan_in + fan_out)), +sqrt(6 / (fan_in + fan_out))]

with fan_in = in_channels * kh * kw and fan_out = out_channels * kh * kw for a
convolution, which is what `nn.init.xavier_uniform_` computes and what Keras's
GlorotUniform uses. The draws come from a CPU generator seeded with the init
seed, never from torch's global RNG, so a seed gives the same weights on any
device.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

HIDDEN_UNITS = 500  # each of the DNN's three hidden layers


def _after_features(size: int) -> int:
    """Spatial size after conv3 -> pool2 -> conv3 -> pool2 -> conv3, no padding."""
    size = (size - 2) // 2
    size = (size - 2) // 2
    return size - 2


class CNN(nn.Module):
    """Three convolutional blocks, then dense 64 and the output layer."""

    def __init__(self, in_channels: int, num_classes: int, image_size: tuple[int, int]):
        super().__init__()
        height, width = (_after_features(s) for s in image_size)
        if height < 1 or width < 1:
            raise ValueError(f"images of size {tuple(image_size)} are too small for the CNN")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * height * width, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class DNN(nn.Module):
    """Three hidden layers of 500 ReLU units, then the output layer."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def glorot_init_(model: nn.Module, seed: int) -> nn.Module:
    """Glorot-uniform weights and zero biases for every conv and dense layer, in place.

    Layers are visited in registration order and all draws come from one CPU
    generator seeded with `seed`. The model must be on the CPU.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight.device.type != "cpu":
                raise ValueError("initialise on the CPU, then move the model to its device")
            nn.init.xavier_uniform_(module.weight, generator=generator)
            nn.init.zeros_(module.bias)
    return model


def _cnn(in_channels: int, num_classes: int, input_shape: Sequence[int]) -> nn.Module:
    if len(input_shape) != 3:
        raise ValueError(f"the cnn takes (channels, height, width) inputs, got {input_shape}")
    if input_shape[0] != in_channels:
        raise ValueError(f"model.in_channels is {in_channels}, the data has {input_shape[0]}")
    return CNN(in_channels, num_classes, (input_shape[1], input_shape[2]))


def _dnn(in_channels: int, num_classes: int, input_shape: Sequence[int]) -> nn.Module:
    if len(input_shape) != 1:
        raise ValueError(f"the dnn takes (features,) inputs, got {input_shape}")
    if input_shape[0] != in_channels:
        raise ValueError(
            f"model.in_channels (the dnn's feature count) is {in_channels}, "
            f"the data has {input_shape[0]}"
        )
    return DNN(in_channels, num_classes)


ARCHS = {"cnn": _cnn, "dnn": _dnn}


def build_model(config, input_shape: Sequence[int], init_seed: int) -> nn.Module:
    """The model a ModelConfig names, Glorot-initialised from `init_seed`, on the CPU.

    `input_shape` is one sample's shape, taken from the data: (C, H, W) for the
    cnn, (features,) for the dnn. For the dnn, `config.in_channels` is the
    feature count. Move the model to its device afterwards.
    """
    if config.arch not in ARCHS:
        raise KeyError(f"unknown arch '{config.arch}'; known: {sorted(ARCHS)}")
    model = ARCHS[config.arch](config.in_channels, config.num_classes, tuple(input_shape))
    return glorot_init_(model, init_seed)
