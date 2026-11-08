"""Training and test tensors for model training.

Partitioning reads labels only, from `experiments.datasets`; training needs the
inputs too. Both load the same torchvision object, so row i of the training
tensors is sample i of the label vector a partition indexes into.

Images become float32 in [0, 1] (pixels / 255), laid out (N, C, H, W). There
is no standardisation and no augmentation. The conversion runs on the CPU and
the result is then moved to the device, so the values do not depend on it.

Covtype's features are already float32 (N, 54), split and standardised by
`experiments.covtype`, which also gives `experiments.datasets` its labels; they
are moved to the device as they are.

This module imports torch: entry scripts call
`experiments.seeding.enable_cublas_determinism()` before importing it.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class TensorData:
    train_x: torch.Tensor
    train_y: torch.Tensor  # int64 class indices
    test_x: torch.Tensor
    test_y: torch.Tensor

    @property
    def input_shape(self) -> tuple[int, ...]:
        """One sample's shape, e.g. (3, 32, 32)."""
        return tuple(self.train_x.shape[1:])


def images_to_unit_float(images: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, C) or (N, H, W) images -> float32 (N, C, H, W) in [0, 1]."""
    if images.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {images.dtype}")
    x = torch.from_numpy(np.ascontiguousarray(images))
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    if x.ndim != 4:
        raise ValueError(f"expected (N, H, W) or (N, H, W, C) images, got {tuple(images.shape)}")
    return x.permute(0, 3, 1, 2).contiguous().float().div_(255)


def _cifar10(root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=root, train=train, download=True)
    return np.asarray(ds.data), np.asarray(ds.targets)


def _covtype(root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    from experiments.covtype import load_split

    split = load_split(root)
    return (split.train_x, split.train_y) if train else (split.test_x, split.test_y)


def features_to_float(features: np.ndarray) -> torch.Tensor:
    """float32 (N, F) features -> a float32 tensor, values unchanged."""
    if features.dtype != np.float32 or features.ndim != 2:
        raise ValueError(f"expected float32 (N, F) features, got {features.dtype} {features.shape}")
    return torch.from_numpy(np.ascontiguousarray(features))


def to_float_tensor(inputs: np.ndarray) -> torch.Tensor:
    """uint8 images through `images_to_unit_float`, float32 features as they are."""
    if inputs.dtype == np.uint8:
        return images_to_unit_float(inputs)
    return features_to_float(inputs)


LOADERS = {"cifar10": _cifar10, "covtype": _covtype}


def load_tensors(dataset, device: str | torch.device = "cpu") -> TensorData:
    """Train and test tensors for a DatasetConfig, on `device`."""
    if dataset.name not in LOADERS:
        raise KeyError(f"unknown dataset '{dataset.name}'; known: {sorted(LOADERS)}")
    tensors = []
    for train in (True, False):
        images, labels = LOADERS[dataset.name](str(Path(dataset.root)), train)
        found = len(np.unique(labels))
        if train and found != dataset.num_classes:
            raise ValueError(
                f"{dataset.name} has {found} classes, config says {dataset.num_classes}"
            )
        if labels.min() < 0 or labels.max() >= dataset.num_classes:
            raise ValueError(f"{dataset.name}: labels outside 0..{dataset.num_classes - 1}")
        if len(images) != len(labels):
            raise ValueError(f"{dataset.name}: {len(images)} images, {len(labels)} labels")
        tensors.append(to_float_tensor(images).to(device))
        tensors.append(torch.as_tensor(labels, dtype=torch.long).to(device))
    return TensorData(*tensors)
