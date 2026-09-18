"""Dataset label registry.

Partitioning only ever needs the label vector, so loaders return labels alone -
no images, no tensors, no GPU. `num_classes` is derived from the data and
checked against the config; nothing assumes 10 classes or a uniform prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _cifar10(root: str) -> np.ndarray:
    from torchvision.datasets import CIFAR10
    return np.asarray(CIFAR10(root=root, train=True, download=True).targets)


LOADERS = {"cifar10": _cifar10}


def load_labels(dataset) -> np.ndarray:
    """Training labels for a DatasetConfig, as an int array."""
    if dataset.name not in LOADERS:
        raise KeyError(f"unknown dataset '{dataset.name}'; known: {sorted(LOADERS)}")
    labels = LOADERS[dataset.name](str(Path(dataset.root)))
    found = len(np.unique(labels))
    if found != dataset.num_classes:
        raise ValueError(
            f"{dataset.name} has {found} classes, config says {dataset.num_classes}")
    return labels
