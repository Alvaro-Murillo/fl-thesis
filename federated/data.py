"""Client training data held on the training device.

The whole training set lives on one device as two tensors, and each client is a
tensor of indices into it, taken from the partition store. A batch is a gather
by index, so no DataLoader workers are needed and every client reads the
same bytes the partition recorded.

Batch order comes only from the `torch.Generator` passed in, so the caller
decides which seed stream shuffles which client.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch


class FederatedData:
    """Training set on one device, split into clients by index arrays."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, client_indices: Sequence[np.ndarray]):
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} samples, y has {y.shape[0]}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D class indices, got shape {tuple(y.shape)}")
        if x.device != y.device:
            raise ValueError(f"x is on {x.device}, y on {y.device}")
        n = x.shape[0]
        arrays = [np.asarray(idx) for idx in client_indices]
        for k, idx in enumerate(arrays):
            if idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
                raise ValueError(f"client {k}: indices must be a 1-D integer array")
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"client {k}: index out of range for {n} samples")
        flat = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        if np.unique(flat).size != flat.size:
            raise ValueError("a sample is assigned to more than one client")

        self.x = x
        self.y = y.long()
        self.indices = [torch.as_tensor(idx, dtype=torch.long, device=x.device) for idx in arrays]
        self.sizes = [int(idx.size) for idx in arrays]

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def num_clients(self) -> int:
        return len(self.indices)

    def size(self, client: int) -> int:
        return self.sizes[client]

    def batches(
        self, client: int, batch_size: int, generator: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """One epoch over a client's samples in shuffled order, last partial batch kept."""
        idx = self.indices[client]
        order = torch.randperm(len(idx), generator=generator).to(idx.device)
        shuffled = idx[order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            yield self.x[batch], self.y[batch]
