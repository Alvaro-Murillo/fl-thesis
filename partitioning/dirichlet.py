"""Dirichlet label partitioning (the `natural_per_class` mode).

Draws one Dirichlet vector per class and splits that class's samples across
clients accordingly - the standard construction from Lin et al. / NIID-Bench.

Deliberately WITHOUT NIID-Bench's balancing term (`p * (len(idx_j) < N/K)`,
which zeroes out any client already holding N/K samples). That term damps the
spread in client sizes, which is precisely the quantity-skew byproduct RQ1
exists to measure. FedArtML inherits the term, so its partitions are not
directly comparable on quantity-HD; B2 quantifies that gap.

Randomness comes from an explicit Generator - the global np.random state is
never touched.
"""

from __future__ import annotations

import numpy as np


def natural_per_class(labels, K: int, alpha: float, rng: np.random.Generator,
                      min_size: int = 0, max_redraws: int = 100) -> list[np.ndarray]:
    """Partition sample indices across K clients. Returns sorted int32 arrays."""
    labels = np.asarray(labels)
    classes = np.unique(labels)

    for _ in range(max_redraws):
        client_indices = [[] for _ in range(K)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, K))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client, chunk in zip(client_indices, np.split(idx_c, cuts)):
                client.extend(chunk.tolist())
        if min(len(c) for c in client_indices) >= min_size:
            return [np.sort(np.asarray(c, dtype=np.int32)) for c in client_indices]

    raise RuntimeError(
        f"no draw reached min_size={min_size} for K={K}, alpha={alpha} in "
        f"{max_redraws} attempts; lower min_size_policy or raise alpha")
