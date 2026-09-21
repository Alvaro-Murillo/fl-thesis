"""Dirichlet label partitioning (the `natural_per_class` mode).

Draws one Dirichlet vector per class and splits that class's samples across
clients accordingly - the standard construction from Lin et al. / NIID-Bench.

`size_cap` selects between two variants:

    "none"        (default) no balancing term. Client sizes spread freely, which
                  is precisely the quantity-skew byproduct RQ1 exists to measure.
    "niid_bench"  NIID-Bench's balancing term, replicated from FedArtML 0.1.34
                  `SplitAsFederatedData.dirichlet_method` (read in source):
                      p * (len(idx_j) < N / K), then renormalise
                  checked ONCE per class, before that class is split - so a
                  client under N/K can still overshoot it within one class.
                  That overshoot is FedArtML's behaviour and is kept on purpose.

Deliberately not replicated from FedArtML: its per-class reseeding of the global
np.random (seed += 100 per class) and its hardcoded min_size < 10 redraw. Here
randomness comes from an explicit Generator - the global np.random state is
never touched - and the minimum is `min_size`.
"""

from __future__ import annotations

import numpy as np

SIZE_CAPS = ("none", "niid_bench")


def natural_per_class(labels, K: int, alpha: float, rng: np.random.Generator,
                      min_size: int = 0, max_redraws: int = 100,
                      size_cap: str = "none") -> list[np.ndarray]:
    """Partition sample indices across K clients. Returns sorted int32 arrays."""
    if size_cap not in SIZE_CAPS:
        raise ValueError(f"size_cap must be one of {SIZE_CAPS}, got {size_cap!r}")
    labels = np.asarray(labels)
    classes = np.unique(labels)
    cap = len(labels) / K

    for _ in range(max_redraws):
        client_indices = [[] for _ in range(K)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(alpha, K))
            if size_cap == "niid_bench":
                proportions = proportions * np.array([len(j) < cap for j in client_indices])
                if proportions.sum() == 0:
                    # All uncapped clients drew exactly 0.0; FedArtML would divide
                    # by zero here and produce NaN cut points.
                    raise RuntimeError(
                        f"degenerate draw: every client under the N/K cap drew zero "
                        f"mass for class {c} (K={K}, alpha={alpha})")
                proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for client, chunk in zip(client_indices, np.split(idx_c, cuts)):
                client.extend(chunk.tolist())
        if min(len(c) for c in client_indices) >= min_size:
            return [np.sort(np.asarray(c, dtype=np.int32)) for c in client_indices]

    raise RuntimeError(
        f"no draw reached min_size={min_size} for K={K}, alpha={alpha} in "
        f"{max_redraws} attempts; lower min_size_policy or raise alpha")
