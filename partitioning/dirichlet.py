"""Dirichlet label partitioning: the two "natural" modes of B2.

Both assign every sample and let client sizes emerge from the draw, so any
quantity skew they show is an uncontrolled byproduct - the RQ1 measurement.
Nothing here rebalances or post-processes client sizes.

natural_per_class
-----------------
Draws one Dirichlet vector per class and splits that class's samples across
clients accordingly - the standard construction from Lin et al. / NIID-Bench.

`size_cap` selects between two variants:

    "none"        (default) no balancing term. Client sizes spread freely.
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

natural_per_client
------------------
Written from scratch; FedArtML has no equivalent. Draws a K x C matrix of
per-client label mixes, pi_k ~ Dir(alpha * 1_C), then splits each class's
FULL stock across clients in proportion to that class's column,
pi_kc / sum_j pi_jc. Every sample is assigned; client sizes
n_k = sum_c n_c pi_kc / sum_j pi_jc emerge from the draw.

    Prior Dir(alpha * 1_C), not Dir(alpha * C * Q_global): the column
    normalisation already sends the alpha -> inf limit to Q_global on any
    dataset, and a flat prior gives every class the same concentration across
    clients, as natural_per_class's Dir(alpha * 1_K) does - so the two modes
    stay comparable at equal alpha on imbalanced data (Covtype, RQ4).

    A client's realized mix differs from its drawn pi_k (the column
    normalisation reweights it); only realized metrics are ever recorded.

The per-client construction in the literature (Hsu et al. 2019,
arXiv:1909.06335 sec. 3; Acar et al. 2021, FedDyn) fixes client budgets
instead, so it has no quantity byproduct by construction - that is B4's
territory, not this mode's.
"""

from __future__ import annotations

import numpy as np

SIZE_CAPS = ("none", "niid_bench")


def _split_by_proportions(idx, proportions, client_indices) -> None:
    cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
    for client, chunk in zip(client_indices, np.split(idx, cuts)):
        client.extend(chunk.tolist())


class MinSizeInfeasible(RuntimeError):
    """No draw in `max_redraws` attempts gave every client >= min_size samples.

    Raised, never swallowed: the minimum is not relaxed and the loop does not
    continue. Runners record the cell as explicitly infeasible.
    """

    def __init__(self, K, alpha, min_size, max_redraws):
        self.K, self.alpha, self.min_size, self.max_redraws = K, alpha, min_size, max_redraws
        super().__init__(
            f"no draw reached min_size={min_size} for K={K}, alpha={alpha} in "
            f"{max_redraws} attempts; lower min_size_policy or raise alpha")


def _redraw_until_min_size(draw, K, alpha, min_size, max_redraws,
                           stats: dict | None) -> list[np.ndarray]:
    """Call `draw()` until every client holds >= min_size samples.

    `draw` returns K index lists, or None for a draw rejected outright.
    `stats["n_redraws"]` is set to the number of rejected draws.
    """
    if max_redraws < 1:
        raise ValueError(f"max_redraws must be >= 1, got {max_redraws}")
    for attempt in range(max_redraws):
        client_indices = draw()
        if client_indices is not None and min(map(len, client_indices)) >= min_size:
            if stats is not None:
                stats["n_redraws"] = attempt
            return [np.sort(np.asarray(c, dtype=np.int32)) for c in client_indices]

    if stats is not None:
        stats["n_redraws"] = max_redraws
    raise MinSizeInfeasible(K, alpha, min_size, max_redraws)


def natural_per_class(labels, K: int, alpha: float, rng: np.random.Generator,
                      min_size: int = 0, max_redraws: int = 100,
                      size_cap: str = "none", stats: dict | None = None) -> list[np.ndarray]:
    """Partition sample indices across K clients. Returns sorted int32 arrays."""
    if size_cap not in SIZE_CAPS:
        raise ValueError(f"size_cap must be one of {SIZE_CAPS}, got {size_cap!r}")
    labels = np.asarray(labels)
    classes = np.unique(labels)
    cap = len(labels) / K

    def draw():
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
            _split_by_proportions(idx_c, proportions, client_indices)
        return client_indices

    return _redraw_until_min_size(draw, K, alpha, min_size, max_redraws, stats)


def natural_per_client(labels, K: int, alpha: float, rng: np.random.Generator,
                       min_size: int = 0, max_redraws: int = 100,
                       stats: dict | None = None) -> list[np.ndarray]:
    """Per-client Dirichlet mixes, column-normalised per class. Sorted int32 arrays."""
    labels = np.asarray(labels)
    classes = np.unique(labels)

    def draw():
        mixes = rng.dirichlet(np.repeat(alpha, len(classes)), size=K)  # (K, C)
        column_mass = mixes.sum(axis=0)
        if (column_mass == 0).any():
            # Every client drew exactly 0.0 for some class (float underflow at
            # tiny alpha; never observed in 20k draws at alpha=0.01, K=30):
            # that class cannot be split, so the draw is rejected and redrawn.
            return None
        client_indices = [[] for _ in range(K)]
        for j, c in enumerate(classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            _split_by_proportions(idx_c, mixes[:, j] / column_mass[j], client_indices)
        return client_indices

    return _redraw_until_min_size(draw, K, alpha, min_size, max_redraws, stats)
