"""Quantity-skew partitioning: the quantity_only mode (MinSize-Dirichlet).

Client SIZES are skewed; which samples a client holds is label-blind. One
permutation of all N indices is cut into K consecutive blocks whose lengths
come from `draw_client_sizes`, so every client is a uniform random sample of
the pooled data and its label mix equals the global one only in expectation.
Labels are never read by the assignment - only their count, N.

This is a replication, not a thesis contribution. It reproduces FedArtML
0.1.34 `SplitAsFederatedData.dirichlet_method_quant_skew(method="minsize-dirichlet")`
(read in source) so the quantity-only condition matches the source paper's:

    p ~ Dir(beta * 1_K), p = p / sum(p)
    every p_k < f / N is replaced by f / N, then p = p / sum(p)
    client sizes = consecutive differences of int(N * cumsum(p))

with FedArtML's floor f = 3 * num_classes + 1 samples by default
(`default_quantity_floor`). FedArtML's other variant, method="dirichlet"
(redraw until min size >= 3C), is not implemented.

The floor is approximate
------------------------
Renormalising after the substitution shrinks the floored shares too: with m
floored clients a floored client ends at f / (N * S), S <= 1 + m * f / N, and
truncating the cumulative sum to integers gives every client at least
floor(N * p_k) samples. Realized sizes therefore satisfy only

    min client size >= floor(f * N / (N + m * f))        (`min_size_bound`)

- 30, not 31, on CIFAR-10 at K = 30 once any client is floored. That is
FedArtML's behaviour and is kept on purpose; nothing re-imposes f afterwards.

The work plan paraphrases the source paper's floor as MinRequiredSize / n;
FedArtML's code uses (3C + 1) / N. The code is what is replicated here.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np


def default_quantity_floor(num_classes: int) -> int:
    """FedArtML's floor in samples: `len(np.unique(labels)) * 3`, plus one."""
    return 3 * num_classes + 1


def min_size_bound(N: int, floor: int, n_floored: int) -> int:
    """Smallest client size the floor guarantees, given `n_floored` floored clients."""
    return (floor * N) // (N + n_floored * floor)


# Adapted code: draw_client_sizes, and the permutation + split in quantity_only
#   Source:  FedArtML 0.1.34, fedartml/fl_split_as_federated_data.py,
#            SplitAsFederatedData.dirichlet_method_quant_skew, method="minsize-dirichlet"
#            (l. 336-358) https://github.com/Sapienza-University-Rome/FedArtML
#            (PyPI 0.1.34; git commit unknown), whose permutation + Dir(beta) + cumsum split
#            comes from NIID-Bench partition.py, partition == "iid-diff-quantity"
#            https://github.com/Xtra-Computing/NIID-Bench  (git commit unknown)
#   Licence: MIT, Copyright (c) 2022 La-Sapienza-University-Rome   - LICENSES/FedArtML-MIT.txt
#            MIT, Copyright (c) 2021 Yiqun Diao, Qinbin Li;
#                 Copyright (c) 2020 International Business Machines - LICENSES/NIID-Bench-MIT.txt
#   Changes: explicit np.random.Generator instead of np.random.seed on the global state;
#            floor parameterised (FedArtML derives 3 * #classes + 1 from the labels; here it is
#            an argument, by default from the registry's num_classes); size draw split from the
#            index split so B5 can reuse the sizes; returns integer sizes; no pandas, no
#            per-client reseed-and-shuffle (clients come back sorted); reports the number of
#            floored clients; input validation.
def draw_client_sizes(
    N: int, K: int, beta: float, floor: int, rng: np.random.Generator, stats: dict | None = None
) -> np.ndarray:
    """K integer client sizes summing to N, MinSize-Dirichlet with a `floor`-sample floor.

    Adapted third-party code (FedArtML / NIID-Bench, MIT; see the header above),
    not a thesis contribution - B5 imports it rather than copying it.
    `stats["n_floored_clients"]` is set to the number of drawn shares below
    floor / N. Realized sizes are bounded below by `min_size_bound`, not by
    `floor` (see the module docstring).
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if floor < 1 or K * floor > N:
        raise ValueError(f"need 1 <= floor and K * floor <= N; got floor={floor}, K={K}, N={N}")
    proportions = rng.dirichlet(np.repeat(beta, K))
    proportions = proportions / proportions.sum()
    floor_share = floor / N
    # A list mixing Python-float floors with np.float64 draws, summed by the
    # built-in sum(), exactly as FedArtML does: since Python 3.12 sum() is
    # compensated over a leading run of exact floats, so a numpy sum can differ
    # in the last bit.
    shares = [floor_share if p < floor_share else p for p in proportions]
    total = sum(shares)
    shares = np.array([s / total for s in shares])
    cuts = (np.cumsum(shares) * N).astype(int)[:-1]
    sizes = np.diff(np.concatenate(([0], cuts, [N])))
    if sizes.min() < 0 or sizes.sum() != N:
        raise RuntimeError(f"invalid client sizes {sizes.tolist()} for N={N}")
    if stats is not None:
        stats["n_floored_clients"] = int((proportions < floor_share).sum())
    return sizes


def quantity_only(
    labels, K: int, beta: float, floor: int, rng: np.random.Generator, stats: dict | None = None
) -> list[np.ndarray]:
    """Label-blind split into MinSize-Dirichlet sizes. Returns sorted int32 arrays.

    The permutation is drawn before the sizes, from the same Generator, in
    FedArtML's order. Only len(labels) is used. Adapted code: see the header
    above `draw_client_sizes`.
    """
    N = len(labels)
    order = rng.permutation(N)
    sizes = draw_client_sizes(N, K, beta, floor, rng, stats)
    return [np.sort(chunk.astype(np.int32)) for chunk in np.split(order, np.cumsum(sizes)[:-1])]
