"""Skew metrics for a partition.

Label-HD is measured against the **empirical global label distribution**, never
against uniform: CIFAR-10 and FMNIST happen to be class-balanced, Covtype (RQ4)
is not. Quantity-HD is measured against uniform 1/K, which is correct by
definition - every client holding an equal share is the no-quantity-skew case.

The thesis metrics, recorded per partition:

    realized_label_hd              unweighted mean over NON-EMPTY clients of
                                   H(p_client, p_global)
    realized_quantity_hd           H(n/N, uniform 1/K) - one distance; empty
                                   clients count in it

An empty client has no label distribution, so it is excluded from label-HD and
counted in `n_empty_clients`.

`partition_metrics` also records two `*_pairwise` values on FedArtML's scale,
for translation to that literature only; they come from
`partitioning.fedartml_compat`, which holds the adapted FedArtML code.
`quantity_hd_from_pairwise` converts that quantity scale to ours exactly.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np

from partitioning.fedartml_compat import pairwise_hellinger


def hellinger(p, q) -> float:
    """Hellinger distance between two probability vectors, in [0, 1]."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2))


def quantity_hd_from_pairwise(hd_pairwise: float, K: int) -> float:
    """Convert FedArtML's quantity value to ours: sqrt(1 - sqrt(1 - (K-1) HD_lib^2)).

    Exact, not an approximation: with S = sum_k sqrt(n_k/N), the pairwise scalar
    on one-element vectors is HD_lib^2 = (K - S^2) / (K (K-1)), and ours is
    HD^2 = 1 - S / sqrt(K). (FedArtML's clamp at 1 never binds: HD_lib^2 <= 1/(K-1).)
    """
    inner = max(1.0 - (K - 1) * hd_pairwise ** 2, 0.0)
    return float(np.sqrt(max(1.0 - np.sqrt(inner), 0.0)))


def class_counts(labels, client_indices, num_classes) -> np.ndarray:
    """(K, num_classes) matrix of per-client class counts."""
    labels = np.asarray(labels)
    return np.stack([np.bincount(labels[idx], minlength=num_classes)
                     for idx in client_indices])


def partition_metrics(labels, client_indices, num_classes) -> dict:
    """Realized skew metrics for one partition, as recorded in partitions.jsonl."""
    labels = np.asarray(labels)
    counts = class_counts(labels, client_indices, num_classes)
    sizes = counts.sum(axis=1)
    nonempty = sizes > 0
    client_dists = counts[nonempty] / sizes[nonempty][:, None]
    global_dist = np.bincount(labels, minlength=num_classes) / len(labels)

    per_client_hd = np.array([hellinger(p, global_dist) for p in client_dists])
    size_shares = sizes / sizes.sum()
    uniform = np.full(len(sizes), 1 / len(sizes))

    return {
        "realized_label_hd": float(per_client_hd.mean()),
        # Undefined with fewer than two non-empty clients; None keeps the row valid JSON.
        "realized_label_hd_pairwise": (pairwise_hellinger(client_dists)
                                       if len(client_dists) >= 2 else None),
        "realized_quantity_hd": hellinger(size_shares, uniform),
        "realized_quantity_hd_pairwise": pairwise_hellinger(size_shares[:, None]),
        "mean_per_client_label_deviation": float(per_client_hd.mean()),
        "max_per_client_label_deviation": float(per_client_hd.max()),
        "min_client_size": int(sizes.min()),
        "max_client_size": int(sizes.max()),
        "mean_client_size": float(sizes.mean()),
        "n_empty_clients": int((~nonempty).sum()),
        "n_empty_class_slots": int((counts == 0).sum()),
    }
