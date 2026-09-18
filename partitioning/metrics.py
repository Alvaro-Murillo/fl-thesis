"""Skew metrics for a partition.

Label-HD is measured against the **empirical global label distribution**, never
against uniform: CIFAR-10 and FMNIST happen to be class-balanced, Covtype (RQ4)
is not. Quantity-HD is measured against uniform 1/K, which is correct by
definition - every client holding an equal share is the no-quantity-skew case.

Two label-HD scalars are recorded per partition:

    label_hd           mean over clients of H(p_client, p_global) - the thesis
                       definition, reference-based
    label_hd_pairwise  RMS of all pairwise H(p_i, p_j) - FedArtML's definition,
                       kept so values stay comparable with that literature
"""

from __future__ import annotations

import numpy as np


def hellinger(p, q) -> float:
    """Hellinger distance between two probability vectors, in [0, 1]."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2))


def pairwise_hellinger(distributions) -> float:
    """RMS of all pairwise Hellinger distances (matches FedArtML's scalar)."""
    d = np.asarray(distributions, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("pairwise Hellinger needs at least 2 distributions")
    sq = np.sqrt(d)
    squared = np.sum((sq[:, None, :] - sq[None, :, :]) ** 2, axis=2)
    return float(min(np.sqrt(squared.sum() / (2 * n * (n - 1))), 1.0))


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
    # An empty client has no label distribution; min_size_policy normally rules
    # this out, and the n_empty_class_slots / min_client_size columns expose it.
    client_dists = counts / np.maximum(sizes, 1)[:, None]
    global_dist = np.bincount(labels, minlength=num_classes) / len(labels)

    per_client_hd = np.array([hellinger(p, global_dist) for p in client_dists])
    size_shares = sizes / sizes.sum()
    uniform = np.full(len(sizes), 1 / len(sizes))

    return {
        "realized_label_hd": float(per_client_hd.mean()),
        "realized_label_hd_pairwise": pairwise_hellinger(client_dists),
        "realized_quantity_hd": hellinger(size_shares, uniform),
        "mean_per_client_label_deviation": float(per_client_hd.mean()),
        "max_per_client_label_deviation": float(per_client_hd.max()),
        "min_client_size": int(sizes.min()),
        "max_client_size": int(sizes.max()),
        "n_empty_class_slots": int((counts == 0).sum()),
    }
