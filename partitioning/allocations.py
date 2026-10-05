
 
from __future__ import annotations
 
import numpy as np
 
from partitioning.metrics import hellinger
 
 
def largest_remainder(weights, total: int) -> np.ndarray:
    
    weights = np.asarray(weights, dtype=float)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if (weights < 0).any() or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    quota = weights / weights.sum() * total
    counts = np.floor(quota).astype(np.int64)
    missing = int(total) - int(counts.sum())
    by_remainder = np.argsort(-(quota - counts), kind="stable")
    counts[by_remainder[:missing]] += 1
    return counts
 
 
def equal_budgets(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
   
    if not 1 <= K <= N:
        raise ValueError(f"need 1 <= K <= N, got K={K}, N={N}")
    budgets = np.full(K, N // K, dtype=np.int64)
    if N % K:
        budgets[rng.choice(K, size=N % K, replace=False)] += 1
    return budgets
 
 
def allocate_to_budgets(
    labels, budgets, proportions, rng: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray]:
    
    labels = np.asarray(labels)
    classes, stock = np.unique(labels, return_counts=True)
    stock = stock.astype(np.int64)
    budgets = np.asarray(budgets)
    proportions = np.asarray(proportions, dtype=float)
    K, C = len(budgets), len(classes)
 
    if not np.issubdtype(budgets.dtype, np.integer) or (budgets < 0).any():
        raise ValueError("budgets must be non-negative integers")
    if budgets.sum() != len(labels):
        raise ValueError(f"budgets sum to {budgets.sum()}, but there are {len(labels)} samples")
    if proportions.shape != (K, C):
        raise ValueError(f"proportions must have shape {(K, C)}, got {proportions.shape}")
    if (proportions < 0).any() or not np.isfinite(proportions).all():
        raise ValueError("proportions must be finite and non-negative")
 
    counts = np.zeros((K, C), dtype=np.int64)
    for k in rng.permutation(K):
        need = int(budgets[k])
        while need > 0:
            in_stock = stock > 0
            weights = np.where(in_stock, proportions[k], 0.0)
            if weights.sum() <= 0:
                
                weights = np.where(in_stock, stock, 0).astype(float)
            taken = np.minimum(largest_remainder(weights, need), stock)
            if taken.sum() == 0:
                raise RuntimeError(f"allocation stalled for client {k} with {need} samples to go")
            counts[k] += taken
            stock -= taken
            need -= int(taken.sum())
 
    if stock.any() or not np.array_equal(counts.sum(axis=1), budgets):
        raise RuntimeError("allocation did not use every sample exactly once")
 
    held = [[] for _ in range(K)]
    for j, c in enumerate(classes):
        idx_c = np.flatnonzero(labels == c)
        rng.shuffle(idx_c)
        start = 0
        for k in range(K):
            held[k].append(idx_c[start : start + counts[k, j]])
            start += counts[k, j]
    client_indices = [np.sort(np.concatenate(parts).astype(np.int32)) for parts in held]
    return client_indices, counts
 
 
def label_drift(proportions, counts) -> np.ndarray:
    
    proportions = np.asarray(proportions, dtype=float)
    counts = np.asarray(counts)
    sizes = counts.sum(axis=1)
    drift = np.full(len(counts), np.nan)
    for k in np.flatnonzero(sizes > 0):
        drift[k] = hellinger(proportions[k] / proportions[k].sum(), counts[k] / sizes[k])
    return drift
 
 
def pure_label(
    labels, K: int, alpha: float, rng: np.random.Generator, stats: dict | None = None
) -> list[np.ndarray]:
   
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    labels = np.asarray(labels)
    N = len(labels)
    classes, class_counts = np.unique(labels, return_counts=True)
 
    proportions = rng.dirichlet(alpha * len(classes) * class_counts / N, size=K)
    if not np.isfinite(proportions).all():
        raise RuntimeError(f"Dirichlet draw is not finite at alpha={alpha}")
    budgets = equal_budgets(N, K, rng)
    client_indices, counts = allocate_to_budgets(labels, budgets, proportions, rng)
 
    if stats is not None:
        stats["requested_proportions"] = proportions
        stats["realized_counts"] = counts
        stats["label_drift"] = label_drift(proportions, counts)
    return client_indices
 