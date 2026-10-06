"""Two-stage partitioning: the combined mode (quantity skew, then label skew).

Stage 1 draws the client sizes with `draw_client_sizes`, as quantity_only
does. Stage 2 draws the label mixes p_k ~ Dir(alpha * C * Q_global), as
pure_label does, and `fit_client_mixes` fits them to the stage-1 sizes and the
class stock. `round_to_margins` then rounds the fitted counts.

Authorship
----------
Written by the author. Changed with Claude Code (Anthropic) on 2026-10-06, at
the author's request:

- `fit_client_mixes`: `target` was used in the dual objective and its
  gradient but never defined, so every call raised NameError. It is now
  defined as stock / budgets.max(), the class stock in the same units as
  `weight`.
- Formatted with black by the repository's format-on-edit hook: trailing
  whitespace removed, two blank lines between functions.
- This module docstring and the marker in `fit_client_mixes` added.

AI assistance: the changes listed above were made with Claude Code (Anthropic) at the author's request; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from partitioning.allocations import label_drift, round_to_margins
from partitioning.quantity import draw_client_sizes

_LOG_UNDERFLOW = float(np.log(np.nextafter(0.0, 1.0)))


def fit_client_mixes(
    proportions, budgets, stock, tol: float = 1e-3, max_iter: int = 2000
) -> tuple[np.ndarray, int]:
    """Changed with Claude Code: `target` defined (see the module docstring)."""
    proportions = np.asarray(proportions, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    stock = np.asarray(stock, dtype=float)
    with np.errstate(divide="ignore"):
        log_p = np.where(proportions > 0, np.log(proportions), _LOG_UNDERFLOW)
    weight = budgets / budgets.max()
    target = stock / budgets.max()

    def evaluate(nu):
        tilted = log_p + weight[:, None] * nu[None, :]
        log_z = logsumexp(tilted, axis=1)
        mixes = np.exp(tilted - log_z[:, None])
        error = np.abs((budgets[:, None] * mixes).sum(axis=0) - stock).sum()
        return mixes, log_z.sum() - target @ nu, error

    nu = np.zeros(len(stock))
    mixes, value, error = evaluate(nu)
    damping = 1e-8
    for step in range(max_iter):
        if error <= tol:
            return budgets[:, None] * mixes, step
        weighted = weight[:, None] * mixes
        grad = (weighted.sum(axis=0) - target)[:-1]
        hess = np.diag((weight[:, None] * weighted).sum(axis=0)) - weighted.T @ weighted
        hess = hess[:-1, :-1]
        scale = max(hess.diagonal().max(), 1e-6)
        while True:
            direction = np.linalg.solve(hess + damping * scale * np.eye(len(grad)), -grad)
            nu_new = nu + np.append(direction, 0.0)
            mixes_new, value_new, error_new = evaluate(nu_new)
            if value_new <= value + 1e-4 * (grad @ direction) or (
                value_new <= value + 1e-12 * abs(value) and error_new < error
            ):
                damping = max(damping / 10, 1e-12)
                break
            damping *= 10
            if damping > 1e30:
                raise RuntimeError(f"mix fitting stalled at column error {error:.3g}")
        nu, mixes, value, error = nu_new, mixes_new, value_new, error_new
    if error <= tol:
        return budgets[:, None] * mixes, max_iter
    raise RuntimeError(f"mix fitting did not converge in {max_iter} steps (error {error:.3g})")


def allocate_client_weighted(
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
    active = budgets > 0
    fitted, _ = fit_client_mixes(proportions[active], budgets[active], stock)
    counts[active] = round_to_margins(fitted, budgets[active], stock)
    if not (
        np.array_equal(counts.sum(axis=1), budgets) and np.array_equal(counts.sum(axis=0), stock)
    ):
        raise RuntimeError("allocation did not use every sample exactly once")

    client_indices = [np.empty(0, dtype=np.int32) for _ in range(K)]
    for j, c in enumerate(classes):
        shuffled = rng.permutation(np.flatnonzero(labels == c))
        ends = np.cumsum(counts[:, j])
        for k in range(K):
            taken = shuffled[ends[k] - counts[k, j] : ends[k]]
            client_indices[k] = np.concatenate([client_indices[k], taken.astype(np.int32)])
    return [np.sort(idx) for idx in client_indices], counts


def combined(
    labels,
    K: int,
    alpha: float,
    beta: float,
    floor: int,
    rng: np.random.Generator,
    stats: dict | None = None,
) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    labels = np.asarray(labels)
    N = len(labels)
    classes, class_counts = np.unique(labels, return_counts=True)
    size_rng, mix_rng, cut_rng = rng.spawn(3)

    budgets = draw_client_sizes(N, K, beta, floor, size_rng, stats)
    proportions = mix_rng.dirichlet(alpha * len(classes) * class_counts / N, size=K)
    if not np.isfinite(proportions).all():
        raise RuntimeError(f"Dirichlet draw is not finite at alpha={alpha}")
    client_indices, counts = allocate_client_weighted(labels, budgets, proportions, cut_rng)

    if not np.array_equal([len(idx) for idx in client_indices], budgets):
        raise RuntimeError("client sizes differ from the stage-1 draw")
    if stats is not None:
        stats["client_budgets"] = budgets
        stats["requested_proportions"] = proportions
        stats["realized_counts"] = counts
        stats["label_drift"] = label_drift(proportions, counts)
    return client_indices
