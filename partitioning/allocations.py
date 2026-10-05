"""Quantity-corrected label skew: the pure_label mode.

Every client holds the same number of samples (N // K, or one more for N % K
clients chosen at random), so the partition has no quantity skew by
construction. Only the label mixes are skewed:

    p_k ~ Dir(alpha * C * Q_global),  k = 1..K      (Q_global: empirical label distribution)

The prior's mean is Q_global, so the total demand for each class matches its
stock in expectation, on imbalanced data too. On balanced data the prior is
Dir(alpha * 1_C). Hsu et al. 2019 (arXiv:1909.06335) write Dir(alpha' * p);
their alpha' equals C * alpha here.

Within one draw the demand sum_k b_k p_k never matches the stock exactly, so
the requested counts b_k p_kc cannot all be met. The count matrix is the
closest one that can be:

    1. `sinkhorn_scale` gives X, the matrix with row sums = budgets and column
       sums = class stock that is closest in KL divergence to the requested
       counts. This is the Sinkhorn / iterative proportional fitting scaling.
       The correction is spread multiplicatively over every client.
    2. `round_to_margins` rounds every cell of X down or up, keeping both
       margins exact.

Each class's indices are then shuffled and cut into those counts. Only
realized metrics are recorded. `label_drift` measures how far each client's
realized mix is from its draw.

Authorship
----------
Written by the author. Changed with Claude Code (Anthropic) on 2026-10-05, at
the author's request:

- `allocate_to_budgets`: the count step is replaced. It used to fill clients
  one at a time, in random order. Each client took its largest-remainder share
  of the classes still in stock, falling back to stock proportions once its
  own classes ran out. That pushed every supply/demand mismatch onto the
  clients filled last. On CIFAR-10 at K = 10, alpha = 1, the first half of the
  clients drifted 0.004 (Hellinger) from their draws, and the last 10% drifted
  0.70. The counts now come from `sinkhorn_scale` followed by
  `round_to_margins`. Clients with a zero budget are skipped by the scaling.
  The input checks and the index assignment are unchanged, and the final check
  now compares the counts' margins with the budgets and the stock. The random
  client order (`rng.permutation(K)`) is gone, so a seed now gives a different
  partition than before.
- `largest_remainder` removed: its only caller was the old count step.
- `sinkhorn_scale` and `round_to_margins` added.
- Module and function docstrings added; formatted with black.
`equal_budgets`, `label_drift` and `pure_label` are unchanged apart from their
docstrings.

AI assistance: the changes listed above were made with Claude Code (Anthropic) at the author's request; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array
from scipy.special import logsumexp

from partitioning.metrics import hellinger

# An exact 0 from rng.dirichlet is an underflowed positive draw. It is scaled as
# the smallest positive double, never as a closed cell, so the margins stay reachable.
_LOG_UNDERFLOW = float(np.log(np.nextafter(0.0, 1.0)))
# Largest change to any log-scaling factor in one Newton step, in nats.
_MAX_STEP = 30.0


def equal_budgets(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """K integer client sizes summing to N: N // K each, plus 1 for N % K random clients."""
    if not 1 <= K <= N:
        raise ValueError(f"need 1 <= K <= N, got K={K}, N={N}")
    budgets = np.full(K, N // K, dtype=np.int64)
    if N % K:
        budgets[rng.choice(K, size=N % K, replace=False)] += 1
    return budgets


def sinkhorn_scale(
    weights, row_sums, col_sums, tol: float = 1e-3, max_iter: int = 1000
) -> tuple[np.ndarray, int]:
    """Scale `weights` (K x C) to the given row and column sums.

    Returns (X, n_steps). X = diag(a) W diag(b) is the Sinkhorn / iterative
    proportional fitting scaling of W (Deming & Stephan 1940; Sinkhorn 1964).
    Among all matrices with those margins, X is the closest to W in KL
    divergence (Csiszar 1975). X is unique because every cell is positive:
    exact zeros in W count as the smallest positive double. For rows of
    probabilities, such a zero is an underflowed draw. Because this floor does
    not scale with the row, the result is invariant to rescaling a row only
    when W has no zeros.

    The scaling is computed with Newton's method on the dual, the logs of a and
    b (Brauer et al. 2017, arXiv:1710.06635), not with Sinkhorn's alternating
    row and column updates. At alpha <= 0.01 the log-weights span hundreds of
    nats, and the alternating updates stall: on CIFAR-10 at K = 10,
    alpha = 0.01 they did not converge in 50 000 iterations. Newton took at
    most about 110 steps on every case tried. Where both converge, they agree
    to within the tolerance.

    `tol` is the total absolute margin error, in samples. Any value below 1
    lets `round_to_margins` keep both margins exact. Raises RuntimeError if
    `max_iter` steps do not reach it.

    Added with Claude Code (see the module docstring).
    """
    weights = np.asarray(weights, dtype=float)
    with np.errstate(divide="ignore"):
        log_w = np.where(weights > 0, np.log(weights), _LOG_UNDERFLOW)
    rows = np.asarray(row_sums, dtype=float)
    cols = np.asarray(col_sums, dtype=float)
    K = len(rows)

    def evaluate(u, v):
        with np.errstate(over="ignore", invalid="ignore"):
            X = np.exp(log_w + u[:, None] + v[None, :])
            dual = X.sum() - rows @ u - cols @ v
            error = np.abs(X.sum(axis=1) - rows).sum() + np.abs(X.sum(axis=0) - cols).sum()
        return X, dual, error

    # (u + t, v - t) gives the same X, so v[-1] stays at 0. Start from one row update.
    v = np.zeros(len(cols))
    u = np.log(rows) - logsumexp(log_w, axis=1)
    X, dual, error = evaluate(u, v)
    for step in range(max_iter):
        if error <= tol:
            return X, step
        row_mass, col_mass = X.sum(axis=1), X.sum(axis=0)[:-1]
        grad = np.concatenate([row_mass - rows, col_mass - cols[:-1]])
        hess = np.block([[np.diag(row_mass), X[:, :-1]], [X[:, :-1].T, np.diag(col_mass)]])
        # The ridge keeps directions with (numerically) no curvature solvable; along
        # them the step becomes a long gradient step, which the cap below limits.
        hess[np.diag_indices_from(hess)] += 1e-10 * hess.diagonal().max()
        d = np.linalg.solve(hess, -grad)
        d *= min(1.0, _MAX_STEP / np.abs(d).max())
        du, dv = d[:K], np.append(d[K:], 0.0)
        slope = grad @ d
        t = 1.0
        while True:
            X_new, dual_new, error_new = evaluate(u + t * du, v + t * dv)
            # Armijo on the dual. Near the optimum, changes in the dual fall below float
            # precision, so a step that keeps it level and lowers the margin error is
            # accepted too.
            if dual_new <= dual + 1e-4 * t * slope or (
                dual_new <= dual + 1e-12 * abs(dual) and error_new < error
            ):
                break
            t /= 2
            if t < 1e-12:
                raise RuntimeError(f"matrix scaling stalled at margin error {error:.3g}")
        u, v, X, dual, error = u + t * du, v + t * dv, X_new, dual_new, error_new
    if error <= tol:
        return X, max_iter
    raise RuntimeError(f"matrix scaling did not converge in {max_iter} steps (error {error:.3g})")


def round_to_margins(X, row_sums, col_sums) -> np.ndarray:
    """Round every cell of X down or up so the integer margins are met exactly.

    Of all such roundings it returns the one closest to X in L1. Which cells
    round up is a linear program over the transportation polytope: maximise
    the sum of the rounded-up cells' fractional parts. The constraint matrix
    is totally unimodular, so the optimal vertex is integral. X's margins must
    be within a total of 1 of the targets, which `sinkhorn_scale`'s tolerance
    guarantees.

    Added with Claude Code (see the module docstring).
    """
    X = np.asarray(X, dtype=float)
    rows = np.asarray(row_sums, dtype=np.int64)
    cols = np.asarray(col_sums, dtype=np.int64)
    K, C = X.shape
    base = np.floor(X).astype(np.int64)
    need = np.concatenate([rows - base.sum(axis=1), cols - base.sum(axis=0)])
    if not need.any():
        return base

    frac = (X - base).ravel()
    can_round_up = np.flatnonzero(frac > 0)
    row_of, col_of = np.divmod(can_round_up, C)
    n = len(can_round_up)
    incidence = coo_array(
        (np.ones(2 * n), (np.concatenate([row_of, K + col_of]), np.tile(np.arange(n), 2))),
        shape=(K + C, n),
    )
    result = linprog(
        -frac[can_round_up], A_eq=incidence, b_eq=need, bounds=(0, 1), method="highs-ds"
    )
    if result.status != 0:
        raise RuntimeError(f"no rounding of X meets both margins: {result.message}")
    up = np.rint(result.x)
    if np.abs(result.x - up).max() > 1e-6:
        raise RuntimeError("rounding LP returned a fractional vertex")

    counts = base.ravel()
    counts[can_round_up] += up.astype(np.int64)
    counts = counts.reshape(K, C)
    if not (np.array_equal(counts.sum(axis=1), rows) and np.array_equal(counts.sum(axis=0), cols)):
        raise RuntimeError("rounded counts do not meet the margins")
    return counts


def allocate_to_budgets(
    labels, budgets, proportions, rng: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray]:
    """Give client k exactly budgets[k] samples, with label mixes close to proportions[k].

    The count matrix is `proportions` scaled to the budgets and class stock
    (`sinkhorn_scale`), then rounded with both margins exact
    (`round_to_margins`). Each class's indices are shuffled and cut into the
    clients' counts. Returns (sorted int32 index arrays, (K, C) counts), with
    classes in np.unique order.

    Count step replaced with Claude Code; the input checks and the index
    assignment are the author's (see the module docstring).
    """
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
    target, _ = sinkhorn_scale(proportions[active], budgets[active], stock)
    counts[active] = round_to_margins(target, budgets[active], stock)

    if not (
        np.array_equal(counts.sum(axis=0), stock) and np.array_equal(counts.sum(axis=1), budgets)
    ):
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
    """Per-client Hellinger distance between the drawn mix and the realized mix.

    NaN for a client with no samples.
    """
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
    """Label skew with no quantity skew. Returns K sorted int32 index arrays.

    Draws p_k ~ Dir(alpha * C * Q_global), gives every client an equal budget
    (`equal_budgets`), and allocates with `allocate_to_budgets`. RNG order:
    the Dirichlet draw, then the budget remainder, then one shuffle per class.
    `stats`, if given, receives `requested_proportions` (K x C),
    `realized_counts` (K x C) and `label_drift` (K).
    """
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
