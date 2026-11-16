"""Descriptive statistics for the paper's results: means, paired differences, RQ3 additivity.

Every summary is descriptive. Across trials: the mean and the sample standard deviation
(n - 1 in the denominator; empty for one value). A comparison of two conditions is paired by
trial: the per-trial differences, their mean and sd, and how many trials have each sign. No
test statistic is turned into a p-value, and no result is called significant.

RQ3 additivity, per non-edge cell (label level L > 0, quantity level Q > 0) and trial, against
that trial's (0, 0) corner:

    interaction = delta_combined - delta_label-only - delta_quantity-only
                = acc(L, Q) - acc(L, 0) - acc(0, Q) + acc(0, 0)

with delta_x = acc(x) - acc(0, 0). A negative interaction means the two skews together cost
more than the sum of their separate costs.

The RQ3 regression (supporting evidence) is ordinary least squares, per algorithm:

    acc ~ (L + L^2) x C(Q_level) + C(trial)

with L the run's realized label-HD: an intercept, the quantity-level dummies, L and L^2, their
products with the quantity-level dummies, and the trial dummies. Its model-implied interaction
at a cell compares the cell with the label-only cell at the same label-HD:

    [m(Lc, Q) - m(Lq, Q)] - [m(Lc, 0) - m(L0, 0)]

where m is the fitted mean with the trial effects averaged out, Lc the cell's mean realized
label-HD, Lq that of the quantity-only cell at Q (label level 0) and L0 the corner's. The
model-free interaction instead compares with the label-only cell at its own realized label-HD,
which is lower than the cell's because label-HD rises with quantity skew at a fixed alpha.

Lack of fit is measured against the cell-means model C(L_level) x C(Q_level) + C(trial). The
regression uses each run's own L, which varies within a cell, so the two models are not
nested; for this comparison only, the regression is refitted with each run's L replaced by its
cell's mean realized L, which nests it in the cell-means model, and the F statistic is

    F = [(RSS_regression - RSS_cell_means) / (p_cell_means - p_regression)]
        / [RSS_cell_means / (n - p_cell_means)]

reported with its degrees of freedom only.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mean_sd(values) -> tuple[float, float, int]:
    """(mean, sample sd, n); sd is NaN for fewer than two values, everything NaN for none."""
    v = np.asarray([x for x in values if x is not None and not pd.isna(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else np.nan, int(v.size)


@dataclass(frozen=True)
class Paired:
    """A paired-by-trial comparison a - b."""

    trials: tuple[int, ...]
    differences: tuple[float, ...]
    mean: float
    sd: float
    n: int
    n_positive: int
    n_negative: int
    n_zero: int


def paired(a: dict, b: dict) -> Paired:
    """a - b over the trials both hold; `a` and `b` map trial -> value."""
    trials = tuple(sorted(set(a) & set(b)))
    if set(a) != set(b):
        raise ValueError(f"unpaired trials: {sorted(set(a) ^ set(b))}")
    d = np.array([a[t] - b[t] for t in trials], dtype=float)
    mean, sd, n = mean_sd(d)
    return Paired(
        trials,
        tuple(float(x) for x in d),
        mean,
        sd,
        n,
        int((d > 0).sum()),
        int((d < 0).sum()),
        int((d == 0).sum()),
    )


def interactions(acc: dict, label_levels, quantity_levels) -> dict:
    """(L, Q) -> {trial: interaction} for every non-edge cell.

    `acc` maps (L, Q, trial) -> accuracy; the lowest level on each axis is the edge.
    """
    l0, q0 = min(label_levels), min(quantity_levels)
    trials = sorted({t for (_, _, t) in acc})
    out = {}
    for L in label_levels:
        for Q in quantity_levels:
            if L == l0 or Q == q0:
                continue
            out[(L, Q)] = {
                t: acc[(L, Q, t)] - acc[(L, q0, t)] - acc[(l0, Q, t)] + acc[(l0, q0, t)]
                for t in trials
            }
    return out


def deltas(acc: dict, L, Q, l0, q0) -> dict:
    """{'combined'|'label'|'quantity': {trial: delta}} for cell (L, Q) against the corner."""
    trials = sorted({t for (_, _, t) in acc})
    corner = {t: acc[(l0, q0, t)] for t in trials}
    return {
        "combined": {t: acc[(L, Q, t)] - corner[t] for t in trials},
        "label": {t: acc[(L, q0, t)] - corner[t] for t in trials},
        "quantity": {t: acc[(l0, Q, t)] - corner[t] for t in trials},
    }


# ---------------------------------------------------------------- regression
@dataclass
class Fit:
    names: list[str]
    coef: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rss: float
    n: int
    p: int


def ols(X: np.ndarray, y: np.ndarray, names: list[str]) -> Fit:
    """Least squares by numpy's SVD solver; refuses a rank-deficient design."""
    rank = np.linalg.matrix_rank(X)
    if rank != X.shape[1]:
        raise ValueError(f"design matrix has rank {rank} < {X.shape[1]} columns")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    residuals = y - fitted
    return Fit(names, coef, fitted, residuals, float(residuals @ residuals), len(y), X.shape[1])


def regression_design(runs: pd.DataFrame, L_column: str = "L", quadratic: bool = True):
    """Design for acc ~ (L [+ L^2]) x C(Q_level) + C(trial), reference levels the lowest.

    `runs` needs columns Q_level, trial and `L_column`.
    """
    q_levels = sorted(runs["Q_level"].unique())
    trials = sorted(runs["trial"].unique())
    L = runs[L_column].to_numpy(dtype=float)
    columns, names = [np.ones(len(runs))], ["intercept"]
    powers = (1, 2) if quadratic else (1,)
    for q in q_levels[1:]:
        columns.append((runs["Q_level"] == q).to_numpy(dtype=float))
        names.append(f"Q={q:g}")
    for k in powers:
        columns.append(L**k)
        names.append("L" if k == 1 else f"L^{k}")
        for q in q_levels[1:]:
            columns.append(L**k * (runs["Q_level"] == q).to_numpy(dtype=float))
            names.append(("L" if k == 1 else f"L^{k}") + f":Q={q:g}")
    for t in trials[1:]:
        columns.append((runs["trial"] == t).to_numpy(dtype=float))
        names.append(f"trial={t}")
    return np.column_stack(columns), names


def cell_means_design(runs: pd.DataFrame):
    """Design for acc ~ C(L_level) x C(Q_level) + C(trial): one mean per cell plus trials."""
    cells = sorted(set(zip(runs["L_level"], runs["Q_level"])))
    trials = sorted(runs["trial"].unique())
    columns, names = [], []
    for L, Q in cells:
        columns.append(((runs["L_level"] == L) & (runs["Q_level"] == Q)).to_numpy(dtype=float))
        names.append(f"cell L={L:g} Q={Q:g}")
    for t in trials[1:]:
        columns.append((runs["trial"] == t).to_numpy(dtype=float))
        names.append(f"trial={t}")
    return np.column_stack(columns), names


def predict(fit: Fit, L: float, Q: float, q_levels, trials, quadratic: bool = True) -> float:
    """Fitted mean at (L, Q) with the trial effects averaged over the trials."""
    row = {"intercept": 1.0}
    for q in q_levels[1:]:
        row[f"Q={q:g}"] = float(Q == q)
    for k in (1, 2) if quadratic else (1,):
        name = "L" if k == 1 else f"L^{k}"
        row[name] = L**k
        for q in q_levels[1:]:
            row[f"{name}:Q={q:g}"] = L**k * float(Q == q)
    for t in trials[1:]:
        row[f"trial={t}"] = 1.0 / len(trials)
    x = np.array([row.get(name, 0.0) for name in fit.names])
    return float(x @ fit.coef)


def model_interaction(fit: Fit, cell_L: dict, L, Q, q_levels, trials, quadratic=True) -> float:
    """[m(Lc, Q) - m(Lq, Q)] - [m(Lc, 0) - m(L0, 0)]; `cell_L` maps (L_level, Q_level) -> mean L."""
    l0, q0 = min(lv for lv, _ in cell_L), min(q_levels)
    Lc, Lq, L0 = cell_L[(L, Q)], cell_L[(l0, Q)], cell_L[(l0, q0)]

    def m(x, q):
        return predict(fit, x, q, q_levels, trials, quadratic)

    return (m(Lc, Q) - m(Lq, Q)) - (m(Lc, q0) - m(L0, q0))


def lack_of_fit(runs: pd.DataFrame, y_column: str = "acc") -> dict:
    """F of the regression on cell-mean L against the cell-means model, with its df."""
    cell_L = runs.groupby(["L_level", "Q_level"])["L"].transform("mean")
    nested = runs.assign(L_cell=cell_L)
    y = nested[y_column].to_numpy(dtype=float)
    X_reg, reg_names = regression_design(nested, "L_cell")
    X_cm, cm_names = cell_means_design(nested)
    reg, cm = ols(X_reg, y, reg_names), ols(X_cm, y, cm_names)
    df1, df2 = cm.p - reg.p, cm.n - cm.p
    F = ((reg.rss - cm.rss) / df1) / (cm.rss / df2)
    return {
        "F": float(F),
        "df_num": int(df1),
        "df_den": int(df2),
        "rss_regression_cell_L": reg.rss,
        "rss_cell_means": cm.rss,
        "p_regression": reg.p,
        "p_cell_means": cm.p,
    }


def fit_regression(runs: pd.DataFrame, y_column: str = "acc", quadratic: bool = True) -> Fit:
    X, names = regression_design(runs, "L", quadratic)
    return ols(X, runs[y_column].to_numpy(dtype=float), names)
