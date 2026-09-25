"""B3 phase 2: independent-stream comparison of quantity_only against FedArtML.

    python -m analysis.b3_fedartml_comparison [--seeds 10] [--betas 1000 10 1 0.3 0.1 0.03]

Real CIFAR-10 labels, K = 30, two variants per (beta, seed):

    fedartml  FedArtML 0.1.34 SplitAsFederatedData.dirichlet_method_quant_skew,
              method="minsize-dirichlet", random_state=seed, inside
              `preserved_global_rng()` - it reseeds the global np.random
    local     partitioning.quantity.quantity_only, np.random.default_rng(seed),
              floor default_quantity_floor(10) = 31 (FedArtML derives the same
              3 * 10 + 1 from the labels)

The two run the same algorithm on different RNG streams (tests/test_quantity.py
shows they coincide exactly on a shared stream), so they should agree in
distribution, not byte-for-byte. The summary gives, per beta and metric,
z = (mean_fedartml - mean_local) / SE with SE = sqrt(sd_f^2/n + sd_l^2/n).

FedArtML does not report how many clients it floored; that count is recovered
by replaying its seed -> permutation -> dirichlet draws on the global RNG
(saved and restored).

No training. Per-seed rows go to results/analysis/b3_fedartml_comparison.csv.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.b2_fedartml_comparison import comparison_seeds, preserved_global_rng
from experiments.config import DatasetConfig
from experiments.datasets import load_labels
from experiments.recording import atomic_write_text, git_provenance, utc_stamp
from partitioning.metrics import max_quantity_hd, partition_metrics
from partitioning.quantity import default_quantity_floor, min_size_bound, quantity_only

K = 30
BETAS = (1000.0, 10.0, 1.0, 0.3, 0.1, 0.03)
N_SEEDS = 10
OUT = Path("results/analysis/b3_fedartml_comparison.csv")
VARIANTS = ("fedartml", "local")
COMPARED = (
    "realized_quantity_hd",
    "realized_quantity_hd_pairwise",
    "realized_label_hd",
    "min_client_size",
    "n_floored_clients",
)


def fedartml_partition(labels, K: int, beta: float, seed: int) -> list[np.ndarray]:
    from fedartml import SplitAsFederatedData

    with preserved_global_rng():
        _, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method_quant_skew(
            labels, K, alpha=beta, random_state=seed, method="minsize-dirichlet"
        )
    return [np.sort(np.asarray(idx, dtype=np.int64)) for idx in idx_distr]


def fedartml_floored(N: int, K: int, beta: float, seed: int, floor: int) -> int:
    """Clients FedArtML floored for this seed, by replaying its draws.

    Replays seed -> permutation -> dirichlet on the global RNG; the replay is
    checked against FedArtML's own client sizes in tests/test_b3_fedartml_comparison.py.
    """
    return int((_replayed_shares(N, K, beta, seed) < floor / N).sum())


def _replayed_shares(N: int, K: int, beta: float, seed: int) -> np.ndarray:
    with preserved_global_rng():
        np.random.seed(seed)
        np.random.permutation(N)
        p = np.random.dirichlet(np.repeat(beta, K))
    return p / p.sum()


def run(labels, betas=BETAS, seeds=None, K: int = K) -> pd.DataFrame:
    seeds = comparison_seeds(N_SEEDS) if seeds is None else seeds
    num_classes = len(np.unique(labels))
    floor = default_quantity_floor(num_classes)
    N = len(labels)
    rows = []
    for beta in betas:
        for variant in VARIANTS:
            for trial, seed in enumerate(seeds):
                if variant == "fedartml":
                    parts = fedartml_partition(labels, K, beta, seed)
                    n_floored = fedartml_floored(N, K, beta, seed, floor)
                else:
                    stats = {}
                    parts = quantity_only(
                        labels, K, beta, floor, np.random.default_rng(seed), stats=stats
                    )
                    n_floored = stats["n_floored_clients"]
                m = partition_metrics(labels, parts, num_classes)
                rows.append(
                    {
                        "beta": beta,
                        "variant": variant,
                        "trial": trial,
                        "seed": seed,
                        "K": K,
                        "quantity_floor": floor,
                        "n_floored_clients": n_floored,
                        "min_size_bound": min_size_bound(N, floor, n_floored),
                        "quantity_hd_ceiling": max_quantity_hd(N, K, m["min_client_size"]),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def z_scores(df: pd.DataFrame, columns=COMPARED) -> pd.DataFrame:
    """Per beta and metric: both means, the two-sample SE and z = gap / SE."""
    out = []
    for beta, group in df.groupby("beta", sort=False):
        fed = group[group.variant == "fedartml"]
        loc = group[group.variant == "local"]
        for column in columns:
            f, l = fed[column].astype(float), loc[column].astype(float)
            se = float(np.sqrt(f.var() / len(f) + l.var() / len(l)))
            gap = float(f.mean() - l.mean())
            out.append(
                {
                    "beta": beta,
                    "metric": column,
                    "fedartml": f.mean(),
                    "local": l.mean(),
                    "gap": gap,
                    "se": se,
                    "z": gap / se if se > 0 else (0.0 if gap == 0 else np.inf),
                    "n": len(f),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    labels = load_labels(DatasetConfig(name="cifar10", num_classes=10))
    df = run(labels, args.betas, comparison_seeds(args.seeds))
    provenance = git_provenance()
    df["git_commit"], df["git_dirty"], df["generated_at"] = (
        provenance["git_commit"],
        provenance["git_dirty"],
        utc_stamp(),
    )
    atomic_write_text(args.out, df.to_csv(index=False, lineterminator="\n"))

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(z_scores(df).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(df)} rows -> {args.out}  (seeds: {sorted(set(df.seed))})")


if __name__ == "__main__":
    main()
