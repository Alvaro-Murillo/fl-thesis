"""B2 phase 3: shared-seed comparison of our partitioner against FedArtML.

    python -m analysis.b2_fedartml_comparison [--seeds 10] [--alphas 1000 6 1 0.3 0.03]

Real CIFAR-10 labels, K = 30, three variants per (alpha, seed):

    fedartml          FedArtML 0.1.34 SplitAsFederatedData.dirichlet_method,
                      random_state=seed, inside `preserved_global_rng()` - it
                      reseeds the global np.random internally and hardcodes a
                      min_size < 10 redraw with no attempt limit
    local_niid_bench  natural_per_class, size_cap="niid_bench", min_size=10
    local_uncapped    natural_per_class, size_cap="none",       min_size=10

fedartml and local_niid_bench run the same algorithm on different RNG streams
(tests/test_dirichlet.py shows they coincide exactly on a shared stream), so
they should agree in distribution, not byte-for-byte.

No training. Per-seed rows go to results/analysis/b2_fedartml_comparison.csv.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import DatasetConfig
from experiments.datasets import load_labels
from experiments.recording import atomic_write_text, git_provenance, utc_stamp
from experiments.seeding import derive_seeds
from partitioning.dirichlet import natural_per_class
from partitioning.metrics import partition_metrics

K = 30
ALPHAS = (1000.0, 6.0, 1.0, 0.3, 0.03)
N_SEEDS = 10
MIN_SIZE = 10          # FedArtML's hardcoded value, matched locally
MAX_REDRAWS = 5000     # ~1% of draws pass min_size at alpha=0.03, K=30 (observed)
OUT = Path("results/analysis/b2_fedartml_comparison.csv")
VARIANTS = ("fedartml", "local_niid_bench", "local_uncapped")


@contextmanager
def preserved_global_rng():
    """Save and restore global np.random state around a call that reseeds it."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def comparison_seeds(n: int) -> list[int]:
    """The trial-derived partition seeds, folded below 2**31.

    FedArtML adds 100 to its seed per class and per redraw, and np.random.seed
    rejects values >= 2**32; folding leaves ample headroom for that walk.
    """
    return [derive_seeds(trial).partition % 2 ** 31 for trial in range(n)]


def fedartml_partition(labels, K: int, alpha: float, seed: int) -> list[np.ndarray]:
    from fedartml import SplitAsFederatedData

    with preserved_global_rng():
        _, _, idx_distr, _ = SplitAsFederatedData.dirichlet_method(
            labels, K, alpha=alpha, random_state=seed)
    return [np.sort(np.asarray(idx, dtype=np.int64)) for idx in idx_distr]


def partition(variant: str, labels, K: int, alpha: float, seed: int):
    if variant == "fedartml":
        return fedartml_partition(labels, K, alpha, seed)
    size_cap = {"local_niid_bench": "niid_bench", "local_uncapped": "none"}[variant]
    return natural_per_class(labels, K, alpha, np.random.default_rng(seed),
                             min_size=MIN_SIZE, max_redraws=MAX_REDRAWS, size_cap=size_cap)


def run(labels, alphas=ALPHAS, seeds=None, K: int = K, variants=VARIANTS) -> pd.DataFrame:
    seeds = comparison_seeds(N_SEEDS) if seeds is None else seeds
    num_classes = len(np.unique(labels))
    rows = []
    for alpha in alphas:
        for variant in variants:
            for trial, seed in enumerate(seeds):
                start = time.perf_counter()
                parts = partition(variant, labels, K, alpha, seed)
                wall = time.perf_counter() - start
                m = partition_metrics(labels, parts, num_classes)
                rows.append({"alpha": alpha, "variant": variant, "trial": trial,
                             "seed": seed, "K": K, "min_size": MIN_SIZE,
                             "max_over_mean_size": m["max_client_size"] / m["mean_client_size"],
                             **m, "wall_s": round(wall, 3)})
    return pd.DataFrame(rows)


SUMMARY_COLUMNS = {
    "realized_label_hd": "label_hd",
    "realized_label_hd_pairwise": "label_hd_pw",
    "realized_quantity_hd": "qty_hd",
    "realized_quantity_hd_pairwise": "qty_hd_pw",
    "max_over_mean_size": "max/mean",
    "n_empty_clients": "n_empty",
}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- sd over seeds, per alpha x variant."""
    grouped = df.groupby(["alpha", "variant"], sort=False)
    out = pd.DataFrame(index=grouped.size().index)
    for column, label in SUMMARY_COLUMNS.items():
        mean, sd = grouped[column].mean(), grouped[column].std(ddof=1)
        out[label] = [f"{m:.4f} +/- {s:.4f}" if column != "max_over_mean_size"
                      else f"{m:.3f} +/- {s:.3f}" for m, s in zip(mean, sd)]
    out["n"] = grouped.size()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    labels = load_labels(DatasetConfig(name="cifar10", num_classes=10))
    df = run(labels, args.alphas, comparison_seeds(args.seeds))
    provenance = git_provenance()
    df["git_commit"], df["git_dirty"], df["generated_at"] = (
        provenance["git_commit"], provenance["git_dirty"], utc_stamp())
    atomic_write_text(args.out, df.to_csv(index=False, lineterminator="\n"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summarise(df).to_string())
    print(f"\n{len(df)} rows -> {args.out}  (seeds: {sorted(set(df.seed))})")


if __name__ == "__main__":
    main()
