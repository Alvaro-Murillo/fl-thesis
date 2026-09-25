"""FedArtML's minsize-dirichlet and quantity_only agree in distribution.

Tolerance, as for the label-skew comparison: |mean_fedartml - mean_local| <= 4 * SE,
SE = sqrt(sd_f^2/n + sd_l^2/n) over n = 10 seeds per beta, on independent RNG
streams; P(|t| > 4) ~ 1e-3 per beta and metric at ~18 df.
"""

import numpy as np
import pytest

from analysis.fedartml_comparison_quantity_skew import BETAS, comparison_seeds, run, z_scores
from experiments.config import DatasetConfig
from experiments.datasets import load_labels

pytest.importorskip("fedartml")


@pytest.mark.slow
def test_fedartml_and_quantity_only_agree_on_real_cifar10():
    labels = load_labels(DatasetConfig(name="cifar10", num_classes=10))
    df = run(labels, BETAS, comparison_seeds(10))
    z = z_scores(df, ("realized_quantity_hd", "realized_quantity_hd_pairwise", "realized_label_hd"))
    bad = z[np.abs(z.gap) > 4 * z.se]
    assert bad.empty, bad.to_string()


@pytest.mark.parametrize("beta", [0.01, 1.0])
def test_floored_count_replay_matches_fedartml_sizes(beta):
    """The replayed draw, pushed through the floor, gives FedArtML's own client sizes."""
    from analysis.fedartml_comparison_quantity_skew import _replayed_shares, fedartml_partition

    labels, K, floor, seed = np.repeat(np.arange(10), 300), 30, 31, 12345
    N = len(labels)
    p = _replayed_shares(N, K, beta, seed)
    shares = np.where(p < floor / N, floor / N, p)
    shares = shares / shares.sum()
    cuts = (np.cumsum(shares) * N).astype(int)[:-1]
    expected = np.diff(np.concatenate(([0], cuts, [N])))
    sizes = [len(c) for c in fedartml_partition(labels, K, beta, seed)]
    assert np.array_equal(sizes, expected)
    if beta == 0.01:
        assert (p < floor / N).sum() > 0


def test_global_rng_is_restored_around_fedartml():
    from analysis.fedartml_comparison_quantity_skew import fedartml_floored, fedartml_partition

    labels = np.repeat(np.arange(10), 100)
    np.random.seed(99)
    expected = np.random.random()
    np.random.seed(99)
    parts = fedartml_partition(labels, 5, 0.1, seed=3)
    fedartml_floored(len(labels), 5, 0.1, seed=3, floor=31)
    assert np.random.random() == expected
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(1000))
