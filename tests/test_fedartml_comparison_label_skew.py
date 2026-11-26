"""FedArtML and our capped label-skew partitioner agree in distribution.

Tolerance: |mean_fedartml - mean_local| <= 4 * SE, SE = sqrt(sd_f^2/n + sd_l^2/n)
over the n = 10 shared seeds. Both variants run the same algorithm on
independent RNG streams, so the gap is a two-sample difference of means with
~18 df; P(|t| > 4) ~ 1e-3 per alpha. Observed (seeds from comparison_seeds):
largest gap 2.5 SE at alpha = 1000 with 10 seeds, which shrank to 0.05 SE with
100 seeds - seed noise, not a systematic difference.
"""

import numpy as np
import pytest

from analysis.fedartml_comparison_label_skew import ALPHAS, comparison_seeds, run_comparison
from experiments.config import DatasetConfig
from experiments.datasets import load_labels

pytest.importorskip("fedartml")


@pytest.mark.slow
def test_fedartml_and_local_capped_agree_on_quantity_hd():
    labels = load_labels(DatasetConfig(name="cifar10", num_classes=10))
    df = run_comparison(
        labels, ALPHAS, comparison_seeds(10), variants=("fedartml", "local_niid_bench")
    )
    for alpha, group in df.groupby("alpha"):
        fed = group[group.variant == "fedartml"].realized_quantity_hd
        loc = group[group.variant == "local_niid_bench"].realized_quantity_hd
        se = np.sqrt(fed.var() / len(fed) + loc.var() / len(loc))
        gap = abs(fed.mean() - loc.mean())
        assert gap <= 4 * se, f"alpha={alpha}: gap {gap:.4f} > 4 SE ({4 * se:.4f})"


def test_global_rng_is_restored_around_fedartml():
    from analysis.fedartml_comparison_label_skew import fedartml_partition

    labels = np.repeat(np.arange(10), 100)
    np.random.seed(99)
    expected = np.random.random()
    np.random.seed(99)
    parts = fedartml_partition(labels, 5, 1.0, seed=3)
    assert np.random.random() == expected
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(1000))
