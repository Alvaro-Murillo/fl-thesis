"""FedArtML's pairwise Hellinger scale, kept only to translate to that literature.

FedArtML reports one scalar per partition: the RMS of all pairwise Hellinger
distances between client distributions. `partitioning.metrics` records it next
to the thesis's own HD-to-global metrics as

    realized_label_hd_pairwise     over non-empty clients' label distributions
    realized_quantity_hd_pairwise  over one-element share vectors [[n_k/N]], as
                                   FedArtML's `create_clients()` computes it

These values are a translation table only; they are never grid axes. The code
below is adapted from FedArtML, not the thesis's own.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np


# Adapted from FedArtML 0.1.34, fedartml/function_base.py, hellinger_distance
#   https://github.com/Sapienza-University-Rome/FedArtML  (PyPI release 0.1.34; git commit unknown)
#   MIT License, Copyright (c) 2022 La-Sapienza-University-Rome - see LICENSES/FedArtML-MIT.txt
#   Changes: np.asarray input coercion; ValueError for fewer than 2 distributions; renamed locals.
#   Kept numerically identical on purpose (tests/test_hd.py::test_pairwise_matches_fedartml).
def pairwise_hellinger(distributions) -> float:
    """RMS of all pairwise Hellinger distances (matches FedArtML's scalar)."""
    d = np.asarray(distributions, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("pairwise Hellinger needs at least 2 distributions")
    sq = np.sqrt(d)
    squared = np.sum((sq[:, None, :] - sq[None, :, :]) ** 2, axis=2)
    return float(min(np.sqrt(squared.sum() / (2 * n * (n - 1))), 1.0))
