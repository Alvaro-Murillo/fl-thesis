"""Covtype: fetch, split and standardise the UCI Covertype data.

    python -m experiments.covtype                     # fetch if missing, check, print counts
    python -m experiments.covtype --archive PATH      # first fetch from a local covtype.data.gz

Source: Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C50K5N. Licence: CC BY 4.0. 581,012 samples, each a 30 x 30 m
cell, with 54 features (10 continuous cartographic variables, then 4 wilderness-area
and 40 soil-type indicators, each group one-hot) and 7 forest cover types.

The arrays come from scikit-learn's `fetch_covtype` (scikit-learn 1.9.0, pinned in
requirements.txt), cached under `{root}/covtype/` (data/ is gitignored).
`fetch_covtype` downloads covtype.data.gz from scikit-learn's mirror, checks the
file's sha256 against the value scikit-learn pins, parses it and caches the arrays.
When the mirror refuses the download (figshare answered 403 on 2026-11-08), pass
the same file, from the UCI zip, with `--archive`: it is handed to `fetch_covtype`
in place of the mirror's URL, and scikit-learn checks it against the same sha256
before it parses it, so the cached arrays are the ones a mirror download gives.
Every load then checks the arrays against `RAW_SHA256`.

The source paper (Jimenez-Gutierrez et al., arXiv:2503.17070 v2, Section 4, Table 2)
gives only the sizes, 522,910 training and 58,102 test samples, 54 features and 7
unbalanced classes. It states no split and no preprocessing. The following are this
module's choices:

    labels      1..7 become class indices 0..6
    split       stratified on the label, test size exactly 58,102, drawn by
                scikit-learn's `train_test_split` with seed SPLIT_SEED; both sides keep
                the file's row order, so the training set is the file minus the test rows
    features    the 10 continuous columns are standardised with the training set's mean
                and standard deviation (population, ddof 0); the 44 binary columns are
                left as 0/1; float32

Row i of the training arrays is sample i of the label vector partitions index into
(`experiments.datasets`, `experiments.tensors`).

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_SAMPLES = 581_012
NUM_FEATURES = 54
NUM_CONTINUOUS = 10  # columns 0-9; 10-13 wilderness area, 14-53 soil type
NUM_CLASSES = 7
TEST_SIZE = 58_102  # the source paper's test set; the training set is then 522,910
SPLIT_SEED = 20260912  # the project's master seed, written out so the split never moves with it
SKLEARN_VERSION = "1.9.0"  # the version the arrays below were fetched and split with
# sha256 of the arrays fetch_covtype returns (data float64 (581012, 54), target int32 1..7)
RAW_SHA256 = {
    "data": "d9a3cc751bf9e6d3ab6ab547d84a7f78b89b7768f183c2ed4f55d8dc27f6c55f",
    "target": "e2a1e0212425a50ff29fc4d8f9be7e520411832e97eeeae99d16c301cfc8b1c3",
}


def array_sha256(array: np.ndarray) -> str:
    """sha256 of an array's bytes in C order."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@contextmanager
def _archive_from(path: Path):
    """Point fetch_covtype at a local covtype.data.gz, keeping scikit-learn's checksum."""
    from sklearn.datasets import _covtype

    original = _covtype.ARCHIVE
    _covtype.ARCHIVE = original._replace(url=Path(path).resolve().as_uri())
    try:
        yield
    finally:
        _covtype.ARCHIVE = original


def fetch_raw(root: str | Path = "data", archive: str | Path | None = None):
    """(data float64 (581012, 54), target int32 1..7) in file order, checked against RAW_SHA256."""
    from sklearn.datasets import fetch_covtype

    data_home = str(Path(root) / "covtype")
    if archive is None:
        X, y = fetch_covtype(data_home=data_home, shuffle=False, return_X_y=True)
    else:
        with _archive_from(archive):
            X, y = fetch_covtype(data_home=data_home, shuffle=False, return_X_y=True)
    X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)
    if X.shape != (NUM_SAMPLES, NUM_FEATURES) or y.shape != (NUM_SAMPLES,):
        raise ValueError(f"covtype: got data {X.shape} and target {y.shape}")
    for name, array in (("data", X), ("target", y)):
        if array_sha256(array) != RAW_SHA256[name]:
            raise ValueError(
                f"covtype {name} does not match its recorded sha256; "
                f"delete {data_home} and fetch it again"
            )
    return X, y


def split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) row indices, each in ascending order: stratified, TEST_SIZE test rows."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        np.arange(len(labels)), test_size=TEST_SIZE, stratify=labels, random_state=SPLIT_SEED
    )
    return np.sort(train), np.sort(test)


def standardise(train_x: np.ndarray, test_x: np.ndarray):
    """Both sets with the continuous columns scaled by the training set's mean and sd, float32.

    Returns (train, test, mean, sd); mean and sd are float64, one per continuous column.
    """
    continuous = train_x[:, :NUM_CONTINUOUS]
    mean, sd = continuous.mean(axis=0), continuous.std(axis=0)
    if (sd == 0).any():
        raise ValueError("a continuous covtype column is constant in the training set")
    scaled = []
    for x in (train_x, test_x):
        x = x.astype(np.float64, copy=True)
        x[:, :NUM_CONTINUOUS] = (x[:, :NUM_CONTINUOUS] - mean) / sd
        scaled.append(x.astype(np.float32))
    return scaled[0], scaled[1], mean, sd


@dataclass(frozen=True)
class CovtypeSplit:
    train_x: np.ndarray  # float32 (522910, 54)
    train_y: np.ndarray  # int64 class indices 0..6
    test_x: np.ndarray  # float32 (58102, 54)
    test_y: np.ndarray
    mean: np.ndarray  # the continuous columns' training mean, float64
    sd: np.ndarray  # and standard deviation (ddof 0)


def load_split(root: str | Path = "data", archive: str | Path | None = None) -> CovtypeSplit:
    """The training and test sets, split and standardised as the module docstring says."""
    X, y = fetch_raw(root, archive)
    binary = X[:, NUM_CONTINUOUS:]
    if not np.isin(binary, (0.0, 1.0)).all():
        raise ValueError("covtype: the 44 indicator columns are not all 0/1")
    if not (binary[:, :4].sum(axis=1) == 1).all() or not (binary[:, 4:].sum(axis=1) == 1).all():
        raise ValueError("covtype: a sample is not in exactly one wilderness area and soil type")
    if y.min() != 1 or y.max() != NUM_CLASSES:
        raise ValueError(f"covtype: labels outside 1..{NUM_CLASSES}")
    labels = y.astype(np.int64) - 1
    train, test = split_indices(labels)
    train_x, test_x, mean, sd = standardise(X[train], X[test])
    return CovtypeSplit(train_x, labels[train], test_x, labels[test], mean, sd)


def describe(split: CovtypeSplit) -> str:
    train = np.bincount(split.train_y, minlength=NUM_CLASSES)
    test = np.bincount(split.test_y, minlength=NUM_CLASSES)
    lines = [f"{'class':>5} {'label':>5} {'train':>8} {'share':>7} {'test':>7} {'share':>7}"]
    for c in range(NUM_CLASSES):
        lines.append(
            f"{c:>5} {c + 1:>5} {train[c]:>8} {train[c] / train.sum():>7.4f} "
            f"{test[c]:>7} {test[c] / test.sum():>7.4f}"
        )
    lines.append(f"{'all':>5} {'':>5} {train.sum():>8} {'':>7} {test.sum():>7}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default="data")
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="covtype.data.gz to fetch from instead of scikit-learn's mirror (first fetch only)",
    )
    args = parser.parse_args()

    import sklearn

    X, y = fetch_raw(args.root, args.archive)
    print(f"scikit-learn {sklearn.__version__} (arrays recorded with {SKLEARN_VERSION})")
    print(f"sha256 data   {array_sha256(X)}")
    print(f"sha256 target {array_sha256(y)}")
    split = load_split(args.root)
    for name in ("train_x", "train_y", "test_x", "test_y"):
        print(f"sha256 {name:<7} {array_sha256(getattr(split, name))}")
    print(describe(split))


if __name__ == "__main__":
    main()
