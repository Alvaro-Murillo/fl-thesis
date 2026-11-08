"""Tests for the Covtype loader: split, standardisation, checksums.

The fast tests use synthetic arrays. The slow ones (--runslow) read the real data from
data/covtype/, fetched once with `python -m experiments.covtype`.
"""

import numpy as np
import pytest

from experiments import covtype
from experiments.config import DatasetConfig, ModelConfig
from experiments.covtype import (
    NUM_CLASSES,
    NUM_CONTINUOUS,
    TEST_SIZE,
    array_sha256,
    split_indices,
    standardise,
)

# Class counts of the 581,012 samples, labels 0..6 (UCI covtype.info).
CLASS_COUNTS = [211840, 283301, 35754, 2747, 9493, 17367, 20510]
TRAIN_COUNTS = [190656, 254970, 32179, 2472, 8544, 15630, 18459]
TEST_COUNTS = [21184, 28331, 3575, 275, 949, 1737, 2051]
SPLIT_SHA256 = {
    "train_x": "ddfd8054faabd987232e1464d530bc12b885061154dce8bb435f5b14292de4b5",
    "train_y": "9d3f78e2061636a0ee963d93d2dd5c91d06909b2502b0f8bdf4fc59142b529f7",
    "test_x": "47d7835f6bdb0c40291c5a8ebd4e74e16499f4116738aa0aa1302bd460bd8db7",
    "test_y": "da3e857833f858b3dd860f20e2c9f53b5e14bd8a6d4c3779185284b397ecc4b2",
}


def synthetic_labels():
    return np.repeat(np.arange(NUM_CLASSES), CLASS_COUNTS)[
        np.random.default_rng(0).permutation(sum(CLASS_COUNTS))
    ]


def test_the_split_is_stratified_disjoint_complete_and_in_file_order():
    labels = synthetic_labels()
    train, test = split_indices(labels)
    assert len(test) == TEST_SIZE and len(train) == len(labels) - TEST_SIZE == 522_910
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(labels)))
    assert (np.diff(train) > 0).all() and (np.diff(test) > 0).all()
    shares = np.array(CLASS_COUNTS) / len(labels)
    assert np.abs(np.bincount(labels[test]) - shares * TEST_SIZE).max() <= 1
    assert np.array_equal(np.bincount(labels[test]), TEST_COUNTS)  # same counts as the real split


def test_the_split_depends_only_on_the_labels():
    labels = synthetic_labels()
    first, again = split_indices(labels), split_indices(labels.copy())
    assert all(np.array_equal(a, b) for a, b in zip(first, again))


def test_standardisation_uses_the_training_set_and_leaves_the_indicators():
    rng = np.random.default_rng(1)
    train = np.hstack([rng.normal(50, 7, (1000, NUM_CONTINUOUS)), rng.integers(0, 2, (1000, 44))])
    test = np.hstack([rng.normal(80, 3, (200, NUM_CONTINUOUS)), rng.integers(0, 2, (200, 44))])
    train_x, test_x, mean, sd = standardise(train, test)
    assert train_x.dtype == test_x.dtype == np.float32
    assert np.allclose(train_x[:, :NUM_CONTINUOUS].mean(axis=0), 0, atol=1e-5)
    assert np.allclose(train_x[:, :NUM_CONTINUOUS].std(axis=0), 1, atol=1e-5)
    assert np.array_equal(train_x[:, NUM_CONTINUOUS:], train[:, NUM_CONTINUOUS:])
    assert np.array_equal(test_x[:, NUM_CONTINUOUS:], test[:, NUM_CONTINUOUS:])
    expected = (test[:, :NUM_CONTINUOUS] - train[:, :NUM_CONTINUOUS].mean(0)) / train[
        :, :NUM_CONTINUOUS
    ].std(0)
    assert np.array_equal(test_x[:, :NUM_CONTINUOUS], expected.astype(np.float32))
    # the test set never enters the statistics
    _, _, mean2, sd2 = standardise(train, test * 2)
    assert np.array_equal(mean, mean2) and np.array_equal(sd, sd2)


def test_a_constant_continuous_column_is_refused():
    train = np.zeros((10, 54))
    with pytest.raises(ValueError, match="constant"):
        standardise(train, train)


def test_fetched_arrays_must_match_their_recorded_sha256(monkeypatch):
    import sklearn.datasets

    X = np.zeros((covtype.NUM_SAMPLES, covtype.NUM_FEATURES))
    y = np.ones(covtype.NUM_SAMPLES, dtype=np.int32)
    monkeypatch.setattr(sklearn.datasets, "fetch_covtype", lambda **kw: (X, y))
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        covtype.fetch_raw("unused")
    monkeypatch.setitem(covtype.RAW_SHA256, "data", array_sha256(X))
    monkeypatch.setitem(covtype.RAW_SHA256, "target", array_sha256(y))
    assert covtype.fetch_raw("unused")[0] is not None


def test_a_local_archive_keeps_scikit_learns_checksum_and_is_put_back(tmp_path):
    from sklearn.datasets import _covtype

    original = _covtype.ARCHIVE
    with covtype._archive_from(tmp_path / "covtype.data.gz"):
        assert _covtype.ARCHIVE.url == (tmp_path / "covtype.data.gz").resolve().as_uri()
        assert _covtype.ARCHIVE.checksum == original.checksum
        assert _covtype.ARCHIVE.filename == original.filename
    assert _covtype.ARCHIVE is original


# --- the real data -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def split():
    return covtype.load_split("data")


@pytest.mark.slow
def test_real_split_sizes_counts_and_bytes(split):
    assert split.train_x.shape == (522_910, 54) and split.test_x.shape == (58_102, 54)
    assert split.train_x.dtype == np.float32 and split.train_y.dtype == np.int64
    assert np.bincount(split.train_y).tolist() == TRAIN_COUNTS
    assert np.bincount(split.test_y).tolist() == TEST_COUNTS
    for name, digest in SPLIT_SHA256.items():
        assert array_sha256(getattr(split, name)) == digest, name


@pytest.mark.slow
def test_real_features_are_standardised_on_the_training_set(split):
    continuous = split.train_x[:, :NUM_CONTINUOUS].astype(np.float64)
    assert np.allclose(continuous.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(continuous.std(axis=0), 1, atol=1e-4)
    assert np.isin(split.train_x[:, NUM_CONTINUOUS:], (0, 1)).all()


@pytest.mark.slow
def test_labels_and_tensors_come_from_the_same_split(split):
    from experiments.datasets import load_labels
    from experiments.tensors import load_tensors
    from models.networks import build_model

    dataset = DatasetConfig("covtype", NUM_CLASSES)
    assert np.array_equal(load_labels(dataset), split.train_y)
    tensors = load_tensors(dataset, "cpu")
    assert np.array_equal(tensors.train_x.numpy(), split.train_x)
    assert np.array_equal(tensors.test_y.numpy(), split.test_y)
    assert tensors.input_shape == (54,)
    model = build_model(ModelConfig("dnn", 54, NUM_CLASSES), tensors.input_shape, init_seed=0)
    assert model(tensors.test_x[:5]).shape == (5, NUM_CLASSES)
