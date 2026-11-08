"""Tests for the test-set confusion matrix and the accuracies derived from it."""

import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix as sklearn_confusion
from torch import nn

from federated.confusion import (
    EVAL_BATCH,
    balanced_accuracy,
    confusion_matrix,
    per_class_accuracy,
)
from federated.loop import evaluate


def test_confusion_matrix_matches_scikit_learn_and_evaluate():
    torch.manual_seed(0)
    model = nn.Linear(5, 4)
    x, y = torch.randn(2 * EVAL_BATCH + 17, 5), torch.randint(0, 4, (2 * EVAL_BATCH + 17,))
    cm = confusion_matrix(model, x, y, 4)
    preds = model(x).argmax(dim=1)
    assert np.array_equal(cm, sklearn_confusion(y.numpy(), preds.numpy(), labels=range(4)))
    assert np.trace(cm) / cm.sum() == evaluate(model, x, y)[0]


@pytest.mark.parametrize("seed", range(5))
def test_balanced_accuracy_is_scikit_learns(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.choice(7, size=500, p=[0.36, 0.49, 0.06, 0.005, 0.016, 0.03, 0.039])
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 7, 500))
    cm = sklearn_confusion(y_true, y_pred, labels=range(7))
    assert balanced_accuracy(cm) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred), abs=1e-12
    )


def test_a_class_with_no_test_samples_is_left_out_of_balanced_accuracy():
    cm = np.array([[3, 1, 0], [0, 0, 0], [0, 2, 2]])
    assert np.isnan(per_class_accuracy(cm)[1])
    assert balanced_accuracy(cm) == pytest.approx((0.75 + 0.5) / 2)
    with pytest.raises(ValueError, match="no samples"):
        balanced_accuracy(np.zeros((3, 3)))


def test_recording_imports_no_torch_through_it():
    """The partition tools import experiments.recording; it must not bring torch in."""
    import subprocess

    code = "import experiments.recording, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
