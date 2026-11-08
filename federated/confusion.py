"""The global model's confusion matrix on the test set, and the accuracies derived from it.

    accuracy            correct predictions over all test samples: the source paper's
                        metric (Jimenez-Gutierrez et al., arXiv:2503.17070 v2, Section 4.3),
                        and what `federated.loop.evaluate` returns
    per-class accuracy  the recall of each class: correct predictions over the test
                        samples of that class
    balanced accuracy   the unweighted mean of the per-class accuracies, over the classes
                        the test set holds. This is the thesis's addition, not the paper's.
                        On Covtype the classes hold 0.47% to 48.8% of the test set, so
                        accuracy mostly measures the two largest classes; balanced
                        accuracy weighs every class equally. It is scikit-learn's
                        `balanced_accuracy_score` with its default adjusted=False

The matrix comes from a forward pass in eval mode under no_grad, at `evaluate`'s batch
size, and is counted on the CPU. It changes no model state and draws no random number,
so a run that also records it trains exactly as one that does not.

torch is imported inside `confusion_matrix` only, so `experiments.recording`, which the
partition tools import, can derive the accuracies without importing torch.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

from __future__ import annotations

import numpy as np

EVAL_BATCH = 1000  # federated.loop.evaluate's default batch size


def confusion_matrix(model, x, y, num_classes: int) -> np.ndarray:
    """(num_classes, num_classes) counts; rows are the true class, columns the predicted one."""
    import torch

    model.eval()
    with torch.no_grad():
        preds = torch.cat(
            [model(x[s : s + EVAL_BATCH]).argmax(dim=1) for s in range(0, len(y), EVAL_BATCH)]
        )
    # Counted on the CPU: no CUDA op beyond the forward pass evaluate() already runs.
    pairs = y.cpu().numpy() * num_classes + preds.cpu().numpy()
    return np.bincount(pairs, minlength=num_classes**2).reshape(num_classes, num_classes)


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    """Recall of each class: correct predictions over the test samples of that class.

    NaN for a class with no test samples.
    """
    confusion = np.asarray(confusion)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(confusion) / confusion.sum(axis=1)


def balanced_accuracy(confusion: np.ndarray) -> float:
    """Mean per-class accuracy over the classes with test samples (the thesis's addition)."""
    confusion = np.asarray(confusion)
    present = confusion.sum(axis=1) > 0
    if not present.any():
        raise ValueError("the confusion matrix holds no samples")
    return float(per_class_accuracy(confusion)[present].mean())
