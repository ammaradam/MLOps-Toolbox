from __future__ import annotations

import numpy as np

from mlops_toolbox.evaluation import ClassificationEvaluator


def test_perfect_predictions_score_one() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    report = ClassificationEvaluator().evaluate(y_true, y_pred)
    assert report.task_type == "classification"
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["f1_macro"] == 1.0
    assert report.n_samples == 4


def test_confusion_matrix_shape_matches_labels() -> None:
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    report = ClassificationEvaluator().evaluate(y_true, y_pred)
    assert len(report.confusion_matrix) == 3
    assert all(len(row) == 3 for row in report.confusion_matrix)


def test_mismatched_lengths_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        ClassificationEvaluator().evaluate(np.array([0, 1]), np.array([0]))
