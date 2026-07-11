from __future__ import annotations

import numpy as np
import pytest

from mlops_toolbox.evaluation import RegressionEvaluator


def test_perfect_predictions_zero_error() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    report = RegressionEvaluator().evaluate(y_true, y_pred)
    assert report.metrics["mae"] == pytest.approx(0.0)
    assert report.metrics["rmse"] == pytest.approx(0.0)
    assert report.metrics["r2"] == pytest.approx(1.0)


def test_known_errors() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    report = RegressionEvaluator().evaluate(y_true, y_pred)
    assert report.metrics["mae"] == pytest.approx(0.5)
    assert report.n_samples == 4


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        RegressionEvaluator().evaluate(np.array([1.0, 2.0]), np.array([1.0]))
