from __future__ import annotations

from typing import Any

import numpy as np

from mlops_toolbox.core.models import EvaluationReport
from mlops_toolbox.evaluation.base import Evaluator


class RegressionEvaluator(Evaluator):
    task_type = "regression"

    def evaluate(
        self, y_true: Any, y_pred: Any, y_proba: Any | None = None
    ) -> EvaluationReport:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")

        errors = y_true_arr - y_pred_arr
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        nonzero = y_true_arr != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / y_true_arr[nonzero])) * 100)
            if np.any(nonzero)
            else 0.0
        )

        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
        return EvaluationReport(
            task_type="regression",
            metrics=metrics,
            n_samples=int(y_true_arr.shape[0]),
            confusion_matrix=None,
            extra={},
        )
