from __future__ import annotations

from typing import Any

import numpy as np

from mlops_toolbox.core.models import EvaluationReport
from mlops_toolbox.evaluation.base import Evaluator


class ClassificationEvaluator(Evaluator):
    task_type = "classification"

    def evaluate(
        self, y_true: Any, y_pred: Any, y_proba: Any | None = None
    ) -> EvaluationReport:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        if y_true_arr.shape[0] != y_pred_arr.shape[0]:
            raise ValueError("y_true and y_pred must have the same length")

        labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        cm = _confusion_matrix(y_true_arr, y_pred_arr, labels)

        total = int(cm.sum())
        accuracy = float(np.trace(cm) / total) if total else 0.0
        precision, recall, f1 = _precision_recall_f1(cm)

        metrics = {
            "accuracy": accuracy,
            "precision_macro": float(np.mean(precision)),
            "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1)),
        }
        return EvaluationReport(
            task_type="classification",
            metrics=metrics,
            n_samples=int(y_true_arr.shape[0]),
            confusion_matrix=cm.tolist(),
            extra={"labels": labels.tolist()},
        )


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for true_val, pred_val in zip(y_true, y_pred, strict=True):
        cm[label_to_idx[true_val], label_to_idx[pred_val]] += 1
    return cm


def _precision_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return precision, recall, f1
