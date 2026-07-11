from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mlops_toolbox.core.models import EvaluationReport


class Evaluator(ABC):
    task_type: ClassVar[str]

    @abstractmethod
    def evaluate(
        self, y_true: Any, y_pred: Any, y_proba: Any | None = None
    ) -> EvaluationReport: ...
