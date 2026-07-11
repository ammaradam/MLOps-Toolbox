from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from mlops_toolbox.core.models import DriftReport


class DriftDetector(ABC):
    @abstractmethod
    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> DriftReport: ...
