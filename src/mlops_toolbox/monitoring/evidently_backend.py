from __future__ import annotations

import pandas as pd

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.models import ColumnDriftResult, DriftReport
from mlops_toolbox.monitoring.base import DriftDetector


class EvidentlyDriftDetector(DriftDetector):
    """DriftDetector backed by Evidently's DataDriftPreset report.

    Runs fully offline against two in-memory dataframes, no account or
    external service required.
    """

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> DriftReport:
        if column_mapping:
            raise NotImplementedError(
                "column_mapping is not yet supported by the evidently backend (v1 relies on "
                "evidently's auto-inferred DataDefinition); rename columns before calling "
                "compare()."
            )

        evidently = import_optional_dependency("evidently", extra="monitoring")
        presets = import_optional_dependency("evidently.presets", extra="monitoring")

        report = evidently.Report(metrics=[presets.DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)
        result = snapshot.dict()

        dataset_metric = result["metrics"][0]
        drift_share = float(dataset_metric["value"]["share"])
        drift_share_threshold = float(dataset_metric["config"].get("drift_share", 0.5))
        dataset_drift_detected = drift_share >= drift_share_threshold

        column_results: dict[str, ColumnDriftResult] = {}
        for metric in result["metrics"][1:]:
            config = metric["config"]
            column = config.get("column")
            if column is None:
                continue
            p_value = float(metric["value"])
            threshold = float(config.get("threshold", 0.05))
            column_results[column] = ColumnDriftResult(
                column=column,
                drift_detected=p_value < threshold,
                drift_score=1.0 - p_value,
                stattest=str(config.get("method", "")),
            )

        return DriftReport(
            dataset_drift_detected=dataset_drift_detected,
            drift_share=drift_share,
            column_results=column_results,
        )
