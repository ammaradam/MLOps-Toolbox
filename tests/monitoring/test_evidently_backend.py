from __future__ import annotations

import pytest

pytest.importorskip("evidently")

from mlops_toolbox.monitoring.evidently_backend import EvidentlyDriftDetector


def test_detects_drift_in_shifted_column(sample_dataframe_pair_with_drift) -> None:
    reference, current = sample_dataframe_pair_with_drift
    report = EvidentlyDriftDetector().compare(reference, current)

    assert "drifted_feature" in report.column_results
    assert "stable_feature" in report.column_results
    assert report.column_results["drifted_feature"].drift_detected is True
    assert 0.0 <= report.drift_share <= 1.0
