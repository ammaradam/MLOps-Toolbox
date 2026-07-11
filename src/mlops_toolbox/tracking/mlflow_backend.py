from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import TrackingError
from mlops_toolbox.core.models import RunResult
from mlops_toolbox.tracking.base import ExperimentTracker


class MLflowTracker(ExperimentTracker):
    """ExperimentTracker backed by MLflow.

    Defaults to a local SQLite tracking store (rather than the plain file
    store) so that tracking and the Model Registry both work fully offline
    with no external server or account.
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlops.db",
        experiment_name: str = "default",
    ) -> None:
        self._mlflow = import_optional_dependency("mlflow", extra="tracking")
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._active_run: Any | None = None
        self._run_result: RunResult | None = None

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        self._active_run = self._mlflow.start_run(run_name=run_name, tags=tags)
        self._run_result = RunResult(
            run_id=self._active_run.info.run_id,
            backend="mlflow",
            status="RUNNING",
        )
        return str(self._active_run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_active_run()
        self._mlflow.log_params(params)
        assert self._run_result is not None
        self._run_result.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._require_active_run()
        self._mlflow.log_metrics(metrics, step=step)
        assert self._run_result is not None
        self._run_result.metrics.update(metrics)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        self._require_active_run()
        self._mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def end_run(self, status: Literal["FINISHED", "FAILED"] = "FINISHED") -> RunResult:
        self._require_active_run()
        assert self._active_run is not None
        assert self._run_result is not None
        self._mlflow.end_run(status=status)
        result = self._run_result.model_copy(
            update={
                "status": status,
                "artifacts_uri": self._active_run.info.artifact_uri,
                "end_time": datetime.now(UTC),
            }
        )
        self._active_run = None
        self._run_result = None
        return result

    def _require_active_run(self) -> None:
        if self._active_run is None or self._run_result is None:
            raise TrackingError("No active run. Call start_run() first (or use `with tracker:`).")
