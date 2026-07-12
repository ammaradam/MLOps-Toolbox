from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.core.exceptions import RegistryError
from mlops_toolbox.core.models import ModelContract, ModelInfo
from mlops_toolbox.core.protocols import ModelAdapter
from mlops_toolbox.registry.base import ModelRegistry

_CONTRACT_ARTIFACT_DIR = "contract"
_CONTRACT_FILE = "contract.json"
_REFERENCE_FILE = "reference.parquet"
_REFERENCE_PREDICTIONS_FILE = "reference_predictions.parquet"
PREDICTION_COLUMN = "prediction"


class _LoadedPyfuncAdapter:
    """ModelAdapter wrapping a model loaded back from the MLflow registry."""

    framework = "mlflow_pyfunc"

    def __init__(self, pyfunc_model: Any) -> None:
        self._model = pyfunc_model

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray | None:
        return None


class MLflowModelRegistry(ModelRegistry):
    """ModelRegistry backed by MLflow's Model Registry (MLflow 3.x).

    Requires a database-backed tracking store (the file store does not support
    the registry), so this defaults to the same local SQLite URI used by
    MLflowTracker.

    Everywhere a `version` is accepted it may be a version number ("3"),
    "latest", or a registered-model alias like "production" — aliases are
    MLflow 3's replacement for the removed stage mechanism.

    `registry_uri` registers models on a different MLflow server than the
    tracking store — useful when tracking lands somewhere whose registry shim
    is limited (e.g. Azure ML's, which lacks aliases).
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlops.db",
        experiment_name: str = "Default",
        registry_uri: str | None = None,
    ) -> None:
        self._mlflow = import_optional_dependency("mlflow", extra="registry")
        self._mlflow.set_tracking_uri(tracking_uri)
        if registry_uri is not None:
            self._mlflow.set_registry_uri(registry_uri)
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        # Use the object-oriented MlflowClient (bound to this tracking_uri) rather than
        # mlflow's fluent API for experiment resolution: the fluent API tracks the active
        # experiment as module-level global state, which leaks across MLflowTracker /
        # MLflowModelRegistry instances pointed at different tracking URIs within the same
        # process. We resolve an experiment_id up front and pass it explicitly to every
        # start_run() call instead of relying on ambient fluent state.
        self._client = self._mlflow.tracking.MlflowClient(
            tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        self._default_experiment_id = self._get_or_create_experiment_id(experiment_name)

    def _get_or_create_experiment_id(self, name: str) -> str:
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return str(experiment.experiment_id)
        return str(self._client.create_experiment(name))

    def register_model(
        self,
        model: ModelAdapter,
        name: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        contract: ModelContract | None = None,
        reference_data: pd.DataFrame | None = None,
        max_reference_rows: int = 10_000,
    ) -> ModelInfo:
        active_run = self._mlflow.active_run()
        started_own_run = active_run is None
        if started_own_run:
            experiment_id = (
                self._client.get_run(run_id).info.experiment_id
                if run_id is not None
                else self._default_experiment_id
            )
            active_run = self._mlflow.start_run(run_id=run_id, experiment_id=experiment_id)

        try:
            self._mlflow.pyfunc.log_model(
                name="model",
                python_model=_adapter_to_pyfunc(self._mlflow, model),
                registered_model_name=name,
            )
            if contract is not None or reference_data is not None:
                self._log_contract_artifacts(model, contract, reference_data, max_reference_rows)
            source_run_id = active_run.info.run_id
        finally:
            if started_own_run:
                self._mlflow.end_run()

        versions = self._client.search_model_versions(f"name='{name}'")
        if not versions:
            raise RegistryError(f"Failed to register model '{name}'")
        latest = max(versions, key=lambda v: int(v.version))

        all_tags = dict(tags or {})
        if contract is not None:
            all_tags["mlops_toolbox.has_contract"] = "true"
        if reference_data is not None:
            all_tags["mlops_toolbox.has_reference"] = "true"
        if all_tags:
            for key, value in all_tags.items():
                self._client.set_model_version_tag(name, latest.version, key, value)
            latest = self._client.get_model_version(name, latest.version)

        return self._to_model_info(latest, source_run_id=source_run_id)

    def _log_contract_artifacts(
        self,
        model: ModelAdapter,
        contract: ModelContract | None,
        reference_data: pd.DataFrame | None,
        max_reference_rows: int,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            if contract is not None:
                contract_path = tmp_dir / _CONTRACT_FILE
                contract_path.write_text(contract.model_dump_json(indent=2), encoding="utf-8")
                self._mlflow.log_artifact(str(contract_path), artifact_path=_CONTRACT_ARTIFACT_DIR)
            if reference_data is not None:
                if len(reference_data) > max_reference_rows:
                    reference_data = reference_data.sample(n=max_reference_rows, random_state=0)
                reference_path = tmp_dir / _REFERENCE_FILE
                reference_data.to_parquet(reference_path, index=False)
                self._mlflow.log_artifact(str(reference_path), artifact_path=_CONTRACT_ARTIFACT_DIR)
                # Baseline predictions over the reference enable prediction-drift
                # checks later without re-running the (possibly retired) model.
                predictions = pd.DataFrame(
                    {PREDICTION_COLUMN: np.asarray(model.predict(reference_data)).ravel()}
                )
                predictions_path = tmp_dir / _REFERENCE_PREDICTIONS_FILE
                predictions.to_parquet(predictions_path, index=False)
                self._mlflow.log_artifact(
                    str(predictions_path), artifact_path=_CONTRACT_ARTIFACT_DIR
                )

    def get_model(self, name: str, version: str = "latest") -> ModelAdapter:
        model_uri = self._resolve_uri(name, version)
        pyfunc_model = self._mlflow.pyfunc.load_model(model_uri)
        return _LoadedPyfuncAdapter(pyfunc_model)

    def get_model_info(self, name: str, version: str = "latest") -> ModelInfo:
        return self._to_model_info(self._resolve_model_version(name, version))

    def export_model(self, name: str, dst_dir: str | Path, version: str = "latest") -> Path:
        """Download a version's model artifacts into dst_dir/model — a self-contained
        copy loadable with `mlflow.pyfunc.load_model()` and no tracking store."""
        artifacts_module = import_optional_dependency("mlflow.artifacts", extra="registry")
        mv = self._resolve_model_version(name, version)
        dst = Path(dst_dir) / "model"
        dst.mkdir(parents=True, exist_ok=True)
        local = artifacts_module.download_artifacts(
            artifact_uri=f"models:/{name}/{mv.version}",
            dst_path=str(dst),
            tracking_uri=self.tracking_uri,
        )
        return Path(local)

    def get_contract(self, name: str, version: str = "latest") -> ModelContract | None:
        with tempfile.TemporaryDirectory() as tmp:
            local = self._download_contract_artifact(name, version, _CONTRACT_FILE, tmp)
            if local is None:
                return None
            return ModelContract.model_validate_json(local.read_text(encoding="utf-8"))

    def get_reference_data(self, name: str, version: str = "latest") -> pd.DataFrame | None:
        with tempfile.TemporaryDirectory() as tmp:
            local = self._download_contract_artifact(name, version, _REFERENCE_FILE, tmp)
            if local is None:
                return None
            return pd.read_parquet(local)

    def get_reference_predictions(self, name: str, version: str = "latest") -> pd.DataFrame | None:
        with tempfile.TemporaryDirectory() as tmp:
            local = self._download_contract_artifact(
                name, version, _REFERENCE_PREDICTIONS_FILE, tmp
            )
            if local is None:
                return None
            return pd.read_parquet(local)

    def _download_contract_artifact(
        self, name: str, version: str, file_name: str, dst_dir: str
    ) -> Path | None:
        """Download one contract artifact into dst_dir, or None if it was never stored."""
        mv = self._resolve_model_version(name, version)
        if mv.run_id is None:
            return None
        try:
            local = self._client.download_artifacts(
                mv.run_id, f"{_CONTRACT_ARTIFACT_DIR}/{file_name}", dst_dir
            )
        except Exception:
            # MLflow raises store-specific errors (OSError, MlflowException, ...) when the
            # artifact path does not exist; absence of a contract is not an error here.
            return None
        return Path(local)

    def _resolve_model_version(self, name: str, version: str) -> Any:
        """Resolve "latest", a version number, or an alias to a ModelVersion."""
        if version == "latest":
            versions = self._client.search_model_versions(f"name='{name}'")
            if not versions:
                raise RegistryError(f"No versions found for model '{name}'")
            latest = max(versions, key=lambda v: int(v.version))
            # Re-fetch by version: search results omit aliases in some stores.
            return self._client.get_model_version(name, latest.version)
        if version.isdigit():
            return self._client.get_model_version(name, version)
        try:
            return self._client.get_model_version_by_alias(name, version)
        except Exception as exc:
            raise RegistryError(f"Model '{name}' has no alias {version!r}.") from exc

    def set_alias(self, name: str, version: str, alias: str) -> ModelInfo:
        """Point an alias (e.g. "production") at a version — MLflow 3's stage replacement."""
        try:
            self._client.set_registered_model_alias(name, alias, version)
        except Exception as exc:
            raise RegistryError(
                f"Failed to set alias {alias!r} on model '{name}' version {version}: {exc}. "
                "If this registry does not support model aliases (e.g. the Azure ML MLflow "
                "shim), point `registry_uri` at an alias-capable MLflow server instead."
            ) from exc
        return self._to_model_info(self._client.get_model_version(name, version))

    def list_versions(self, name: str) -> list[ModelInfo]:
        versions = self._client.search_model_versions(f"name='{name}'")
        return [self._to_model_info(v) for v in versions]

    def _resolve_uri(self, name: str, version: str) -> str:
        mv = self._resolve_model_version(name, version)
        return f"models:/{name}/{mv.version}"

    def _to_model_info(self, mv: Any, source_run_id: str | None = None) -> ModelInfo:
        return ModelInfo(
            name=mv.name,
            version=str(mv.version),
            backend="mlflow",
            aliases=sorted(mv.aliases) if mv.aliases else [],
            source_run_id=source_run_id or mv.run_id,
            uri=f"models:/{mv.name}/{mv.version}",
            tags=dict(mv.tags) if mv.tags else {},
        )


def _adapter_to_pyfunc(mlflow_module: Any, adapter: ModelAdapter) -> Any:
    class _AdapterPyfuncModel(mlflow_module.pyfunc.PythonModel):  # type: ignore[misc]
        # No type hints on predict(): MLflow attempts to infer a schema from them,
        # and warns because a non-list-wrapped hint doesn't fit its multi-instance
        # assumption. Omitting hints here avoids that warning entirely.
        def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
            return adapter.predict(model_input)

    return _AdapterPyfuncModel()
