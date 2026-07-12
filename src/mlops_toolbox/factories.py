"""Backend-agnostic factories — the primary way to construct each lifecycle stage.

Code written against these factories (and the base interfaces they return)
never names the tool working behind the scenes, so the backend of any stage
can be replaced without touching user code. Concrete backend classes remain
importable for anyone who needs backend-specific behavior.

Where a URI is omitted, factories resolve it through settings
(`mlops.toml` / ``MT_*`` env vars — see :mod:`mlops_toolbox.settings`),
falling back to the same local zero-config defaults as always.
"""

from __future__ import annotations

from mlops_toolbox.deployment.base import ModelServer
from mlops_toolbox.monitoring.base import DriftDetector
from mlops_toolbox.registry.base import ModelRegistry
from mlops_toolbox.settings import load_settings
from mlops_toolbox.tracking.base import ExperimentTracker

DEFAULT_TRACKING_URI = "sqlite:///mlops.db"


def tracker(
    tracking_uri: str | None = None, experiment_name: str = "default"
) -> ExperimentTracker:
    """Experiment tracker for a local (or remote) tracking store."""
    from mlops_toolbox.tracking.mlflow_backend import MLflowTracker

    uri = tracking_uri if tracking_uri is not None else load_settings().tracking_uri
    return MLflowTracker(tracking_uri=uri, experiment_name=experiment_name)


def model_registry(
    tracking_uri: str | None = None, registry_uri: str | None = None
) -> ModelRegistry:
    """Model registry sharing the tracking store; stores contracts with models.

    `registry_uri` (or the `registry_uri` setting) registers models on a different
    MLflow server than tracking — e.g. tracking on Azure ML, registry on a shared
    MLflow server that supports aliases.
    """
    from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry

    settings = load_settings()
    uri = tracking_uri if tracking_uri is not None else settings.tracking_uri
    return MLflowModelRegistry(
        tracking_uri=uri,
        registry_uri=registry_uri if registry_uri is not None else settings.registry_uri,
    )


def drift_detector() -> DriftDetector:
    """Data drift detector comparing two dataframes offline."""
    from mlops_toolbox.monitoring.evidently_backend import EvidentlyDriftDetector

    return EvidentlyDriftDetector()


def model_server() -> ModelServer:
    """HTTP model server; contract-aware when a contract is supplied."""
    from mlops_toolbox.deployment.fastapi_backend import FastAPIModelServer

    return FastAPIModelServer()
