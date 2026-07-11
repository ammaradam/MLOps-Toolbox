"""Backend-agnostic factories — the primary way to construct each lifecycle stage.

Code written against these factories (and the base interfaces they return)
never names the tool working behind the scenes, so the backend of any stage
can be replaced without touching user code. Concrete backend classes remain
importable for anyone who needs backend-specific behavior.
"""

from __future__ import annotations

from mlops_toolbox.deployment.base import ModelServer
from mlops_toolbox.monitoring.base import DriftDetector
from mlops_toolbox.registry.base import ModelRegistry
from mlops_toolbox.tracking.base import ExperimentTracker

DEFAULT_TRACKING_URI = "sqlite:///mlops.db"


def tracker(
    tracking_uri: str = DEFAULT_TRACKING_URI, experiment_name: str = "default"
) -> ExperimentTracker:
    """Experiment tracker for a local (or remote) tracking store."""
    from mlops_toolbox.tracking.mlflow_backend import MLflowTracker

    return MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)


def model_registry(tracking_uri: str = DEFAULT_TRACKING_URI) -> ModelRegistry:
    """Model registry sharing the tracking store; stores contracts with models."""
    from mlops_toolbox.registry.mlflow_backend import MLflowModelRegistry

    return MLflowModelRegistry(tracking_uri=tracking_uri)


def drift_detector() -> DriftDetector:
    """Data drift detector comparing two dataframes offline."""
    from mlops_toolbox.monitoring.evidently_backend import EvidentlyDriftDetector

    return EvidentlyDriftDetector()


def model_server() -> ModelServer:
    """HTTP model server; contract-aware when a contract is supplied."""
    from mlops_toolbox.deployment.fastapi_backend import FastAPIModelServer

    return FastAPIModelServer()
