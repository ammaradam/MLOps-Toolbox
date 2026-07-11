"""mlops-toolbox: composable building blocks for the end-to-end ML lifecycle."""

from __future__ import annotations

import importlib
from typing import Any

from mlops_toolbox.adapters import SklearnAdapter, adapt
from mlops_toolbox.cards import generate_model_card
from mlops_toolbox.contracts import infer_contract, validate_against_contract
from mlops_toolbox.core.exceptions import (
    BackendNotInstalledError,
    MLOpsToolboxError,
    MonitoringError,
    ProjectAlreadyRegisteredError,
    ProjectError,
    ProjectNotFoundError,
    RegistryError,
    TrackingError,
    ValidationFailedError,
)
from mlops_toolbox.core.models import (
    ColumnDriftResult,
    ColumnSpec,
    DatasetSplit,
    DriftReport,
    EvaluationReport,
    GateCheck,
    GateResult,
    ModelContract,
    ModelInfo,
    OutputSpec,
    ProjectInfo,
    RunResult,
    ValidationResult,
)
from mlops_toolbox.core.protocols import ModelAdapter
from mlops_toolbox.data import (
    DataValidator,
    PydanticDataFrameValidator,
    train_test_split,
    validate_dataframe,
)
from mlops_toolbox.deployment.base import ModelServer
from mlops_toolbox.evaluation import (
    ClassificationEvaluator,
    Evaluator,
    RegressionEvaluator,
    check_gates,
)
from mlops_toolbox.factories import (
    DEFAULT_TRACKING_URI,
    drift_detector,
    model_registry,
    model_server,
    tracker,
)
from mlops_toolbox.monitoring import check_drift, check_prediction_drift
from mlops_toolbox.monitoring.base import DriftDetector
from mlops_toolbox.projects import get_project, list_projects, register_project, unregister_project
from mlops_toolbox.registry.base import ModelRegistry
from mlops_toolbox.scoring import score_dataframe
from mlops_toolbox.tracking.base import ExperimentTracker

__version__ = "0.1.0"

# Heavy backend classes are resolved lazily (PEP 562) so that `import
# mlops_toolbox` never forces an import of mlflow/evidently/fastapi. Each
# entry maps the public name to (submodule, attribute).
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    "MLflowTracker": ("mlops_toolbox.tracking.mlflow_backend", "MLflowTracker"),
    "MLflowModelRegistry": ("mlops_toolbox.registry.mlflow_backend", "MLflowModelRegistry"),
    "EvidentlyDriftDetector": (
        "mlops_toolbox.monitoring.evidently_backend",
        "EvidentlyDriftDetector",
    ),
    "FastAPIModelServer": ("mlops_toolbox.deployment.fastapi_backend", "FastAPIModelServer"),
    "run_dashboard": ("mlops_toolbox.dashboard.app", "run_dashboard"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_BACKENDS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    value = getattr(importlib.import_module(module_path), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "__version__",
    # backend-agnostic factories (the primary construction API)
    "tracker",
    "model_registry",
    "drift_detector",
    "model_server",
    "DEFAULT_TRACKING_URI",
    # core data contracts
    "RunResult",
    "ModelInfo",
    "ModelContract",
    "ColumnSpec",
    "OutputSpec",
    "EvaluationReport",
    "ColumnDriftResult",
    "DriftReport",
    "ValidationResult",
    "DatasetSplit",
    "ProjectInfo",
    # exceptions
    "MLOpsToolboxError",
    "BackendNotInstalledError",
    "ValidationFailedError",
    "TrackingError",
    "RegistryError",
    "MonitoringError",
    "ProjectError",
    "ProjectNotFoundError",
    "ProjectAlreadyRegisteredError",
    # adapters
    "ModelAdapter",
    "SklearnAdapter",
    "adapt",
    # contracts
    "infer_contract",
    "validate_against_contract",
    # data
    "train_test_split",
    "validate_dataframe",
    "DataValidator",
    "PydanticDataFrameValidator",
    # evaluation
    "Evaluator",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    "check_gates",
    "GateCheck",
    "GateResult",
    # tracking
    "ExperimentTracker",
    "MLflowTracker",
    # registry
    "ModelRegistry",
    "MLflowModelRegistry",
    # monitoring
    "DriftDetector",
    "EvidentlyDriftDetector",
    "check_drift",
    "check_prediction_drift",
    # scoring and cards
    "score_dataframe",
    "generate_model_card",
    # deployment
    "ModelServer",
    "FastAPIModelServer",
    # projects
    "register_project",
    "list_projects",
    "get_project",
    "unregister_project",
    # dashboard
    "run_dashboard",
]
