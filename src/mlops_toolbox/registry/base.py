from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from mlops_toolbox.core.models import ModelContract, ModelInfo
from mlops_toolbox.core.protocols import ModelAdapter


class ModelRegistry(ABC):
    """Backends accept a `version` that is a number ("3"), "latest", or an alias
    like "production" pointing at a specific version."""

    @abstractmethod
    def register_model(
        self,
        model: ModelAdapter,
        name: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        contract: ModelContract | None = None,
        reference_data: pd.DataFrame | None = None,
    ) -> ModelInfo: ...

    @abstractmethod
    def get_model(self, name: str, version: str = "latest") -> ModelAdapter: ...

    @abstractmethod
    def get_model_info(self, name: str, version: str = "latest") -> ModelInfo: ...

    @abstractmethod
    def export_model(self, name: str, dst_dir: str | Path, version: str = "latest") -> Path: ...

    @abstractmethod
    def get_contract(self, name: str, version: str = "latest") -> ModelContract | None: ...

    @abstractmethod
    def get_reference_data(self, name: str, version: str = "latest") -> pd.DataFrame | None: ...

    @abstractmethod
    def get_reference_predictions(
        self, name: str, version: str = "latest"
    ) -> pd.DataFrame | None: ...

    @abstractmethod
    def set_alias(self, name: str, version: str, alias: str) -> ModelInfo: ...

    @abstractmethod
    def list_versions(self, name: str) -> list[ModelInfo]: ...
