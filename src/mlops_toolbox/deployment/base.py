from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pydantic

from mlops_toolbox.core.models import ModelContract
from mlops_toolbox.core.protocols import ModelAdapter


class ModelServer(ABC):
    @abstractmethod
    def build_app(
        self,
        model: ModelAdapter,
        input_schema: type[pydantic.BaseModel] | None = None,
        contract: ModelContract | None = None,
    ) -> Any: ...

    @abstractmethod
    def serve(self, app: Any, host: str = "127.0.0.1", port: int = 8000) -> None: ...
