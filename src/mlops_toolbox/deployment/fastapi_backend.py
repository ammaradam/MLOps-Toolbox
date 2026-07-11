from typing import Any

import numpy as np
import pandas as pd
import pydantic

from mlops_toolbox._utils.optional_deps import import_optional_dependency
from mlops_toolbox.contracts.schema import row_model_from_contract
from mlops_toolbox.core.models import ModelContract
from mlops_toolbox.core.protocols import ModelAdapter
from mlops_toolbox.deployment.base import ModelServer


class _DefaultPredictRequest(pydantic.BaseModel):
    instances: list[list[float]]


class FastAPIModelServer(ModelServer):
    """ModelServer that exposes a ModelAdapter behind a minimal FastAPI app.

    NOTE: this module intentionally avoids `from __future__ import annotations`
    — build_app() defines an inner route handler whose request-body annotation
    is a locally-scoped class, and FastAPI resolves annotations directly from
    `__annotations__` rather than via module globals, which only works with
    real (non-string) annotation objects.
    """

    def build_app(
        self,
        model: ModelAdapter,
        input_schema: type[pydantic.BaseModel] | None = None,
        contract: ModelContract | None = None,
    ) -> Any:
        fastapi_module = import_optional_dependency("fastapi", extra="deploy")

        app = fastapi_module.FastAPI(title="mlops-toolbox model server")

        @app.get("/health")  # type: ignore[untyped-decorator]
        def health() -> dict[str, str]:
            return {"status": "ok"}

        if contract is not None:
            self._add_contract_routes(app, model, contract)
            return app

        schema = input_schema or _DefaultPredictRequest

        @app.post("/predict")  # type: ignore[untyped-decorator]
        def predict(request: schema) -> dict[str, list[Any]]:  # type: ignore[valid-type]
            payload = getattr(request, "instances", None)
            if payload is None:
                payload = list(request.model_dump().values())  # type: ignore[attr-defined]
            X = np.asarray(payload)
            predictions = model.predict(X)
            return {"predictions": predictions.tolist()}

        return app

    def _add_contract_routes(self, app: Any, model: ModelAdapter, contract: ModelContract) -> None:
        """Typed /predict (schema generated from the contract) plus GET /contract."""
        row_model = row_model_from_contract(contract)
        request_model = pydantic.create_model("PredictRequest", records=(list[row_model], ...))  # type: ignore[valid-type]
        column_order = [spec.name for spec in contract.input_columns]

        @app.get("/contract")  # type: ignore[untyped-decorator]
        def get_contract() -> dict[str, Any]:
            return contract.model_dump(mode="json")

        @app.post("/predict")  # type: ignore[untyped-decorator]
        def predict(request: request_model) -> dict[str, list[Any]]:  # type: ignore[valid-type]
            records = [record.model_dump() for record in request.records]  # type: ignore[attr-defined]
            X = pd.DataFrame.from_records(records)[column_order]
            predictions = model.predict(X)
            return {"predictions": predictions.tolist()}

    def serve(self, app: Any, host: str = "127.0.0.1", port: int = 8000) -> None:
        uvicorn = import_optional_dependency("uvicorn", extra="deploy")
        uvicorn.run(app, host=host, port=port)
