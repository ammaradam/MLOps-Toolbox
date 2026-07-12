from __future__ import annotations

from typing import Annotated, Any

import typer

from mlops_toolbox.cli._options import ModelVersion, TrackingUri


def build_serving_app(name: str, version: str, tracking_uri: str | None) -> tuple[Any, bool]:
    """Load a registered model + its contract and return (app, has_contract)."""
    from mlops_toolbox.factories import model_registry, model_server

    registry = model_registry(tracking_uri)
    model = registry.get_model(name, version=version)
    contract = registry.get_contract(name, version=version)
    app = model_server().build_app(model, contract=contract)
    return app, contract is not None


def serve(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = None,
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Serve a registered model over HTTP, typed by its contract when one is stored."""
    from mlops_toolbox.factories import model_server

    app, has_contract = build_serving_app(name, version, tracking_uri)
    schema_note = "contract-typed" if has_contract else "untyped (no stored contract)"
    typer.echo(f"Serving '{name}' ({version}) at http://{host}:{port} — /predict is {schema_note}")
    model_server().serve(app, host=host, port=port)
