from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_toolbox.cli import templates
from mlops_toolbox.cli._options import DEFAULT_TRACKING_URI, ModelVersion, TrackingUri


def ship(
    name: Annotated[str, typer.Argument(help="Registered model name.")],
    version: ModelVersion = "latest",
    tracking_uri: TrackingUri = DEFAULT_TRACKING_URI,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output directory (default: deploy/<name>).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite a non-empty output directory.")
    ] = False,
) -> None:
    """Export a registered model as a self-contained, deployable service folder.

    The folder holds the model artifacts, its contract (when stored), a FastAPI
    serve.py, pinned requirements, and a Dockerfile — no tracking store needed
    at runtime.
    """
    from mlops_toolbox.factories import model_registry

    out_dir = out if out is not None else Path("deploy") / name
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        typer.echo(
            f"Output directory '{out_dir}' is not empty; pass --force to overwrite.", err=True
        )
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = model_registry(tracking_uri)
    info = registry.get_model_info(name, version=version)
    registry.export_model(name, out_dir, version=version)
    contract = registry.get_contract(name, version=version)
    if contract is not None:
        (out_dir / "contract.json").write_text(contract.model_dump_json(indent=2), encoding="utf-8")

    (out_dir / "serve.py").write_text(templates.SERVE_PY, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(templates.requirements_txt(), encoding="utf-8")
    (out_dir / "Dockerfile").write_text(templates.DOCKERFILE, encoding="utf-8")
    (out_dir / "README.md").write_text(
        templates.readme_md(name, info.version, contract is not None), encoding="utf-8"
    )

    contract_note = (
        "with contract" if contract is not None else "without contract (untyped /predict)"
    )
    typer.echo(f"Shipped '{name}' v{info.version} to {out_dir} ({contract_note}).")
    typer.echo(f"Try it:  cd {out_dir} && uvicorn serve:app --port 8000")
