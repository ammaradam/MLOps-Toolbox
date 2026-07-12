from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_toolbox.cli import templates
from mlops_toolbox.cli._options import AssumeYes, confirm_or_abort


def init(
    name: Annotated[str, typer.Argument(help="Project (and model) name.")],
    directory: Annotated[
        Path | None, typer.Option("--dir", help="Target directory (default: ./<name>).")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Scaffold into a non-empty directory.")
    ] = False,
    yes: AssumeYes = False,
) -> None:
    """Scaffold an ML project: contract-registered training plus a CI quality-gate workflow."""
    target = directory if directory is not None else Path(name)
    if target.exists() and any(target.iterdir()):
        if not force:
            typer.echo(
                f"Directory '{target}' is not empty; pass --force to scaffold anyway.", err=True
            )
            raise typer.Exit(code=1)
        confirm_or_abort(
            f"Scaffold into non-empty directory '{target}' (may overwrite files)?", yes
        )

    files = {
        "train.py": templates.INIT_TRAIN_PY,
        "mlops.toml": templates.INIT_MLOPS_TOML,
        "requirements.txt": templates.INIT_REQUIREMENTS,
        "README.md": templates.INIT_README,
        ".gitignore": templates.INIT_GITIGNORE,
        ".github/workflows/model-ci.yml": templates.INIT_CI_WORKFLOW,
    }
    for relative_path, template in files.items():
        path = target / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(templates.render_init_file(template, name), encoding="utf-8")

    typer.echo(f"Scaffolded '{name}' in {target}:")
    for relative_path in files:
        typer.echo(f"  {relative_path}")
    typer.echo(f"Next:  cd {target} && pip install -r requirements.txt && python train.py")
