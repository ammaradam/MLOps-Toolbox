from __future__ import annotations

import typer


def config() -> None:
    """Show resolved settings and where each value came from (env, mlops.toml, or default)."""
    from mlops_toolbox.settings import explain_settings, find_config_file

    config_path = find_config_file()
    typer.echo(f"Config file: {config_path if config_path is not None else '(none found)'}")
    typer.echo("Resolution order: CLI flags > MT_* env vars > mlops.toml > defaults.")

    rows = explain_settings()
    key_width = max(len(row.key) for row in rows)
    for row in rows:
        rendered = "-" if row.value is None else str(row.value)
        typer.echo(f"  {row.key:<{key_width}}  {rendered:<40}  [{row.source}]")
