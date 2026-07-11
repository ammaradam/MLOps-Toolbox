"""Dataframe file IO shared by CLI commands (parquet/csv, chosen by suffix)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported input format '{suffix}' — use .parquet or .csv.")


def write_frame(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise typer.BadParameter(f"Unsupported output format '{suffix}' — use .parquet or .csv.")
