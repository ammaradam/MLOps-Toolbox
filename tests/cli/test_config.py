from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlops_toolbox.cli._app import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_config_shows_defaults_without_file() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert "(none found)" in result.output
    assert "sqlite:///mlops.db" in result.output
    assert "[default]" in result.output


def test_config_shows_file_and_env_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlops.toml").write_text(
        'tracking_uri = "https://mlflow.example.com"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MT_AWS__REGION", "us-east-1")

    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert str(tmp_path / "mlops.toml") in result.output
    assert "https://mlflow.example.com" in result.output
    assert "[file]" in result.output
    assert "us-east-1" in result.output
    assert "[env]" in result.output
