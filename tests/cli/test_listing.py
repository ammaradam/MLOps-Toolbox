from __future__ import annotations

from typer.testing import CliRunner

from mlops_toolbox.cli._app import app
from mlops_toolbox.projects import register_project

runner = CliRunner()


def test_models_lists_contract_coverage(registered_model_store) -> None:
    uri, name = registered_model_store
    result = runner.invoke(app, ["models", "--tracking-uri", uri])
    assert result.exit_code == 0, result.output
    assert name in result.output
    assert "contract: yes" in result.output
    assert "drift baseline: yes" in result.output


def test_models_empty_store(tmp_path) -> None:
    result = runner.invoke(
        app, ["models", "--tracking-uri", f"sqlite:///{tmp_path / 'empty.db'}"]
    )
    assert result.exit_code == 0
    assert "No registered models" in result.output


def test_models_resolves_project_name(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    register_project("listing-project", uri, registry_path=tmp_path / "projects.json")
    env = {"MLOPS_TOOLBOX_HOME": str(tmp_path)}
    result = runner.invoke(app, ["models", "listing-project"], env=env)
    assert result.exit_code == 0, result.output
    assert name in result.output


def test_projects_lists_registered_projects(tmp_path) -> None:
    register_project(
        "proj-a",
        "sqlite:///a.db",
        description="first",
        tags={"team": "x"},
        registry_path=tmp_path / "projects.json",
    )
    env = {"MLOPS_TOOLBOX_HOME": str(tmp_path)}
    result = runner.invoke(app, ["projects"], env=env)
    assert result.exit_code == 0, result.output
    assert "proj-a" in result.output
    assert "sqlite:///a.db" in result.output
    assert "first" in result.output
    assert "[team=x]" in result.output


def test_projects_empty(tmp_path) -> None:
    env = {"MLOPS_TOOLBOX_HOME": str(tmp_path)}
    result = runner.invoke(app, ["projects"], env=env)
    assert result.exit_code == 0
    assert "No projects registered yet" in result.output
