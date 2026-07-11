from __future__ import annotations

from typer.testing import CliRunner

from mlops_toolbox.cli._app import app
from mlops_toolbox.projects import register_project

runner = CliRunner()


def test_doctor_passes_for_contracted_model(registered_model_store) -> None:
    uri, _ = registered_model_store
    result = runner.invoke(app, ["doctor", "--tracking-uri", uri])
    assert result.exit_code == 0, result.output
    assert "[PASS]" in result.output


def test_doctor_fails_with_hints_for_empty_store(tmp_path) -> None:
    result = runner.invoke(
        app, ["doctor", "--tracking-uri", f"sqlite:///{tmp_path / 'empty.db'}"]
    )
    assert result.exit_code == 1
    assert "[FAIL]" in result.output
    assert "hint:" in result.output


def test_doctor_resolves_project_name(registered_model_store, tmp_path) -> None:
    uri, _ = registered_model_store
    registry_path = tmp_path / "projects.json"
    register_project("doctor-project", uri, registry_path=registry_path)

    env = {"MLOPS_TOOLBOX_HOME": str(tmp_path)}
    result = runner.invoke(app, ["doctor", "doctor-project"], env=env)
    assert result.exit_code == 0, result.output


def test_doctor_unknown_project_exits_one(tmp_path) -> None:
    env = {"MLOPS_TOOLBOX_HOME": str(tmp_path)}
    result = runner.invoke(app, ["doctor", "no-such-project"], env=env)
    assert result.exit_code == 1
