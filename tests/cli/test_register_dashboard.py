from __future__ import annotations

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")

from mlops_toolbox.cli._app import app
from mlops_toolbox.projects import get_project, list_projects

runner = CliRunner()


def _env(tmp_path) -> dict[str, str]:
    return {"MLOPS_TOOLBOX_HOME": str(tmp_path)}


def test_register_creates_project_with_tags(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "register",
            "my-proj",
            "--tracking-uri",
            "sqlite:///store.db",
            "--description",
            "demo",
            "--tag",
            "team=growth",
            "--tag",
            "stage=poc",
        ],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output

    info = get_project("my-proj", registry_path=tmp_path / "projects.json")
    # Relative sqlite paths are normalized to absolute at registration time.
    assert info.tracking_uri.startswith("sqlite:///")
    assert info.tracking_uri.endswith("/store.db")
    assert info.description == "demo"
    assert info.tags == {"team": "growth", "stage": "poc"}


def test_register_duplicate_requires_overwrite(tmp_path) -> None:
    env = _env(tmp_path)
    first = runner.invoke(app, ["register", "dup", "--tracking-uri", "sqlite:///a.db"], env=env)
    assert first.exit_code == 0

    second = runner.invoke(app, ["register", "dup", "--tracking-uri", "sqlite:///b.db"], env=env)
    assert second.exit_code == 1
    assert "--overwrite" in second.output

    # --overwrite prompts for confirmation; declining aborts without changes.
    declined = runner.invoke(
        app,
        ["register", "dup", "--tracking-uri", "sqlite:///b.db", "--overwrite"],
        env=env,
        input="n\n",
    )
    assert declined.exit_code == 1
    assert get_project("dup", registry_path=tmp_path / "projects.json").tracking_uri.endswith(
        "/a.db"
    )

    forced = runner.invoke(
        app,
        ["register", "dup", "--tracking-uri", "sqlite:///b.db", "--overwrite", "--yes"],
        env=env,
    )
    assert forced.exit_code == 0
    assert get_project("dup", registry_path=tmp_path / "projects.json").tracking_uri.endswith(
        "/b.db"
    )


def test_register_rejects_malformed_tag(tmp_path) -> None:
    result = runner.invoke(
        app, ["register", "p", "--tag", "not-a-pair"], env=_env(tmp_path)
    )
    assert result.exit_code != 0
    assert "key=value" in result.output


def test_unregister_confirms_then_removes_project(tmp_path) -> None:
    env = _env(tmp_path)
    runner.invoke(app, ["register", "gone", "--tracking-uri", "sqlite:///a.db"], env=env)

    declined = runner.invoke(app, ["unregister", "gone"], env=env, input="n\n")
    assert declined.exit_code == 1
    assert len(list_projects(registry_path=tmp_path / "projects.json")) == 1

    confirmed = runner.invoke(app, ["unregister", "gone"], env=env, input="y\n")
    assert confirmed.exit_code == 0, confirmed.output
    assert list_projects(registry_path=tmp_path / "projects.json") == []

    missing = runner.invoke(app, ["unregister", "gone", "--yes"], env=env)
    assert missing.exit_code == 1


def test_dashboard_command_launches_app(tmp_path, monkeypatch) -> None:
    import mlops_toolbox.dashboard.app as dashboard_app

    launched: dict[str, object] = {}

    def fake_run_dashboard(host: str, port: int) -> None:
        launched["host"] = host
        launched["port"] = port

    monkeypatch.setattr(dashboard_app, "run_dashboard", fake_run_dashboard)
    monkeypatch.setenv("MLOPS_TOOLBOX_HOME", str(tmp_path))

    result = runner.invoke(app, ["dashboard", "--port", "9999"])
    assert result.exit_code == 0, result.output
    assert launched == {"host": "127.0.0.1", "port": 9999}
    assert "No projects registered yet" in result.output
