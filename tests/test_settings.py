from __future__ import annotations

from pathlib import Path

import pytest

from mlops_toolbox.settings import explain_settings, find_config_file, load_settings


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from an empty tmp dir so no real mlops.toml leaks in."""
    monkeypatch.chdir(tmp_path)


def test_defaults_without_any_config() -> None:
    settings = load_settings()
    assert settings.tracking_uri == "sqlite:///mlops.db"
    assert settings.registry_uri is None
    assert settings.default_deploy_target == "local"
    assert settings.aws.region is None


def test_toml_file_in_cwd(tmp_path: Path) -> None:
    (tmp_path / "mlops.toml").write_text(
        'tracking_uri = "https://mlflow.example.com"\n\n[aws]\nregion = "eu-west-1"\n',
        encoding="utf-8",
    )
    settings = load_settings()
    assert settings.tracking_uri == "https://mlflow.example.com"
    assert settings.aws.region == "eu-west-1"


def test_toml_discovered_walking_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mlops.toml").write_text('tracking_uri = "sqlite:///team.db"\n', encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert find_config_file() == tmp_path / "mlops.toml"
    assert load_settings().tracking_uri == "sqlite:///team.db"


def test_toml_with_utf8_bom(tmp_path: Path) -> None:
    # Windows editors and PowerShell write UTF-8 with a BOM by default.
    (tmp_path / "mlops.toml").write_bytes(
        b'\xef\xbb\xbftracking_uri = "sqlite:///bom.db"\n'
    )
    assert load_settings().tracking_uri == "sqlite:///bom.db"


def test_env_overrides_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mlops.toml").write_text(
        'tracking_uri = "sqlite:///file.db"\n\n[aws]\nregion = "eu-west-1"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MT_TRACKING_URI", "https://mlflow.from-env.example.com")
    monkeypatch.setenv("MT_AWS__REGION", "us-east-1")

    settings = load_settings()
    assert settings.tracking_uri == "https://mlflow.from-env.example.com"
    assert settings.aws.region == "us-east-1"


def test_init_kwargs_override_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MT_TRACKING_URI", "https://mlflow.from-env.example.com")
    settings = load_settings(tracking_uri="sqlite:///explicit.db")
    assert settings.tracking_uri == "sqlite:///explicit.db"


def test_explain_settings_reports_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mlops.toml").write_text(
        'tracking_uri = "sqlite:///file.db"\n\n[gcp]\nproject = "my-project"\n', encoding="utf-8"
    )
    monkeypatch.setenv("MT_AWS__REGION", "us-east-1")

    sources = {row.key: row.source for row in explain_settings()}
    assert sources["tracking_uri"] == "file"
    assert sources["gcp.project"] == "file"
    assert sources["aws.region"] == "env"
    assert sources["registry_uri"] == "default"


def test_factories_resolve_uri_from_settings(tmp_path: Path) -> None:
    db_path = (tmp_path / "settings.db").as_posix()
    (tmp_path / "mlops.toml").write_text(
        f'tracking_uri = "sqlite:///{db_path}"\n', encoding="utf-8"
    )

    from mlops_toolbox.factories import tracker

    t = tracker()
    assert t.tracking_uri == f"sqlite:///{db_path}"  # type: ignore[attr-defined]
