from __future__ import annotations

from pathlib import Path

from mlops_toolbox.projects import register_project


def test_relative_sqlite_uri_is_absolutized(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    info = register_project("p", "sqlite:///mlops.db", registry_path=tmp_path / "projects.json")
    expected = "sqlite:///" + (tmp_path / "mlops.db").resolve().as_posix()
    assert info.tracking_uri == expected


def test_relative_sqlite_uri_with_subdir(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    info = register_project(
        "p", "sqlite:///stores/mlops.db", registry_path=tmp_path / "projects.json"
    )
    assert info.tracking_uri == "sqlite:///" + (tmp_path / "stores/mlops.db").resolve().as_posix()


def test_absolute_sqlite_uri_unchanged(tmp_path) -> None:
    absolute = "sqlite:///" + (tmp_path / "mlops.db").resolve().as_posix()
    info = register_project("p", absolute, registry_path=tmp_path / "projects.json")
    assert info.tracking_uri == absolute


def test_non_sqlite_uri_unchanged(tmp_path) -> None:
    info = register_project(
        "p", "http://tracking.internal:5000", registry_path=tmp_path / "projects.json"
    )
    assert info.tracking_uri == "http://tracking.internal:5000"


def test_normalization_does_not_create_the_store(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    register_project("p", "sqlite:///new.db", registry_path=tmp_path / "projects.json")
    assert not Path(tmp_path / "new.db").exists()
