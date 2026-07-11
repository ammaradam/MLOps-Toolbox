from __future__ import annotations

import ast

from typer.testing import CliRunner

from mlops_toolbox.cli._app import app

runner = CliRunner()

_EXPECTED = [
    "train.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".github/workflows/model-ci.yml",
]


def test_init_scaffolds_project(tmp_path) -> None:
    target = tmp_path / "churn"
    result = runner.invoke(app, ["init", "churn", "--dir", str(target)])
    assert result.exit_code == 0, result.output

    for relative_path in _EXPECTED:
        assert (target / relative_path).exists(), relative_path

    train_source = (target / "train.py").read_text(encoding="utf-8")
    ast.parse(train_source)  # generated code must be valid Python
    assert 'MODEL_NAME = "churn"' in train_source
    assert "infer_contract" in train_source
    assert "{{project}}" not in train_source

    workflow = (target / ".github/workflows/model-ci.yml").read_text(encoding="utf-8")
    assert "mt gate" in workflow
    assert "--model churn" in workflow


def test_init_refuses_non_empty_dir(tmp_path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("data", encoding="utf-8")

    result = runner.invoke(app, ["init", "existing", "--dir", str(target)])
    assert result.exit_code == 1
    assert "not empty" in result.output

    forced = runner.invoke(app, ["init", "existing", "--dir", str(target), "--force"])
    assert forced.exit_code == 0, forced.output
