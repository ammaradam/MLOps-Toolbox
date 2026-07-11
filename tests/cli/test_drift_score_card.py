from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

pytest.importorskip("evidently")

from mlops_toolbox.adapters import adapt
from mlops_toolbox.cli._app import app
from mlops_toolbox.contracts import infer_contract
from mlops_toolbox.factories import model_registry

runner = CliRunner()


@pytest.fixture
def drift_store(tmp_path, dummy_model, sample_dataframe_pair_with_drift):
    """Store with a model registered against a 300-row reference. Returns (uri, ref, drifted)."""
    reference, drifted = sample_dataframe_pair_with_drift
    uri = f"sqlite:///{tmp_path / 'store.db'}"
    registry = model_registry(uri)
    registry.register_model(
        adapt(dummy_model),
        name="drift-model",
        contract=infer_contract(reference),
        reference_data=reference,
    )
    return uri, reference, drifted


def test_drift_ok_exits_zero(drift_store, tmp_path) -> None:
    uri, reference, _ = drift_store
    current_path = tmp_path / "current.parquet"
    reference.to_parquet(current_path, index=False)

    result = runner.invoke(
        app, ["drift", "drift-model", "--input", str(current_path), "--tracking-uri", uri]
    )
    assert result.exit_code == 0, result.output
    assert "No dataset drift" in result.output


def test_drift_detected_exits_one(drift_store, tmp_path) -> None:
    uri, _, drifted = drift_store
    current_path = tmp_path / "current.parquet"
    drifted.to_parquet(current_path, index=False)

    result = runner.invoke(
        app, ["drift", "drift-model", "--input", str(current_path), "--tracking-uri", uri]
    )
    assert result.exit_code == 1
    assert "Drift DETECTED" in result.output


def test_drift_predictions_mode(drift_store, tmp_path) -> None:
    uri, reference, _ = drift_store
    preds_path = tmp_path / "preds.parquet"
    pd.DataFrame({"prediction": np.ones(len(reference), dtype=int)}).to_parquet(
        preds_path, index=False
    )

    result = runner.invoke(
        app,
        [
            "drift",
            "drift-model",
            "--input",
            str(preds_path),
            "--predictions",
            "--tracking-uri",
            uri,
        ],
    )
    assert result.exit_code == 1
    assert "prediction drift" in result.output


def test_drift_without_baseline_exits_one(tmp_path, dummy_model) -> None:
    uri = f"sqlite:///{tmp_path / 'store.db'}"
    model_registry(uri).register_model(adapt(dummy_model), name="bare")
    current_path = tmp_path / "current.csv"
    pd.DataFrame({"a": [1.0]}).to_csv(current_path, index=False)

    result = runner.invoke(
        app, ["drift", "bare", "--input", str(current_path), "--tracking-uri", uri]
    )
    assert result.exit_code == 1
    assert "no stored reference" in result.output


def test_score_writes_predictions_file(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    input_path = tmp_path / "batch.csv"
    pd.DataFrame({"feature_a": [1.0, -1.0], "feature_b": [3.0, 4.0]}).to_csv(
        input_path, index=False
    )

    result = runner.invoke(
        app, ["score", name, "--input", str(input_path), "--tracking-uri", uri]
    )
    assert result.exit_code == 0, result.output
    scored = pd.read_csv(tmp_path / "batch.scored.csv")
    assert scored["prediction"].tolist() == [1, 0]


def test_score_contract_violation_exits_one(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    input_path = tmp_path / "bad.csv"
    pd.DataFrame({"feature_a": [1.0]}).to_csv(input_path, index=False)

    result = runner.invoke(
        app, ["score", name, "--input", str(input_path), "--tracking-uri", uri]
    )
    assert result.exit_code == 1
    assert "missing column 'feature_b'" in result.output


def test_card_prints_and_writes(registered_model_store, tmp_path) -> None:
    uri, name = registered_model_store
    printed = runner.invoke(app, ["card", name, "--tracking-uri", uri])
    assert printed.exit_code == 0, printed.output
    assert f"# Model card: {name}" in printed.output

    out = tmp_path / "cards" / "model.md"
    written = runner.invoke(app, ["card", name, "--tracking-uri", uri, "--out", str(out)])
    assert written.exit_code == 0, written.output
    assert out.exists()
    assert "## Input schema" in out.read_text(encoding="utf-8")
