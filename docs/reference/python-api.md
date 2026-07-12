# Python API reference

Everything below is importable from the package root:

```python
import mlops_toolbox as mt
```

## Factories (start here)

Backend-agnostic constructors returning stage interfaces — the primary way
to build each stage:

| Factory | Returns | Purpose |
|---|---|---|
| `mt.tracker(tracking_uri, experiment_name)` | `ExperimentTracker` | log params/metrics/artifacts per run; usable as a context manager |
| `mt.model_registry(tracking_uri, registry_uri)` | `ModelRegistry` | register, load, alias, export models and their contracts |
| `mt.drift_detector()` | `DriftDetector` | compare two dataframes for drift |
| `mt.model_server()` | `ModelServer` | build and serve an HTTP app for a model |

Omitted URIs resolve through settings — `MT_TRACKING_URI` env var, the
nearest `mlops.toml`, then `mt.DEFAULT_TRACKING_URI`
(`"sqlite:///mlops.db"`).

## Settings

- `mt.load_settings()` → `ToolboxSettings` — resolved configuration
  (tracking/registry URIs, serve defaults, `[aws]`/`[gcp]`/`[azure]`
  sections). Precedence: kwargs > `MT_*` env vars > `mlops.toml` > defaults.
- `mt.find_config_file()` → `Path | None` — the `mlops.toml` in effect,
  searched upward from the working directory.
- Inspect from the CLI with `mt config`.

## Data

- `mt.validate_dataframe(df, RowSchema)` → `ValidationResult` — validate
  every row against a pydantic schema in one pass.
- `mt.train_test_split(X, y, test_size, random_state)` → `DatasetSplit`.

## Models

- `mt.adapt(model)` → `ModelAdapter` — wrap anything with `predict()` (and
  optionally `predict_proba()`) in a uniform interface.

## Contracts

- `mt.infer_contract(X, y=None, evaluation=None)` → `ModelContract` — infer
  the input schema (types, categories, ranges), output spec, and attach the
  evaluation report.
- `mt.validate_against_contract(df, contract)` → `list[str]` — problems
  found, empty when valid.
- Models: `ModelContract`, `ColumnSpec`, `OutputSpec`.

## Registry

On any `ModelRegistry`:

- `register_model(model, name, run_id=None, tags=None, contract=None, reference_data=None)` → `ModelInfo`
- `get_model(name, version="latest")` / `get_model_info` / `list_versions`
- `set_alias(name, version, alias)` — point `"production"` etc. at a version
- `get_contract` / `get_reference_data` / `get_reference_predictions`
- `export_model(name, dst_dir, version)` — self-contained artifact copy

`version` is always a number, `"latest"`, or an alias.

## Evaluation and gates

- `mt.ClassificationEvaluator().evaluate(y_true, y_pred)` → `EvaluationReport`
- `mt.RegressionEvaluator().evaluate(y_true, y_pred)` → `EvaluationReport`
- `report.check_gates(["f1_macro>=0.85"])` / `mt.check_gates(metrics, gates)`
  → `GateResult`

## Monitoring

- `mt.check_drift(current_df, registry=..., name=..., version="latest")` →
  `DriftReport` — against the stored reference dataset.
- `mt.check_prediction_drift(predictions, registry=..., name=...)` →
  `DriftReport` — against the stored baseline predictions.

## Scoring

- `mt.score_dataframe(df, registry=..., name=..., version="latest",
  prediction_column="prediction")` → dataframe copy with predictions;
  raises `ValidationFailedError` on contract violations.

## Cards

- `mt.generate_model_card(registry, name, version="latest")` → markdown `str`.

## Projects and dashboard

- `mt.register_project(name, tracking_uri, description=None, tags=None)` /
  `mt.list_projects()` / `mt.get_project(name)` / `mt.unregister_project(name)`
- `mt.run_dashboard(host="127.0.0.1", port=8050)`

## Exceptions

All inherit `MLOpsToolboxError`: `BackendNotInstalledError`,
`ValidationFailedError`, `TrackingError`, `RegistryError`,
`MonitoringError`, `ProjectError` (+ `ProjectNotFoundError`,
`ProjectAlreadyRegisteredError`).
