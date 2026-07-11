# Model contracts

A **contract** is everything a model needs downstream, captured once at
registration time and stored with the model version.

## What a contract contains

```python
contract = mt.infer_contract(X_train, y=y_train, evaluation=report)
```

| Field | Inferred from | Used by |
|---|---|---|
| `input_columns` — names and types | training dataframe dtypes | serving, scoring |
| per-column `categories` | low-cardinality string columns (≤ 20 distinct values) | serving, scoring (unknown categories rejected) |
| per-column `min_value` / `max_value` | numeric columns | model cards (informational — never enforced) |
| `output` — dtype and labels | `y` | model cards, docs |
| `evaluation` — the metrics report | `evaluation=` | quality gates, model cards |
| `task_type` | the evaluation report | cards, docs |

Alongside the contract, `register_model(..., reference_data=X_train)` also
stores:

- a **reference dataset** (sampled to 10,000 rows by default) — the baseline
  for input-drift checks, and
- the model's **predictions over that reference** — the baseline for
  prediction-drift checks, computed once so drift can be checked later
  without re-running the original model.

## How it propagates

```python
registry.register_model(
    model, name="my-model",
    contract=mt.infer_contract(X_train, y=y_train, evaluation=report),
    reference_data=X_train,
)
```

After that single call:

- **Serving** — `build_app(model, contract=...)` generates a typed request
  model from the schema. Wrong types, missing fields, and unknown categories
  get a 422 with a precise error; `GET /contract` returns the full contract.
- **Batch scoring** — `mt.score_dataframe(df, ...)` validates `df` against
  the contract (presence, type compatibility, categories), reorders columns
  to match training, carries extra columns through, and appends predictions.
- **Monitoring** — `mt.check_drift(current, ...)` and
  `mt.check_prediction_drift(preds, ...)` compare against the stored
  baselines.
- **Gates** — `report.check_gates(["f1_macro>=0.85"])` or `mt gate` in CI.
- **Cards** — `mt.generate_model_card(...)` renders it all as markdown.

## Validation philosophy

- **Types and categories are enforced.** A request or batch with an unknown
  category or wrong type is rejected at the boundary — that is a data bug.
- **Numeric ranges are informational.** Observed training ranges are
  recorded and shown on model cards, but a value outside the training range
  is *not* rejected — legitimate data moves. Range shift is a job for drift
  detection, not validation.

## Contracts without inference

`infer_contract` covers the common case, but contracts are plain pydantic
models — you can construct or adjust them explicitly:

```python
from mlops_toolbox import ColumnSpec, ModelContract

contract = ModelContract(
    input_columns=[
        ColumnSpec(name="amount", dtype="float"),
        ColumnSpec(name="segment", dtype="string", categories=["a", "b", "c"]),
    ],
)
```
