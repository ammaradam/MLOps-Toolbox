# CLI reference

All commands share two options where a model is involved:

- `--tracking-uri` — the tracking store (default `sqlite:///mlops.db`)
- `--version` — a version number, `latest` (default), or an alias like `production`

Commands designed for automation use exit codes: `0` success / all checks
pass, `1` failure — so they drop straight into CI and cron.

## `mt init NAME`

Scaffold a new project: `train.py` (contract-registered training),
`requirements.txt`, README, `.gitignore`, and a GitHub Actions workflow that
trains, gates, and audits on every PR.

```bash
mt init my-model [--dir path] [--force]
```

## `mt serve NAME`

Serve a registered model over HTTP. With a stored contract, `/predict` is
typed and validated and `GET /contract` describes the schema; without one, a
generic `instances` endpoint is exposed.

```bash
mt serve my-model [--version production] [--host 127.0.0.1] [--port 8000]
```

## `mt ship NAME`

Export a self-contained deployable folder: model artifacts, `serve.py`,
`Dockerfile`, pinned `requirements.txt`, README. Needs no tracking store at
runtime.

```bash
mt ship my-model [--out deploy/my-model] [--force]
```

## `mt score NAME`

Batch-score a `.parquet`/`.csv` file, enforcing the contract on the input.
Writes the input plus a prediction column. Exit `1` on contract violations.

```bash
mt score my-model --input data.parquet [--out scored.parquet] [--prediction-column pred]
```

## `mt gate EXPR...`

Check metric gates against a run's metrics. Exit `1` when any gate fails.
Expressions: `metric OP number` with `>= <= == > <`.

```bash
mt gate "f1_macro>=0.85" "accuracy>0.9" [--model NAME | --run RUN_ID]
```

Defaults to the latest finished run with metrics; `--model` gates the
registered model's source run.

## `mt drift NAME`

Check drift against the baseline stored at registration. Exit `1` when
dataset drift is detected.

```bash
mt drift my-model --input current.parquet              # input-data drift
mt drift my-model --input preds.parquet --predictions  # prediction drift
```

## `mt card NAME`

Render a markdown model card — identity, input schema (with categories and
observed ranges), evaluation metrics, monitoring status.

```bash
mt card my-model [--out MODEL_CARD.md]
```

## `mt doctor [PROJECT]`

Audit lifecycle completeness with fix-it hints. Exit `1` when any check
fails. See [Lifecycle health](../concepts/lifecycle-health.md).

```bash
mt doctor my-project
mt doctor --tracking-uri sqlite:///mlops.db
```

## `mt models [PROJECT]`

List registered models: latest version, aliases, contract and drift-baseline
coverage.

```bash
mt models [--tracking-uri URI]
```

## `mt projects`

List locally registered projects (name, tracking URI, description, tags).
