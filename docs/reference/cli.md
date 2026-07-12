# CLI reference

All commands share two options where a model is involved:

- `--tracking-uri` — the tracking store. When omitted it resolves through
  settings: `MT_TRACKING_URI` env var, then the nearest `mlops.toml`, then
  the local default `sqlite:///mlops.db`. See [`mt config`](#mt-config).
- `--version` — a version number, `latest` (default), or an alias like `production`

Commands designed for automation use exit codes: `0` success / all checks
pass, `1` failure — so they drop straight into CI and cron.

## `mt init NAME`

Scaffold a new project: `train.py` (contract-registered training),
`mlops.toml` (settings — tracking URI, cloud targets), `requirements.txt`,
README, `.gitignore`, and a GitHub Actions workflow that trains, gates, and
audits on every PR.

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
mt ship my-model [--out deploy/my-model] [--force] [--yes]
```

`--force` (overwriting a non-empty directory) asks for confirmation;
`--yes`/`-y` skips the prompt. The same applies to `mt init --force`.

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

## `mt register NAME` / `mt edit NAME` / `mt unregister NAME`

Manage project registrations (name → tracking URI) used by the dashboard,
`mt doctor`, and `mt models`.

```bash
mt register churn --tracking-uri sqlite:///mlops.db \
  --description "Customer churn" --tag team=growth [--overwrite] [--yes]
mt edit churn --name churn-v2 --description "New description"
mt unregister churn [--yes]
```

Relative sqlite paths are **resolved to absolute at registration time** —
otherwise the URI would point at a different (auto-created, empty) store
depending on where later commands run. Overwriting a registration and
unregistering both ask for confirmation; pass `--yes`/`-y` to skip in
scripts.

Registrations live in `~/.mlops-toolbox/projects.json` (override with
`MLOPS_TOOLBOX_HOME`); unregistering never touches the tracking store itself.

## `mt projects`

List locally registered projects (name, tracking URI, description, tags).

## `mt dashboard`

Run the multi-project dashboard over every registered project. Tracking
stores are only ever read; project registrations can be renamed, described,
or removed (with confirmation) from each project's settings panel.

```bash
mt dashboard [--host 127.0.0.1] [--port 8050]
```

## `mt config`

Show every resolved setting and where its value came from — `[env]` for an
`MT_*` environment variable, `[file]` for the nearest `mlops.toml`, or
`[default]`. Settings never contain secrets: cloud credentials come from each
provider's default chain, API keys from environment variables only.

```bash
mt config
```

Resolution order everywhere: CLI flags > `MT_*` env vars (nested fields via
`__`, e.g. `MT_AWS__REGION`) > `mlops.toml` (searched upward from the working
directory, like git config) > built-in defaults.
