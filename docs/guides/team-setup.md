# Team setup: shared tracking and registry

Everything in mlops-toolbox runs local-first by default — one SQLite file, no
server. For a team (or CI) you want one shared source of truth instead: a
remote MLflow tracking server backed by a real database and object storage.
Because the toolbox stores contracts, reference data, and baseline
predictions as ordinary MLflow artifacts, **nothing about your code changes**
— you point the tracking URI at the server and every stage (tracking,
registry, serving, scoring, drift, gates, cards, dashboard) follows.

## 1. Run a shared MLflow server

A production-shaped MLflow server needs a database for metadata and an object
store for artifacts:

```bash
mlflow server \
  --backend-store-uri postgresql://mlflow:...@db.internal:5432/mlflow \
  --artifacts-destination s3://my-ml-artifacts/mlflow \
  --host 0.0.0.0 --port 5000
```

- **Database**: PostgreSQL (or MySQL). Required for the Model Registry.
- **Artifact store**: S3, GCS (`gs://…`), or Azure Blob
  (`wasbs://…`). MLflow speaks all three natively; clients need the matching
  credentials available (see below).

All three clouds also offer a managed or near-managed path:

| Platform | Shared MLflow option |
|---|---|
| AWS | SageMaker managed MLflow — use the tracking-server ARN as the URI |
| Azure | Every Azure ML workspace exposes an MLflow tracking URI (`azureml://…`) |
| GCP | Host MLflow on Cloud Run + Cloud SQL + GCS (standard pattern) |

!!! warning "Azure ML registry limitation"
    Azure ML's MLflow **registry** shim does not support model aliases, which
    the toolbox's promotion flow (`set_alias`, `--version production`) relies
    on. Use `azureml://` for *tracking* and point `registry_uri` at an
    alias-capable MLflow server for the registry. The toolbox raises a clear
    error if an alias operation is unsupported by the store.

## 2. Point the toolbox at it

Create an `mlops.toml` at your project root (checked in — it holds no
secrets):

```toml
tracking_uri = "https://mlflow.internal.example.com"
# registry_uri = "https://mlflow-registry.internal.example.com"  # only if different
```

Or set it per-environment with `MT_TRACKING_URI`. Verify with `mt config`.
Every factory (`mt.tracker()`, `mt.model_registry()`) and every `mt` command
resolves the URI the same way: CLI flag > `MT_*` env var > `mlops.toml` >
local default.

## 3. Credentials

Never put credentials in `mlops.toml`. The clients pick them up from the
environment:

- **MLflow server auth**: `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`
  or `MLFLOW_TRACKING_TOKEN` — passed through untouched.
- **S3 artifacts**: the boto3 default chain (`AWS_ACCESS_KEY_ID`/IAM role/SSO).
- **GCS artifacts**: application-default credentials
  (`GOOGLE_APPLICATION_CREDENTIALS` or workload identity).
- **Azure Blob artifacts**: `AZURE_STORAGE_CONNECTION_STRING` or
  `DefaultAzureCredential`.

## 4. Share the project registry

`~/.mlops-toolbox/projects.json` is per-machine, but a registration is just a
name → tracking URI pointer. Each teammate (and the dashboard host) registers
the same remote URI once:

```bash
mt register churn --tracking-uri https://mlflow.internal.example.com --tag team=growth
mt dashboard
```

## 5. CI against the shared store

With `mlops.toml` checked in, the scaffolded CI workflow trains and gates
against the shared server automatically — supply only the auth env vars as CI
secrets:

```yaml
- name: Train and register
  run: python train.py
  env:
    MLFLOW_TRACKING_TOKEN: ${{ secrets.MLFLOW_TRACKING_TOKEN }}
- name: Quality gate
  run: mt gate "accuracy>=0.85" --model churn
```

To keep CI on an ephemeral local store instead (e.g. for PR validation
without touching the shared registry), override per-job:
`MT_TRACKING_URI=sqlite:///ci.db`.
