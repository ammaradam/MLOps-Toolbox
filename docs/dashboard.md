# Dashboard

One local, read-only web pane across every project you register: runs,
registered models, metric trend sparklines, and lifecycle health badges
backed by the same checks as `mt doctor`.

## Register projects

A project is just a name pointing at a tracking URI:

```python
import mlops_toolbox as mt

mt.register_project(
    "churn-model",
    "sqlite:///mlops.db",
    description="Customer churn prediction",
    tags={"team": "growth"},
)
```

Registration is core-only (no extras needed) and stored in
`~/.mlops-toolbox/projects.json` — override the location with the
`MLOPS_TOOLBOX_HOME` environment variable or an explicit `registry_path=`.

!!! tip "Use absolute tracking URIs"
    A relative URI like `sqlite:///mlops.db` resolves against the *current
    working directory* of whoever reads it. Register projects with absolute
    paths so the dashboard and `mt doctor` find the same store you trained
    against.

## Run it

```python
mt.run_dashboard()  # http://127.0.0.1:8050
```

or from the shell:

```bash
python -m mlops_toolbox.dashboard [--host HOST] [--port PORT]
```

## What you see

- **Index** — every registered project with a health badge
  (pass / warn / fail counts from the lifecycle audit).
- **Project detail** — the full health checklist with fix-it hints, metric
  stat tiles with sparklines across runs, the runs table, and registered
  models with their aliases and tags.

The dashboard never writes to a project's store — it only queries it.
