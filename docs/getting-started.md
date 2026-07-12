# Getting started

## Install

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add "mlops-toolbox[all]"        # everything
# or pick stages:
uv add "mlops-toolbox[tracking]"   # experiment tracking
uv add "mlops-toolbox[registry]"   # model registry (includes tracking)
uv add "mlops-toolbox[monitoring]" # drift detection
uv add "mlops-toolbox[deploy]"     # model serving
uv add "mlops-toolbox[dashboard]"  # multi-project dashboard
uv add "mlops-toolbox[cli]"        # the `mt` command line
```

Prefer pip? `pip install "mlops-toolbox[...]"` works identically.

The base install is deliberately light (pydantic, numpy, pandas). Heavy
backends load lazily, and a missing extra raises an error naming exactly
what to install.

## The fastest path: `mt init`

```bash
uv init my-model && cd my-model
uv add "mlops-toolbox[all]" scikit-learn
uv run mt init my-model --dir . --force
uv run python train.py
```

(In an activated virtualenv or pip install, drop the `uv run` prefix — `mt`
is on your PATH.)

The scaffold gives you a working `train.py` (swap in your data and model at
the two `TODO`s), a CI workflow that gates on model quality, and a README.
Running it trains, evaluates, and registers the model **with its contract**.

From there, each lifecycle action is one command:

```bash
mt doctor my-model                        # audit lifecycle completeness
mt serve my-model --version production    # typed HTTP API
mt score my-model --input new_data.csv    # batch scoring
mt drift my-model --input new_data.csv    # drift check, exit 1 on drift
mt gate "accuracy>=0.85" --model my-model # quality gate, exit 1 on failure
mt card my-model --out MODEL_CARD.md      # model card
mt ship my-model                          # deployable folder
```

## The same lifecycle in Python

```python
import mlops_toolbox as mt

split = mt.train_test_split(X, y, test_size=0.2, random_state=42)
model = mt.adapt(trained_model)  # anything with predict()

tracker = mt.tracker("sqlite:///mlops.db", experiment_name="demo")
tracker.start_run()
report = mt.ClassificationEvaluator().evaluate(split.y_test, model.predict(split.X_test))
tracker.log_metrics(report.metrics)
run = tracker.end_run()

registry = mt.model_registry("sqlite:///mlops.db")
info = registry.register_model(
    model,
    name="my-model",
    run_id=run.run_id,
    contract=mt.infer_contract(split.X_train, y=split.y_train, evaluation=report),
    reference_data=split.X_train,
)
registry.set_alias("my-model", info.version, "production")
```

Everything downstream — scoring, drift, serving, gates, cards — now works
with no further configuration. See the [Python API
reference](reference/python-api.md) for the full surface.

## Versions and aliases

Anywhere a command or function accepts a `version`, you can pass:

- a version number: `"3"`,
- `"latest"` — the highest registered version,
- an **alias** you set, like `"production"` or `"champion"`:

```python
registry.set_alias("my-model", "3", "production")
registry.get_model("my-model", version="production")
```

```bash
mt serve my-model --version production
```
