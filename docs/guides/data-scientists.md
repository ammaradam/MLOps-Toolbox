# For data scientists moving to production

You have a model in a notebook. The gap between that and production is
scaffolding you shouldn't have to hand-build: an API with input validation,
containerization, monitoring, and documentation. `mlops-toolbox` generates
all of it from one registration call.

## Step 1: register with a contract

Wherever your training happens (notebook included):

```python
import mlops_toolbox as mt

model = mt.adapt(clf)  # any object with predict()

registry = mt.model_registry("sqlite:///mlops.db")
registry.register_model(
    model,
    name="churn",
    contract=mt.infer_contract(X_train, y=y_train, evaluation=report),
    reference_data=X_train,
)
```

That's the whole integration. The contract (schema, categories, metrics) and
drift baselines are now stored with the model.

## Step 2: serve it

```bash
mt serve churn --port 8000
```

- `POST /predict` accepts `{"records": [{...one object per column...}]}` and
  validates types and categories — bad payloads get a 422 explaining what's
  wrong, not a numpy traceback.
- `GET /contract` tells consumers exactly what to send.

## Step 3: hand it to the platform team

```bash
mt ship churn
```

produces a self-contained folder — model artifacts, `serve.py`, `Dockerfile`,
pinned `requirements.txt`, README — that runs anywhere with no access to
your tracking store:

```bash
cd deploy/churn && docker build -t churn-service . && docker run -p 8000:8000 churn-service
```

## Step 4: score batches without an endpoint

Most production scoring is batch. Contract enforcement means schema mistakes
surface as readable errors before the model runs:

```bash
mt score churn --input monthly_batch.parquet --out predictions.parquet
```

## Step 5: monitor without keeping the training data

The reference data and baseline predictions were stored at registration:

```bash
mt drift churn --input monthly_batch.parquet                     # input drift
mt drift churn --input predictions.parquet --predictions         # prediction drift
```

## Step 6: document it

```bash
mt card churn --out MODEL_CARD.md
```

renders the schema, metrics, and monitoring status — the artifact reviewers
and stakeholders ask for, generated from what the registry already knows.
