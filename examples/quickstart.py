"""End-to-end mlops-toolbox demo: validate -> split -> train -> track -> evaluate -> register.

Run with: uv run --extra all --group dev python examples/quickstart.py
(requires scikit-learn, which is a dev/example-only dependency, never a library dependency)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
from sklearn.linear_model import LogisticRegression

import mlops_toolbox as mt

TRACKING_URI = "sqlite:///mlops.db"


class CustomerRow(pydantic.BaseModel):
    tenure_months: int = pydantic.Field(ge=0)
    monthly_spend: float = pydantic.Field(ge=0)
    churned: int


def main() -> None:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 60, size=n),
            "monthly_spend": rng.uniform(10, 200, size=n),
            "churned": rng.integers(0, 2, size=n),
        }
    )

    print("1. Validating data...")
    validation = mt.validate_dataframe(df, CustomerRow)
    assert validation.is_valid, validation.errors
    print(f"   OK - {len(df)} rows validated against {validation.schema_name}")

    print("2. Splitting data...")
    X = df[["tenure_months", "monthly_spend"]]
    y = df["churned"]
    split = mt.train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   train={len(split.X_train)} test={len(split.X_test)}")

    print("3. Training model...")
    sklearn_model = LogisticRegression().fit(split.X_train, split.y_train)
    model = mt.adapt(sklearn_model)

    print("4. Tracking experiment run (local store, no server)...")
    tracker = mt.tracker(tracking_uri=TRACKING_URI, experiment_name="quickstart")
    tracker.start_run()
    tracker.log_params({"model": "LogisticRegression"})
    y_pred = model.predict(split.X_test)
    report = mt.ClassificationEvaluator().evaluate(split.y_test, y_pred)
    tracker.log_metrics(report.metrics)
    run_result = tracker.end_run()
    print(f"   run_id={run_result.run_id} metrics={report.metrics}")

    print("5. Registering model (with its contract: schema + eval + drift baseline)...")
    registry = mt.model_registry(TRACKING_URI)
    contract = mt.infer_contract(split.X_train, y=split.y_train, evaluation=report)
    model_info = registry.register_model(
        model,
        name="churn-model",
        run_id=run_result.run_id,
        contract=contract,
        reference_data=split.X_train,
    )
    registry.set_alias("churn-model", model_info.version, "production")
    print(f"   registered {model_info.name} v{model_info.version} -> alias 'production'")

    print("6. Checking for data drift against the stored reference...")
    current = X.iloc[len(X) // 2 :]
    drift_report = mt.check_drift(current, registry=registry, name="churn-model")
    print(f"   dataset_drift_detected={drift_report.dataset_drift_detected}")

    print("7. Batch-scoring with contract enforcement...")
    scored = mt.score_dataframe(current, registry=registry, name="churn-model")
    print(f"   scored {len(scored)} rows -> column '{scored.columns[-1]}'")

    print("8. Registering project for the dashboard / doctor...")
    project = mt.register_project(
        "churn-model",
        TRACKING_URI,
        description="Customer churn prediction",
        tags={"team": "growth"},
    )
    print(f"   registered project: {project.name}")

    print("\nDone. Try: mt doctor churn-model | mt card churn-model | mt serve churn-model")


if __name__ == "__main__":
    main()
