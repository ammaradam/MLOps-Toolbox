# Lifecycle health

`mt doctor` (and the dashboard's health badges) audit a project against an
ordered checklist of what a production-ready model needs:

1. **Tracking store reachable**
2. **Experiment runs recorded**
3. **Runs log metrics**
4. **Model registered**
5. **Model carries a contract**
6. **Drift reference stored**
7. **Contract records evaluation**

Every failing check comes with a *hint naming the exact call that fixes it*
— the doctor doubles as a teaching checklist for anyone new to the ML
lifecycle, and as a smoke test for everyone else.

```bash
$ mt doctor my-model
Auditing sqlite:///mlops.db
  [PASS] tracking store reachable: sqlite:///mlops.db
  [PASS] experiment runs recorded: 3 run(s)
  [PASS] runs log metrics: 3 of 3 run(s) have metrics
  [PASS] model registered: my-model
  [FAIL] model carries a contract: 0 of 1 model(s)
         hint: Store the input schema at registration:
         `register_model(..., contract=mt.infer_contract(X_train, evaluation=report))`
  ...
```

`mt doctor` exits `1` when any check fails, so it can run in CI right next
to `mt gate`. The audit is strictly read-only — it never writes to a
project's store.

You can point it at a registered project name or any tracking URI:

```bash
mt doctor my-project
mt doctor --tracking-uri sqlite:///somewhere/else.db
```
