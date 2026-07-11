# mlops-toolbox

Composable, backend-agnostic building blocks for the end-to-end ML lifecycle
— with one idea that ties them together: **your model remembers its own
contract**.

## The problem

Every MLOps tool covers one lifecycle stage well; almost none carry context
*between* stages. That handoff gap is exactly where models break on the way
to production:

- the serving endpoint doesn't know the input schema,
- the monitoring job doesn't have the training data,
- the CI pipeline doesn't know what "good enough" was.

## The idea: contract propagation

When you register a model, `mlops-toolbox` stores a **contract** alongside
it — the input schema (with category sets and value ranges) inferred from
your training data, the evaluation report, a reference dataset, and baseline
predictions. Every downstream stage then consumes the contract
automatically:

| Stage | What the contract enables |
|---|---|
| Serving | `/predict` typed and validated from the schema; `GET /contract` self-describes the model |
| Batch scoring | Inputs checked before the model ever sees them |
| Monitoring | Input drift *and* prediction drift against stored baselines |
| CI | Quality gates on the metrics the model shipped with |
| Docs | Model cards rendered from what the registry already knows |

One `register_model()` call at training time; zero configuration everywhere
else.

## Design principles

- **Backend-agnostic.** You code against small stage interfaces constructed
  by factories — `tracker()`, `model_registry()`, `drift_detector()`,
  `model_server()` — never against the tools behind them. Backends are
  implementation details and can be replaced any time.
- **Local-first.** One file-based store, no server, no account. `pip
  install` and everything works offline.
- **Strictly typed.** `mypy --strict` with zero errors, pydantic data
  contracts throughout, tests against real backends rather than mocks.

## Where next

- [Getting started](getting-started.md) — from `pip install` to a served,
  monitored model in five minutes.
- [Model contracts](concepts/contracts.md) — how contract propagation works.
- [For software engineers](guides/software-engineers.md) — moving into ML
  with the workflows you already trust.
- [For data scientists](guides/data-scientists.md) — from a notebook model
  to production.
- [CLI reference](reference/cli.md) — every `mt` command.
