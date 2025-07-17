# MLOps-Toolbox

Compilation of all relevant tools for better ML Operations.

This repository provides simple decorators to mark functions as tasks or pipelines.

## Decorators

- `@task`: Mark a Python function as an executable task.
- `@pipeline`: Mark a function that orchestrates multiple tasks.

## Example

```
from mlops_toolbox import task, pipeline

@task
def load_data():
    print("Loading data...")
    return [1, 2, 3]

@task
def preprocess(data):
    print("Preprocessing", data)
    return [x * 2 for x in data]

@pipeline
def my_pipeline():
    data = load_data()
    data = preprocess(data)
    print("Done")

if __name__ == "__main__":
    my_pipeline()
```

See `examples/simple_usage.py` for a complete runnable example.
