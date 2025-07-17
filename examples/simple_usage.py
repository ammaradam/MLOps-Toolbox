from mlops_toolbox import task, pipeline


@task
def load_data():
    print("Loading data...")
    return [1, 2, 3]


@task
def preprocess(data):
    print("Preprocessing", data)
    return [x * 2 for x in data]


@task
def train_model(data):
    print("Training model with", data)
    return {"model": "dummy"}


@pipeline
def training_pipeline():
    data = load_data()
    processed = preprocess(data)
    model = train_model(processed)
    print("Pipeline result:", model)


if __name__ == "__main__":
    training_pipeline()
