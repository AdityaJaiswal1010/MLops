#running the pipeline

from pipelines.training_pipelines import training_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline()

# the below command is to vizualize the accuracy and scores of the model 
# mlflow ui --backend-store-uri "file:/Users/adityajaiswal/Library/Application Support/zenml/local_stores/d38a8183-c5d8-49f4-b568-ee3747756a73/mlruns"