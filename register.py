import os, mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name("Housing-Regression")

# Get latest run
runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time desc"], max_results=1)
latest_run_id = runs[0].info.run_id

# Register the model from latest run
result = mlflow.register_model(
    f"runs:/{latest_run_id}/model",
    "HousingPricePredictor"
)

print("Registered model:", result.name, "version:", result.version)
