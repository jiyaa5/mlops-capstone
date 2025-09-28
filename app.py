# from fastapi import FastAPI
# import joblib

# # TODO: Load model from MLflow instead of local pkl
# model = joblib.load("model.pkl")

# app = FastAPI()

# @app.post("/predict")
# def predict(data: dict):
#     # TODO: Handle proper preprocessing
#     area = data["area"]
#     prediction = model.predict([[area]])[0]
#     return {"prediction": prediction}
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment = client.get_experiment_by_name("Housing-Regression")
runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_MODEL_RUN = f"runs:/{runs[0].info.run_id}/model"

model = mlflow.pyfunc.load_model(MLFLOW_MODEL_RUN)



app = FastAPI()

class InputData(BaseModel):
    area: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([{"area": data.area}])
    try:
        prediction = model.predict(df)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
