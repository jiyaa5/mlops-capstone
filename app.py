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
from dotenv import load_dotenv

load_dotenv()  

# Get environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_RUN = os.getenv("MLFLOW_MODEL_RUN")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
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
