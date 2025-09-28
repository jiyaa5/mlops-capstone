import os, pandas as pd, mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

app = FastAPI()
client = MlflowClient()

EXPERIMENT_NAME = os.environ["MLFLOW_EXPERIMENT_NAME"]
REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_REGISTERED_MODEL_NAME") 

class InputData(BaseModel):
    area: float

def load_model():
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    models = [os.path.join("mlruns", exp.experiment_id, "models", d, "artifacts")
              for d in os.listdir(os.path.join("mlruns", exp.experiment_id, "models"))
              if d.startswith("m-")]
    latest = max(models, key=os.path.getmtime)
    return mlflow.pyfunc.load_model(f"file://{os.path.abspath(latest)}"), latest

def registered_model():
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "HousingPricePredictor")
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    model_uri = f"models:/{model_name}/{latest_version.version}"
    return mlflow.pyfunc.load_model(model_uri), model_uri

@app.get("/health")
def health(): return {"status": "ok", "model_path": load_model()[1]}

@app.post("/predict")
def predict(data: InputData):
    model = load_model()[0]
    # model = registered_model()[0]
    try: 
        df = pd.DataFrame([{"area": data.area}])
        prediction = model.predict(df)[0]
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
