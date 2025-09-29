import os, pandas as pd, mlflow.pyfunc, logging
from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
load_dotenv()  # Loads variables from .env into environment

app = FastAPI()
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

LOG_FOLDER = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(LOG_FOLDER, "logs.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME")



class InputData(BaseModel):
    area: float

# def load_local_model():
#     exp = client.get_experiment_by_name(EXPERIMENT_NAME)
#     models = [os.path.join("mlruns", exp.experiment_id, "models", d, "artifacts")
#               for d in os.listdir(os.path.join("mlruns", exp.experiment_id, "models"))
#               if d.startswith("m-")]
#     latest = max(models, key=os.path.getmtime)
#     return mlflow.pyfunc.load_model(f"file://{os.path.abspath(latest)}"), latest

def load_registered_model():
    print("Using load_registered_model()")
    logging.info("Using load_registered_model()")
    latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version.version}"
    return mlflow.pyfunc.load_model(model_uri)

@app.get("/health")
def health(): 
    logging.info("Health check requested")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        model = load_registered_model()
        area = data.area
        if area is None:
            logging.error(f"BAD INPUT - Missing 'area': {data}")
            raise HTTPException(status_code=422, detail="Missing 'area' field")
        if not isinstance(area, (int, float)):
            logging.error(f"BAD INPUT - Non-numeric 'area': {data}")
            raise HTTPException(status_code=422, detail="'area' must be a number")
        if area <= 0:
            logging.error(f"BAD INPUT - Non-positive 'area': {data}")
            raise HTTPException(status_code=422, detail="'area' must be positive")
        
        df = pd.DataFrame([{"area": area}])
        prediction = model.predict(df)[0]

        # Log successful request
        logging.info(f"SUCCESS - Input: {data.area}, Prediction: {prediction}")
        return {"prediction": prediction}

    except HTTPException:
        # Already logged above
        raise
    except Exception as e:
        logging.error(f"ERROR - Input: {area}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the bad input and the validation errors
    try:
        body = await request.json()
    except:
        body = "<could not read body>"
    logging.error(f"BAD INPUT - Path: {request.url.path}, Body: {body}, Errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body},
    )


