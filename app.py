from fastapi import FastAPI
import joblib

# TODO: Load model from MLflow instead of local pkl
model = joblib.load("model.pkl")

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    # TODO: Handle proper preprocessing
    area = data["area"]
    prediction = model.predict([[area]])[0]
    return {"prediction": prediction}
