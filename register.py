import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

result = mlflow.register_model(
    f"runs:/ba072cf5d783427f9c2e07d09f12f54b/model",
    "HousingPricePredictor"
)

print("Registered model:", result.name, "version:", result.version)
