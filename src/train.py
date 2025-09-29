import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn

def main():
    # Load dataset
    df = pd.read_csv("data/housing.csv")
    X = df[["area"]]
    y = df["price"]

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Set MLflow tracking
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file:///{tracking_dir.replace(os.sep, '/')}")
    mlflow.set_experiment("Housing-Regression")

    # Start MLflow run
    with mlflow.start_run(run_name="linear_regression") as run:
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate metrics
        r2 = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"R^2: {r2}, MSE: {mse}, RMSE: {rmse}, MAE: {mae}")

        # Log params
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "LinearRegression")

        # Log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[:5].astype(float)
        )

        # Plot prediction vs actual
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction vs Actual")
        plt.savefig("prediction_plot.png")
        mlflow.log_artifact("prediction_plot.png", artifact_path="plots")

        joblib.dump(model, "model.pkl")
        print("Run ID:", run.info.run_id)
        print("Model saved and logged with MLflow!")


if __name__ == "__main__":
    main()
