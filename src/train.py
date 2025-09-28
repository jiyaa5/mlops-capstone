import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

def main():

    df = pd.read_csv("data/housing.csv")
    X = df[["area"]]
    y = df["price"]

    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state
    )

    # Connect to MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
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

        # Save model locally
        model_path = "model.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train[:5].astype(float)  # avoid integer schema warning
        )

        # Log local pickle as artifact

        # Plot prediction vs actual and log
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction vs Actual")
        plt.savefig("prediction_plot.png")
        mlflow.log_artifact("prediction_plot.png", artifact_path="plots")

        print("Run ID:", run.info.run_id)
        print("Model saved and logged with MLflow!")

if __name__ == "__main__":
    main()
