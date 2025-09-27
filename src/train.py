import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn

# TODO: Add MLflow tracking params, metrics, and model logging

def main():
    df = pd.read_csv("data/housing.csv")
    X = df[["area"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print("Model R^2 Score:", score)

    # Save model locally
    joblib.dump(model, "model.pkl")

    # TODO: Log params, metrics, and model with MLflow

if __name__ == "__main__":
    main()
