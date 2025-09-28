FROM python:3.11-slim

ENV MLFLOW_EXPERIMENT_NAME=Housing-Regression
ENV MLFLOW_REGISTERED_MODEL_NAME=HousingPricePredictor

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
