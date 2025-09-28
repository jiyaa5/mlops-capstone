FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy mlruns folder
COPY mlruns/ /app/mlruns/

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
