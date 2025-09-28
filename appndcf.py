import os
import glob
from functools import lru_cache
from typing import Any, Dict
import logging

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Housing Regression Service", version="0.1.0")


class InputData(BaseModel):
    area: float


@lru_cache(maxsize=1)
def _load_model_bundle() -> Dict[str, Any]:
    """
    Load the MLflow model directly from the known structure:
    mlruns/286662785541361765/models/m-*/artifacts/
    """
    experiment_id = "286662785541361765"
    
    # Set tracking URI (though we won't use MLflow client)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"Looking for model in experiment: {experiment_id}")
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Build the models directory path - check both Docker and local paths
    if os.path.exists("/app/mlruns"):
        # Running in Docker
        models_base_path = f"/app/mlruns/{experiment_id}/models"
    elif os.path.exists("mlruns"):
        # Running locally
        models_base_path = f"mlruns/{experiment_id}/models"
    else:
        raise RuntimeError("Neither /app/mlruns nor ./mlruns directory found")
    logger.info(f"Models base path: {models_base_path}")
    
    if not os.path.exists(models_base_path):
        raise RuntimeError(f"Models directory not found: {models_base_path}")
    
    # Find all model directories (pattern: m-*)
    model_pattern = f"{models_base_path}/m-*"
    model_dirs = glob.glob(model_pattern)
    
    logger.info(f"Found model directories: {model_dirs}")
    
    if not model_dirs:
        raise RuntimeError(f"No model directories found matching pattern: {model_pattern}")
    
    # Sort to get the latest model (assuming newer models have later timestamps in ID)
    model_dirs.sort(reverse=True)
    latest_model_dir = model_dirs[0]
    
    # Build artifacts path
    artifacts_path = f"{latest_model_dir}/artifacts"
    logger.info(f"Using model artifacts path: {artifacts_path}")
    
    if not os.path.exists(artifacts_path):
        raise RuntimeError(f"Artifacts directory not found: {artifacts_path}")
    
    # List artifacts for debugging
    artifacts = os.listdir(artifacts_path)
    logger.info(f"Available artifacts: {artifacts}")
    
    # Check for essential model files
    essential_files = ["MLmodel"]
    missing_files = [f for f in essential_files if f not in artifacts]
    if missing_files:
        logger.warning(f"Missing essential files: {missing_files}")
    
    try:
        # Load the model directly from artifacts path
        logger.info(f"Loading model from: {artifacts_path}")
        model = mlflow.pyfunc.load_model(artifacts_path)
        logger.info("✅ Model loaded successfully!")
        
        # Extract model ID from path
        model_id = os.path.basename(latest_model_dir)
        
        return {
            "model": model,
            "model_id": model_id,
            "model_path": artifacts_path,
            "experiment_id": experiment_id,
            "artifacts": artifacts
        }
        
    except Exception as e:
        logger.error(f"Failed to load model from {artifacts_path}: {e}")
        
        # Try alternative loading methods
        alternative_paths = [
            artifacts_path,  # Direct artifacts path
            latest_model_dir,  # Model directory
        ]
        
        for alt_path in alternative_paths:
            try:
                logger.info(f"Trying alternative path: {alt_path}")
                model = mlflow.pyfunc.load_model(alt_path)
                logger.info(f"✅ Model loaded from alternative path: {alt_path}")
                
                model_id = os.path.basename(latest_model_dir)
                return {
                    "model": model,
                    "model_id": model_id,
                    "model_path": alt_path,
                    "experiment_id": experiment_id,
                    "artifacts": artifacts
                }
            except Exception as alt_e:
                logger.warning(f"Alternative path {alt_path} failed: {alt_e}")
                continue
        
        raise RuntimeError(f"Failed to load model from any available path. Last error: {e}")


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check."""
    try:
        bundle = _load_model_bundle()
        return {
            "status": "ok",
            "model_id": bundle["model_id"],
            "experiment_id": bundle["experiment_id"],
            "model_path": bundle["model_path"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.post("/predict")
def predict(data: InputData) -> Dict[str, Any]:
    """Predict housing price from input features."""
    try:
        bundle = _load_model_bundle()
        model = bundle["model"]

        # Create input DataFrame
        df = pd.DataFrame([{"area": data.area}])
        logger.info(f"Input data: area={data.area}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        logger.info(f"Prediction: {prediction}")
        
        return {
            "prediction": float(prediction),
            "model_id": bundle["model_id"],
            "input_area": data.area
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    try:
        bundle = _load_model_bundle()
        return {
            "model_id": bundle["model_id"],
            "experiment_id": bundle["experiment_id"],
            "model_path": bundle["model_path"],
            "artifacts": bundle["artifacts"],
            "status": "loaded"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug")
def debug_structure() -> Dict[str, Any]:
    """Debug endpoint to show the actual directory structure."""
    experiment_id = "286662785541361765"
    
    # Check both possible base paths
    possible_bases = ["/app/mlruns", "mlruns"]
    base_path = None
    
    for path in possible_bases:
        if os.path.exists(path):
            base_path = path
            break
    
    result = {
        "possible_base_paths": possible_bases,
        "selected_base_path": base_path,
        "experiment_id": experiment_id,
        "current_working_directory": os.getcwd(),
        "directory_contents": os.listdir(".") if os.path.exists(".") else []
    }
    
    
    if base_path is None:
        result["error"] = "No mlruns directory found"
        return result
    
    # Check experiment directory
    exp_path = f"{base_path}/{experiment_id}"
    result["experiment_path"] = exp_path
    result["experiment_exists"] = os.path.exists(exp_path)
    
    if os.path.exists(exp_path):
        result["experiment_contents"] = os.listdir(exp_path)
        
        # Check models directory
        models_path = f"{exp_path}/models"
        result["models_path"] = models_path
        result["models_exists"] = os.path.exists(models_path)
        
        if os.path.exists(models_path):
            model_dirs = os.listdir(models_path)
            result["model_directories"] = model_dirs
            
            # Check first model directory
            if model_dirs:
                first_model = model_dirs[0]
                first_model_path = f"{models_path}/{first_model}"
                result["first_model_path"] = first_model_path
                result["first_model_contents"] = os.listdir(first_model_path)
                
                # Check artifacts
                artifacts_path = f"{first_model_path}/artifacts"
                if os.path.exists(artifacts_path):
                    result["artifacts"] = os.listdir(artifacts_path)
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)