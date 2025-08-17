"""
Prediction API for stock prediction models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import logging
from pathlib import Path

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.lstm_model import LSTMModel
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Prediction API",
    description="API for predicting stock prices using various ML models",
    version="1.0.0"
)

# Global variables for loaded models
models = {}
data_processor = None

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_type: str  # "lstm", "linear_regression", "random_forest", "xgboost", "lightgbm"
    data: Dict[str, Any]  # Stock data in dictionary format

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_type: str
    prediction: float
    confidence: Optional[float] = None
    model_info: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    available_models: list
    model_details: Dict[str, Any]

def load_models(models_dir: str = "models"):
    """Load all available trained models"""
    global models, data_processor
    
    try:
        models_path = Path(models_dir)
        if not models_path.exists():
            logger.warning(f"Models directory {models_dir} does not exist")
            return
            
        # Initialize data processor
        data_processor = DataProcessor(sequence_length=60)
        
        # Load LSTM model if available
        lstm_path = models_path / "best_model.h5"
        if lstm_path.exists():
            try:
                lstm_model = LSTMModel(sequence_length=60, n_features=1)
                lstm_model.load_weights(str(lstm_path))
                models["lstm"] = lstm_model
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LSTM model: {str(e)}")
        
        # Load ML models if available
        ml_model_patterns = {
            "linear_regression": "best_linear_regression.joblib",
            "random_forest": "best_random_forest.joblib",
            "xgboost": "best_xgboost.joblib",
            "lightgbm": "best_lightgbm.joblib"
        }
        
        for model_type, pattern in ml_model_patterns.items():
            model_path = models_path / pattern
            if model_path.exists():
                try:
                    if model_type == "linear_regression":
                        model = LinearRegressionModel()
                    elif model_type == "random_forest":
                        model = RandomForestModel()
                    elif model_type == "xgboost":
                        model = XGBoostModel()
                    elif model_type == "lightgbm":
                        model = LightGBMModel()
                    
                    model.load_model(str(model_path))
                    models[model_type] = model
                    logger.info(f"{model_type} model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {str(e)}")
        
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Market Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/models", response_model=ModelInfoResponse)
async def get_models():
    """Get information about available models"""
    model_details = {}
    
    for model_type, model in models.items():
        if model_type == "lstm":
            model_details[model_type] = {
                "type": "LSTM Neural Network",
                "architecture": "Sequential LSTM with dropout",
                "sequence_length": 60,
                "features": 1
            }
        else:
            model_details[model_type] = {
                "type": model.model_name,
                "parameters": {
                    "n_estimators": getattr(model, 'n_estimators', None),
                    "max_depth": getattr(model, 'max_depth', None),
                    "learning_rate": getattr(model, 'learning_rate', None)
                }
            }
    
    return ModelInfoResponse(
        available_models=list(models.keys()),
        model_details=model_details
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the specified model"""
    try:
        if request.model_type not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model type '{request.model_type}' not available. Available models: {list(models.keys())}"
            )
        
        model = models[request.model_type]
        
        # Convert request data to DataFrame
        df = pd.DataFrame([request.data])
        
        if request.model_type == "lstm":
            # Prepare data for LSTM
            X = data_processor.prepare_prediction_data(df)
            prediction = model.predict(X)[0][0]
            
            # Inverse transform to get actual price
            prediction = data_processor.inverse_transform([[prediction]])[0][0]
            
        else:
            # Prepare data for ML models
            df_with_features = data_processor.create_features(df)
            X, _ = data_processor.prepare_ml_data(df_with_features)
            prediction = model.predict(X)[0]
        
        # Get model info
        if request.model_type == "lstm":
            model_info = {
                "type": "LSTM Neural Network",
                "sequence_length": 60,
                "features": 1
            }
        else:
            model_info = {
                "type": model.model_name,
                "parameters": {
                    "n_estimators": getattr(model, 'n_estimators', None),
                    "max_depth": getattr(model, 'max_depth', None),
                    "learning_rate": getattr(model, 'learning_rate', None)
                }
            }
        
        return PredictionResponse(
            model_type=request.model_type,
            prediction=float(prediction),
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(
    model_type: str,
    file: UploadFile = File(...)
):
    """Make batch predictions from uploaded CSV file"""
    try:
        if model_type not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model type '{model_type}' not available. Available models: {list(models.keys())}"
            )
        
        # Read CSV file
        df = pd.read_csv(file.file)
        model = models[model_type]
        
        if model_type == "lstm":
            # Prepare data for LSTM
            X = data_processor.prepare_prediction_data(df)
            predictions = model.predict(X)
            
            # Inverse transform
            predictions = data_processor.inverse_transform(predictions)
            
        else:
            # Prepare data for ML models
            df_with_features = data_processor.create_features(df)
            X, _ = data_processor.prepare_ml_data(df_with_features)
            predictions = model.predict(X)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted_price'] = predictions.flatten()
        
        # Save results
        results_path = f"batch_predictions_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_path, index=False)
        
        return {
            "message": f"Batch predictions completed for {len(df)} samples",
            "model_type": model_type,
            "results_file": results_path,
            "sample_predictions": predictions.flatten()[:5].tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.post("/reload_models")
async def reload_models():
    """Reload all models from disk"""
    try:
        load_models()
        return {
            "message": "Models reloaded successfully",
            "models_loaded": len(models),
            "available_models": list(models.keys())
        }
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
