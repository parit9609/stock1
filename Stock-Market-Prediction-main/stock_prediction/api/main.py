"""
FastAPI application for stock prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.lstm_model import LSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Prediction API",
    description="API for predicting stock prices using LSTM",
    version="1.0.0"
)

# Initialize data processor and model
data_processor = DataProcessor(sequence_length=60)
model = LSTMModel(sequence_length=60)

# Load the model weights
try:
    model_path = Path("models/best_model.h5")
    if model_path.exists():
        model.load_weights(str(model_path))
    else:
        logger.warning("Model weights not found. Please train the model first.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

class StockData(BaseModel):
    dates: List[str]
    prices: List[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Stock Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(stock_data: StockData):
    """
    Predict stock price for the next day
    
    Args:
        stock_data (StockData): Historical stock data
        
    Returns:
        PredictionResponse: Predicted price and confidence
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame({
            'Date': pd.to_datetime(stock_data.dates),
            'Close': stock_data.prices
        }).set_index('Date')
        
        # Prepare data for prediction
        X = data_processor.prepare_prediction_data(df)
        
        # Make prediction
        prediction = model.predict(X)
        
        # Convert prediction to original scale
        prediction = data_processor.inverse_transform(prediction)[0][0]
        
        # Calculate simple confidence based on recent volatility
        recent_std = np.std(stock_data.prices[-30:]) if len(stock_data.prices) >= 30 else np.std(stock_data.prices)
        confidence = max(0, min(1, 1 - (recent_std / np.mean(stock_data.prices))))
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=float(confidence)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 