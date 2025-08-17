"""
Tests for LSTM model module
"""

import pytest
import numpy as np
from stock_prediction.models.lstm_model import LSTMModel

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    sequence_length = 60
    n_features = 1
    n_samples = 100
    
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples, 1)
    
    return X, y

@pytest.fixture
def model():
    """Create LSTMModel instance"""
    return LSTMModel(sequence_length=60, n_features=1)

def test_model_initialization(model):
    """Test model initialization"""
    assert model.sequence_length == 60
    assert model.n_features == 1
    assert model.model is not None

def test_model_architecture(model):
    """Test model architecture"""
    # Check input shape
    assert model.model.input_shape == (None, 60, 1)
    
    # Check output shape
    assert model.model.output_shape == (None, 1)
    
    # Check number of layers
    assert len(model.model.layers) == 7  # 3 LSTM, 3 Dropout, 1 Dense

def test_model_training(model, sample_data):
    """Test model training"""
    X, y = sample_data
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=2,
        batch_size=32
    )
    
    # Check if training history contains expected metrics
    assert 'loss' in history
    assert 'val_loss' in history
    assert len(history['loss']) == 2
    assert len(history['val_loss']) == 2

def test_model_prediction(model, sample_data):
    """Test model prediction"""
    X, _ = sample_data
    
    # Make prediction
    predictions = model.predict(X[:1])
    
    # Check prediction shape
    assert predictions.shape == (1, 1)

def test_model_evaluation(model, sample_data):
    """Test model evaluation"""
    X, y = sample_data
    
    # Evaluate model
    loss = model.evaluate(X, y)
    
    # Check if loss is a float
    assert isinstance(loss, float) 