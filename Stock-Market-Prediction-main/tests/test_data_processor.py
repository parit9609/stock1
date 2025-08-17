"""
Tests for data processor module
"""

import pytest
import pandas as pd
import numpy as np
from stock_prediction.data.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    df = pd.DataFrame({
        'Close': prices
    }, index=dates)
    return df

@pytest.fixture
def data_processor():
    """Create DataProcessor instance"""
    return DataProcessor(sequence_length=10)

def test_prepare_data(data_processor, sample_data):
    """Test data preparation"""
    X, y = data_processor.prepare_data(sample_data)
    
    # Check shapes
    assert len(X.shape) == 3
    assert len(y.shape) == 2
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == data_processor.sequence_length
    assert X.shape[2] == 1

def test_split_data(data_processor, sample_data):
    """Test data splitting"""
    X, y = data_processor.prepare_data(sample_data)
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y, train_ratio=0.8)
    
    # Check shapes
    assert len(X_train) > len(X_test)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def test_prepare_prediction_data(data_processor, sample_data):
    """Test prediction data preparation"""
    X = data_processor.prepare_prediction_data(sample_data)
    
    # Check shape
    assert X.shape[1] == data_processor.sequence_length
    assert X.shape[2] == 1

def test_inverse_transform(data_processor, sample_data):
    """Test inverse transform"""
    # Prepare and scale data
    X, y = data_processor.prepare_data(sample_data)
    
    # Inverse transform
    original_scale = data_processor.inverse_transform(y)
    
    # Check shape
    assert original_scale.shape == y.shape 