"""
Unit tests for data processor
Tests data loading, feature engineering, and validation methods
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from stock_prediction.data.data_processor import DataProcessor


class TestDataProcessor:
    """Test the DataProcessor class"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create synthetic stock data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
        
        # Initialize processor
        self.processor = DataProcessor(sequence_length=10)
    
    def test_data_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.sequence_length == 10
        assert hasattr(self.processor, 'create_features')
        assert hasattr(self.processor, 'prepare_ml_data')
        assert hasattr(self.processor, 'create_time_series_splits')
    
    def test_load_data_from_csv(self):
        """Test loading data from CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            self.test_data.to_csv(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load data
            loaded_data = self.processor.load_data(tmp_path)
            
            # Check data integrity
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == 100
            assert 'Open' in loaded_data.columns
            assert 'Close' in loaded_data.columns
            assert 'Volume' in loaded_data.columns
            
            # Check index is datetime
            assert isinstance(loaded_data.index, pd.DatetimeIndex)
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_create_features_technical_indicators(self):
        """Test creation of technical indicators"""
        features_df = self.processor.create_features(self.test_data)
        
        # Check that technical indicators are created
        expected_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_middle'
        ]
        
        for indicator in expected_indicators:
            assert indicator in features_df.columns, f"Missing {indicator}"
        
        # Check RSI values are within valid range
        rsi_values = features_df['rsi'].dropna()
        assert all(0 <= rsi <= 100 for rsi in rsi_values)
        
        # Check Bollinger Bands relationship
        bb_upper = features_df['bb_upper'].dropna()
        bb_lower = features_df['bb_lower'].dropna()
        bb_middle = features_df['bb_middle'].dropna()
        
        assert all(bb_upper >= bb_middle)
        assert all(bb_middle >= bb_lower)
    
    def test_create_features_moving_averages(self):
        """Test creation of moving averages"""
        features_df = self.processor.create_features(self.test_data)
        
        # Check moving averages
        expected_ma = ['ma_5', 'ma_10', 'ma_20', 'ma_50']
        
        for ma in expected_ma:
            assert ma in features_df.columns, f"Missing {ma}"
            
            # Check that MA values are reasonable
            ma_values = features_df[ma].dropna()
            assert len(ma_values) > 0
            
            # Check that MA values are within price range
            close_prices = self.test_data['Close']
            assert all(ma_values >= close_prices.min())
            assert all(ma_values <= close_prices.max())
    
    def test_create_features_volatility(self):
        """Test creation of volatility features"""
        features_df = self.processor.create_features(self.test_data)
        
        # Check volatility features
        expected_vol = ['volatility_5', 'volatility_10', 'volatility_20']
        
        for vol in expected_vol:
            assert vol in features_df.columns, f"Missing {vol}"
            
            # Check that volatility values are positive
            vol_values = features_df[vol].dropna()
            assert all(vol_values >= 0)
    
    def test_create_features_lag_features(self):
        """Test creation of lag features"""
        features_df = self.processor.create_features(self.test_data)
        
        # Check lag features
        expected_lags = ['lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10']
        
        for lag in expected_lags:
            assert lag in features_df.columns, f"Missing {lag}"
            
            # Check that lag features have correct number of NaN values
            lag_values = features_df[lag]
            expected_nans = int(lag.split('_')[1])
            assert lag_values.isna().sum() == expected_nans
    
    def test_create_features_rolling_statistics(self):
        """Test creation of rolling statistics"""
        features_df = self.processor.create_features(self.test_data)
        
        # Check rolling statistics
        expected_rolling = ['rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5']
        
        for rolling in expected_rolling:
            assert rolling in features_df.columns, f"Missing {rolling}"
            
            # Check that rolling values are reasonable
            rolling_values = features_df[rolling].dropna()
            assert len(rolling_values) > 0
    
    def test_create_features_no_data_leakage(self):
        """Test that no data leakage exists in features"""
        features_df = self.processor.create_features(self.test_data)
        
        # This should not raise any exceptions
        self.processor._validate_no_data_leakage(features_df)
        
        # Check that all features use only past data
        # RSI, MACD, Bollinger Bands, MAs, volatility, lags, rolling stats
        # should all be calculated using only past information
        
        # Verify that the first few rows have NaN values (indicating no future data used)
        first_few_rows = features_df.head(20)
        assert first_few_rows.isna().sum().sum() > 0, "Features should have NaN values in early rows"
    
    def test_prepare_ml_data(self):
        """Test preparation of data for ML models"""
        features_df = self.processor.create_features(self.test_data)
        X, y = self.processor.prepare_ml_data(features_df)
        
        # Check data types and shapes
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        
        # Check that X and y have same number of samples
        assert len(X) == len(y)
        
        # Check that no NaN values remain
        assert not np.isnan(X).any(), "X should not contain NaN values"
        assert not np.isnan(y).any(), "y should not contain NaN values"
        
        # Check that X has reasonable number of features
        assert X.shape[1] > 10, "Should have substantial number of features"
        
        # Check that y contains the target variable (Close prices)
        assert y.shape == (len(X),)
    
    def test_create_time_series_splits(self):
        """Test creation of time series splits"""
        features_df = self.processor.create_features(self.test_data)
        X, y = self.processor.prepare_ml_data(features_df)
        
        # Test with different parameters
        n_splits = 3
        test_size = 20
        
        splits = self.processor.create_time_series_splits(X, y, n_splits, test_size)
        
        # Check that correct number of splits are created
        assert len(splits) == n_splits
        
        # Check each split
        for i, (X_train, X_val, y_train, y_val) in enumerate(splits):
            # Check shapes
            assert len(X_train) > len(X_val), f"Split {i}: Train should be larger than validation"
            assert len(y_train) > len(y_val), f"Split {i}: Train targets should be larger than validation targets"
            
            # Check that validation set size is correct
            assert len(X_val) == test_size, f"Split {i}: Validation set size should be {test_size}"
            assert len(y_val) == test_size, f"Split {i}: Validation target size should be {test_size}"
            
            # Check that no data leakage (validation comes after training in time)
            # This is implicit in the walk-forward approach
            
            # Check data types
            assert isinstance(X_train, np.ndarray)
            assert isinstance(X_val, np.ndarray)
            assert isinstance(y_train, np.ndarray)
            assert isinstance(y_val, np.ndarray)
    
    def test_create_time_series_splits_edge_cases(self):
        """Test edge cases for time series splits"""
        features_df = self.processor.create_features(self.test_data)
        X, y = self.processor.prepare_ml_data(features_df)
        
        # Test with very small data
        small_X = X[:30]
        small_y = y[:30]
        
        splits = self.processor.create_time_series_splits(small_X, small_y, n_splits=2, test_size=5)
        assert len(splits) == 2
        
        # Test with large number of splits
        splits = self.processor.create_time_series_splits(X, y, n_splits=10, test_size=5)
        assert len(splits) == 10
    
    def test_data_integrity_through_pipeline(self):
        """Test that data integrity is maintained through the entire pipeline"""
        # Load data
        original_data = self.test_data.copy()
        
        # Create features
        features_df = self.processor.create_features(original_data)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(original_data, self.test_data)
        
        # Prepare ML data
        X, y = self.processor.prepare_ml_data(features_df)
        
        # Check that features data is unchanged
        pd.testing.assert_frame_equal(features_df, self.processor.create_features(original_data))
        
        # Check that X and y correspond to the same time periods
        assert len(X) == len(y)
        
        # Check that no data was lost (accounting for NaN values in early rows)
        expected_samples = len(features_df.dropna())
        assert len(X) == expected_samples
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            self.processor.create_features(empty_df)
        
        # Test with DataFrame missing required columns
        invalid_df = pd.DataFrame({'Date': pd.date_range('2020-01-01', periods=10)})
        with pytest.raises(ValueError):
            self.processor.create_features(invalid_df)
        
        # Test with invalid sequence length
        with pytest.raises(ValueError):
            invalid_processor = DataProcessor(sequence_length=0)
    
    def test_reproducibility(self):
        """Test that feature creation is reproducible"""
        # Create features twice
        features_1 = self.processor.create_features(self.test_data)
        features_2 = self.processor.create_features(self.test_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(features_1, features_2)
        
        # Test with different random seeds (should not affect deterministic features)
        np.random.seed(123)
        features_3 = self.processor.create_features(self.test_data)
        
        # Technical indicators should be identical
        pd.testing.assert_frame_equal(features_1, features_3)


class TestDataValidation:
    """Test data validation methods"""
    
    def setup_method(self):
        """Set up test data"""
        self.processor = DataProcessor(sequence_length=10)
        
        # Create data with known patterns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(150, 250, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.test_data.set_index('Date', inplace=True)
    
    def test_validate_no_data_leakage(self):
        """Test data leakage validation"""
        features_df = self.processor.create_features(self.test_data)
        
        # This should pass without errors
        self.processor._validate_no_data_leakage(features_df)
        
        # Test with artificially introduced data leakage
        features_with_leakage = features_df.copy()
        features_with_leakage['future_price'] = features_with_leakage['Close'].shift(-1)
        
        # This should raise an error
        with pytest.raises(ValueError, match="Data leakage detected"):
            self.processor._validate_no_data_leakage(features_with_leakage)
    
    def test_validate_data_quality(self):
        """Test data quality validation"""
        # Test with clean data
        features_df = self.processor.create_features(self.test_data)
        X, y = self.processor.prepare_ml_data(features_df)
        
        # Check data quality
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        assert not np.isinf(X).any()
        assert not np.isinf(y).any()
        
        # Check reasonable value ranges
        assert X.min() > -1e6  # No extremely negative values
        assert X.max() < 1e6   # No extremely positive values
        assert y.min() > 0      # Stock prices should be positive


if __name__ == "__main__":
    pytest.main([__file__]) 