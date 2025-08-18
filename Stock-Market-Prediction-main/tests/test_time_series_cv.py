"""
Unit tests for time series cross-validation
Tests walk-forward validation, data leakage prevention, and proper evaluation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from stock_prediction.data.time_series_cv import TimeSeriesSplit, WalkForwardValidator


class TestTimeSeriesSplit:
    """Test the TimeSeriesSplit class"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.n_splits = 5
        self.test_size = 20
        
    def test_time_series_split_initialization(self):
        """Test TimeSeriesSplit initialization"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        assert tscv.n_splits == self.n_splits
        assert tscv.test_size == self.test_size
        assert tscv.gap == 0  # Default gap
        assert tscv.expand_window is True  # Default expand_window
    
    def test_time_series_split_generate_splits(self):
        """Test split generation"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        splits = list(tscv.split(self.X, self.y))
        
        # Check number of splits
        assert len(splits) == self.n_splits
        
        # Check each split
        for i, (train_idx, test_idx) in enumerate(splits):
            # Check that train and test are arrays
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            
            # Check that test size is correct
            assert len(test_idx) == self.test_size
            
            # Check that train size increases (expand_window=True)
            if i > 0:
                prev_train_size = len(splits[i-1][0])
                curr_train_size = len(train_idx)
                assert curr_train_size >= prev_train_size
            
            # Check that no overlap between train and test
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set.intersection(test_set)) == 0
            
            # Check that test comes after train (time order)
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert train_idx.max() < test_idx.min()
    
    def test_time_series_split_with_gap(self):
        """Test split generation with gap"""
        gap = 5
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=gap)
        splits = list(tscv.split(self.X, self.y))
        
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Check that gap is maintained
                min_gap = test_idx.min() - train_idx.max()
                assert min_gap >= gap
    
    def test_time_series_split_no_expand_window(self):
        """Test split generation without expanding window"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, expand_window=False)
        splits = list(tscv.split(self.X, self.y))
        
        # Check that train size remains constant
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert len(set(train_sizes)) == 1  # All train sizes should be the same
    
    def test_time_series_split_edge_cases(self):
        """Test edge cases for split generation"""
        # Test with very small data
        small_X = np.random.randn(30, 5)
        small_y = np.random.randn(30)
        
        tscv = TimeSeriesSplit(n_splits=2, test_size=5)
        splits = list(tscv.split(small_X, small_y))
        
        assert len(splits) == 2
        
        # Test with large number of splits
        tscv = TimeSeriesSplit(n_splits=10, test_size=5)
        splits = list(tscv.split(self.X, self.y))
        
        assert len(splits) == 10
    
    def test_time_series_split_get_n_splits(self):
        """Test get_n_splits method"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        assert tscv.get_n_splits() == self.n_splits
    
    def test_time_series_split_data_integrity(self):
        """Test that data integrity is maintained through splits"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        splits = list(tscv.split(self.X, self.y))
        
        # Check that all indices are within bounds
        for train_idx, test_idx in splits:
            assert train_idx.max() < len(self.X)
            assert test_idx.max() < len(self.X)
            assert train_idx.min() >= 0
            assert test_idx.min() >= 0


class TestWalkForwardValidator:
    """Test the WalkForwardValidator class"""
    
    def setup_method(self):
        """Set up test data and mock model"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.n_splits = 3
        self.test_size = 20
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.fit.return_value = None
        self.mock_model.predict.return_value = np.random.randn(20)
        
    def test_walk_forward_validator_initialization(self):
        """Test WalkForwardValidator initialization"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=0,
            min_train_size=50
        )
        
        assert validator.n_splits == self.n_splits
        assert validator.test_size == self.test_size
        assert validator.gap == 0
        assert validator.min_train_size == 50
    
    def test_walk_forward_validator_validate_model(self):
        """Test model validation process"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        results = validator.validate_model(self.mock_model, self.X, self.y)
        
        # Check that results contain expected keys
        assert 'fold_metrics' in results
        assert 'overall_metrics' in results
        assert 'training_times' in results
        
        # Check fold metrics
        fold_metrics = results['fold_metrics']
        assert len(fold_metrics) == self.n_splits
        
        # Check overall metrics
        overall_metrics = results['overall_metrics']
        expected_metrics = ['r2_mean', 'r2_std', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std']
        for metric in expected_metrics:
            assert metric in overall_metrics
        
        # Check training times
        training_times = results['training_times']
        assert len(training_times) == self.n_splits
        assert all(isinstance(t, float) for t in training_times)
    
    def test_walk_forward_validator_model_fitting(self):
        """Test that model is fitted correctly for each fold"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Track fit calls
        fit_calls = []
        predict_calls = []
        
        def mock_fit(X, y):
            fit_calls.append((X, y))
            return None
        
        def mock_predict(X):
            predict_calls.append(X)
            return np.random.randn(len(X))
        
        self.mock_model.fit.side_effect = mock_fit
        self.mock_model.predict.side_effect = mock_predict
        
        validator.validate_model(self.mock_model, self.X, self.y)
        
        # Check that model was fitted for each fold
        assert len(fit_calls) == self.n_splits
        
        # Check that predictions were made for each validation set
        assert len(predict_calls) == self.n_splits
    
    def test_walk_forward_validator_metrics_calculation(self):
        """Test that metrics are calculated correctly"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Create deterministic predictions
        def mock_predict(X):
            return np.ones(len(X)) * 100  # Constant prediction
        
        self.mock_model.predict.side_effect = mock_predict
        
        results = validator.validate_model(self.mock_model, self.X, self.y)
        
        # Check that metrics are reasonable
        overall_metrics = results['overall_metrics']
        
        # With constant predictions, R² should be very low
        assert overall_metrics['r2_mean'] < 0.1
        
        # RMSE and MAE should be positive
        assert overall_metrics['rmse_mean'] > 0
        assert overall_metrics['mae_mean'] > 0
    
    def test_walk_forward_validator_no_data_leakage(self):
        """Test that no data leakage occurs during validation"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Track which data was used for training and validation
        train_indices = []
        test_indices = []
        
        def mock_fit(X, y):
            # Store training indices
            train_indices.append(set(range(len(X))))
            return None
        
        def mock_predict(X):
            # Store validation indices
            test_indices.append(set(range(len(X))))
            return np.random.randn(len(X))
        
        self.mock_model.fit.side_effect = mock_fit
        self.mock_model.predict.side_effect = mock_predict
        
        validator.validate_model(self.mock_model, self.X, self.y)
        
        # Check that no overlap between train and test sets
        for i in range(len(train_indices)):
            for j in range(len(test_indices)):
                if i != j:  # Different folds
                    # Train and test sets should be disjoint
                    assert len(train_indices[i].intersection(test_indices[j])) == 0
    
    def test_walk_forward_validator_error_handling(self):
        """Test error handling during validation"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Test with model that raises error during fit
        error_model = Mock()
        error_model.fit.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception, match="Training failed"):
            validator.validate_model(error_model, self.X, self.y)
    
    def test_walk_forward_validator_feature_names(self):
        """Test validation with feature names"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
        
        results = validator.validate_model(
            self.mock_model, self.X, self.y, feature_names
        )
        
        # Check that results contain feature information if available
        assert 'fold_metrics' in results
        assert 'overall_metrics' in results
    
    def test_walk_forward_validator_performance_tracking(self):
        """Test that training times are tracked correctly"""
        validator = WalkForwardValidator(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Simulate different training times
        training_times = [1.0, 2.0, 3.0]
        time_index = 0
        
        def mock_fit(X, y):
            import time
            time.sleep(training_times[time_index])
            return None
        
        self.mock_model.fit.side_effect = mock_fit
        
        results = validator.validate_model(self.mock_model, self.X, self.y)
        
        # Check that training times are recorded
        assert 'training_times' in results
        training_times_recorded = results['training_times']
        assert len(training_times_recorded) == self.n_splits
        
        # Check that times are reasonable (accounting for sleep)
        assert all(t >= 0.9 for t in training_times_recorded)  # At least 0.9s due to sleep
    
    def test_walk_forward_validator_edge_cases(self):
        """Test edge cases for validation"""
        # Test with very small data
        small_X = np.random.randn(30, 5)
        small_y = np.random.randn(30)
        
        validator = WalkForwardValidator(n_splits=2, test_size=5)
        results = validator.validate_model(self.mock_model, small_X, small_y)
        
        assert len(results['fold_metrics']) == 2
        
        # Test with large number of splits
        validator = WalkForwardValidator(n_splits=10, test_size=5)
        results = validator.validate_model(self.mock_model, self.X, self.y)
        
        assert len(results['fold_metrics']) == 10


class TestTimeSeriesCVIntegration:
    """Test integration between TimeSeriesSplit and WalkForwardValidator"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        # Create splits
        tscv = TimeSeriesSplit(n_splits=3, test_size=20)
        splits = list(tscv.split(self.X, self.y))
        
        # Create validator
        validator = WalkForwardValidator(n_splits=3, test_size=20)
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randn(20)
        
        # Run validation
        results = validator.validate_model(mock_model, self.X, self.y)
        
        # Check consistency
        assert len(results['fold_metrics']) == len(splits)
        assert len(results['training_times']) == len(splits)
        
        # Check that splits match
        for i, (train_idx, test_idx) in enumerate(splits):
            assert len(train_idx) > 0
            assert len(test_idx) == 20
    
    def test_reproducibility(self):
        """Test that validation results are reproducible"""
        validator = WalkForwardValidator(n_splits=3, test_size=20)
        
        # Mock model with deterministic predictions
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.ones(20) * 100
        
        # Run validation twice
        results_1 = validator.validate_model(mock_model, self.X, self.y)
        results_2 = validator.validate_model(mock_model, self.X, self.y)
        
        # Results should be identical
        assert results_1['overall_metrics']['r2_mean'] == results_2['overall_metrics']['r2_mean']
        assert results_1['overall_metrics']['rmse_mean'] == results_2['overall_metrics']['rmse_mean']


if __name__ == "__main__":
    pytest.main([__file__])
