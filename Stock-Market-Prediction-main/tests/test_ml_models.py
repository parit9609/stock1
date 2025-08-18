"""
Unit tests for ML models
Tests all model classes, validation methods, and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from stock_prediction.models.ml_models import (
    BaseMLModel, LinearRegressionModel, RandomForestModel,
    XGBoostModel, LightGBMModel, ModelBenchmark
)


class TestBaseMLModel:
    """Test the abstract base class functionality"""
    
    def test_base_model_initialization(self):
        """Test base model initialization with different scaling options"""
        # Test with scaling
        model = LinearRegressionModel()
        assert model.needs_scaling is True
        assert model.scaler is not None
        assert model.is_fitted is False
        
        # Test without scaling
        model = RandomForestModel()
        assert model.needs_scaling is False
        assert model.scaler is None
        assert model.is_fitted is False
    
    def test_base_model_abstract_methods(self):
        """Test that abstract methods are properly implemented"""
        # LinearRegressionModel should implement _build_model
        model = LinearRegressionModel()
        assert hasattr(model, '_build_model')
        assert callable(model._build_model)
        
        # RandomForestModel should implement _build_model
        model = RandomForestModel()
        assert hasattr(model, '_build_model')
        assert callable(model._build_model)


class TestLinearRegressionModel:
    """Test Linear Regression model implementation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.model = LinearRegressionModel()
    
    def test_linear_regression_build_model(self):
        """Test model building"""
        model_instance = self.model._build_model()
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
        assert hasattr(model_instance, 'predict')
    
    def test_linear_regression_fit(self):
        """Test model fitting"""
        metrics = self.model.fit(self.X, self.y)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
        assert isinstance(metrics['train_mse'], float)
        assert isinstance(metrics['train_mae'], float)
        assert isinstance(metrics['train_r2'], float)
    
    def test_linear_regression_predict(self):
        """Test model prediction"""
        # Fit first
        self.model.fit(self.X, self.y)
        
        # Test prediction
        X_test = np.random.randn(10, 5)
        predictions = self.model.predict(X_test)
        
        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)
    
    def test_linear_regression_predict_unfitted(self):
        """Test prediction on unfitted model raises error"""
        X_test = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Linear Regression model is not fitted yet"):
            self.model.predict(X_test)
    
    def test_linear_regression_evaluate(self):
        """Test model evaluation"""
        # Fit first
        self.model.fit(self.X, self.y)
        
        # Test evaluation
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        metrics = self.model.evaluate(X_test, y_test)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert isinstance(metrics['mse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['r2'], float)
        assert isinstance(metrics['rmse'], float)
    
    def test_linear_regression_scaling(self):
        """Test that scaling is applied correctly"""
        # Create data with different scales
        X_scaled = np.random.randn(100, 5)
        X_unscaled = X_scaled * 1000  # Different scale
        
        # Fit with scaled data
        self.model.fit(X_scaled, self.y)
        pred_scaled = self.model.predict(X_scaled)
        
        # Fit with unscaled data
        model2 = LinearRegressionModel()
        model2.fit(X_unscaled, self.y)
        pred_unscaled = model2.predict(X_unscaled)
        
        # Predictions should be similar (scaling handled internally)
        assert np.allclose(pred_scaled, pred_unscaled, rtol=1e-10)
    
    def test_linear_regression_save_load(self):
        """Test model saving and loading"""
        # Fit model
        self.model.fit(self.X, self.y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            self.model.save_model(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Load model
            new_model = LinearRegressionModel()
            new_model.load_model(tmp_path)
            
            # Test predictions are the same
            X_test = np.random.randn(10, 5)
            pred_original = self.model.predict(X_test)
            pred_loaded = new_model.predict(X_test)
            
            assert np.allclose(pred_original, pred_loaded)
            assert new_model.is_fitted is True
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestRandomForestModel:
    """Test Random Forest model implementation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.model = RandomForestModel(n_estimators=10, random_state=42)
    
    def test_random_forest_build_model(self):
        """Test model building"""
        model_instance = self.model._build_model()
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
        assert hasattr(model_instance, 'predict')
        assert model_instance.n_estimators == 10
        assert model_instance.random_state == 42
    
    def test_random_forest_fit(self):
        """Test model fitting"""
        metrics = self.model.fit(self.X, self.y)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
    
    def test_random_forest_no_scaling(self):
        """Test that no scaling is applied"""
        assert self.model.needs_scaling is False
        assert self.model.scaler is None
        
        # Should work without scaling
        metrics = self.model.fit(self.X, self.y)
        assert self.model.is_fitted is True
        
        # Test prediction
        X_test = np.random.randn(10, 5)
        predictions = self.model.predict(X_test)
        assert predictions.shape == (10,)


class TestXGBoostModel:
    """Test XGBoost model implementation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.model = XGBoostModel(
            n_estimators=10, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=42,
            early_stopping_rounds=5
        )
    
    def test_xgboost_build_model(self):
        """Test model building"""
        model_instance = self.model._build_model()
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
        assert hasattr(model_instance, 'predict')
        assert model_instance.n_estimators == 10
        assert model_instance.max_depth == 3
        assert model_instance.learning_rate == 0.1
    
    def test_xgboost_fit(self):
        """Test model fitting"""
        metrics = self.model.fit(self.X, self.y)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
    
    def test_xgboost_fit_with_validation(self):
        """Test model fitting with validation set for early stopping"""
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)
        
        metrics = self.model.fit(self.X, self.y, X_val, y_val)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
    
    def test_xgboost_no_scaling(self):
        """Test that no scaling is applied"""
        assert self.model.needs_scaling is False
        assert self.model.scaler is None


class TestLightGBMModel:
    """Test LightGBM model implementation"""
    
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        self.model = LightGBMModel(
            n_estimators=10, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=42,
            early_stopping_rounds=5,
            num_leaves=10
        )
    
    def test_lightgbm_build_model(self):
        """Test model building"""
        model_instance = self.model._build_model()
        assert model_instance is not None
        assert hasattr(model_instance, 'fit')
        assert hasattr(model_instance, 'predict')
        assert model_instance.n_estimators == 10
        assert model_instance.max_depth == 3
        assert model_instance.learning_rate == 0.1
        assert model_instance.num_leaves == 10
    
    def test_lightgbm_fit(self):
        """Test model fitting"""
        metrics = self.model.fit(self.X, self.y)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'train_r2' in metrics
    
    def test_lightgbm_fit_with_validation(self):
        """Test model fitting with validation set for early stopping"""
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20)
        
        metrics = self.model.fit(self.X, self.y, X_val, y_val)
        
        assert self.model.is_fitted is True
        assert 'train_mse' in metrics
    
    def test_lightgbm_no_scaling(self):
        """Test that no scaling is applied"""
        assert self.model.needs_scaling is False
        assert self.model.scaler is None


class TestModelBenchmark:
    """Test the ModelBenchmark class"""
    
    def setup_method(self):
        """Set up test data and models"""
        np.random.seed(42)
        self.X_train = np.random.randn(80, 5)
        self.y_train = np.random.randn(80)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20)
        
        self.benchmark = ModelBenchmark()
        self.benchmark.add_model(LinearRegressionModel())
        self.benchmark.add_model(RandomForestModel(n_estimators=10, random_state=42))
    
    def test_benchmark_add_model(self):
        """Test adding models to benchmark"""
        assert len(self.benchmark.models) == 2
        assert 'Linear Regression' in self.benchmark.models
        assert 'Random Forest' in self.benchmark.models
    
    def test_benchmark_run_benchmark(self):
        """Test running benchmark"""
        results = self.benchmark.run_benchmark(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # Two models
        assert 'Model' in results.columns
        assert 'Test_R2' in results.columns
        assert 'Test_RMSE' in results.columns
    
    def test_benchmark_get_best_model(self):
        """Test getting best model"""
        # Run benchmark first
        self.benchmark.run_benchmark(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        best_model, best_score = self.benchmark.get_best_model('Test_R2')
        assert isinstance(best_model, str)
        assert isinstance(best_score, float)
        assert best_model in ['Linear Regression', 'Random Forest']
    
    def test_benchmark_save_results(self):
        """Test saving benchmark results"""
        # Run benchmark first
        self.benchmark.run_benchmark(
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        # Save results
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            self.benchmark.save_results(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Check file exists and has content
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            
            # Load and verify content
            saved_results = pd.read_csv(tmp_path)
            assert len(saved_results) == 2
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data(self):
        """Test handling of empty data"""
        model = LinearRegressionModel()
        
        with pytest.raises(ValueError):
            model.fit(np.array([]), np.array([]))
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions"""
        model = LinearRegressionModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(99)  # Mismatched length
        
        with pytest.raises(ValueError):
            model.fit(X, y)
    
    def test_nan_values(self):
        """Test handling of NaN values"""
        model = LinearRegressionModel()
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan  # Introduce NaN
        y = np.random.randn(100)
        
        with pytest.raises(ValueError):
            model.fit(X, y)
    
    def test_inf_values(self):
        """Test handling of infinite values"""
        model = LinearRegressionModel()
        X = np.random.randn(100, 5)
        X[0, 0] = np.inf  # Introduce inf
        y = np.random.randn(100)
        
        with pytest.raises(ValueError):
            model.fit(X, y)


if __name__ == "__main__":
    pytest.main([__file__])
