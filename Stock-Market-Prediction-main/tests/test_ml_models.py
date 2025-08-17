"""
Tests for ML models
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import tempfile
import os

from stock_prediction.models.ml_models import (
    BaseMLModel, LinearRegressionModel, RandomForestModel, 
    XGBoostModel, LightGBMModel, ModelBenchmark
)

class TestBaseMLModel:
    """Test base ML model functionality"""
    
    def test_base_model_initialization(self):
        """Test base model initialization"""
        model = LinearRegressionModel()
        assert model.model_name == "Linear Regression"
        assert model.model is None
        assert not model.is_fitted
        assert model.scaler is not None
        
    def test_base_model_not_fitted_error(self):
        """Test that unfitted model raises error on predict"""
        model = LinearRegressionModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Linear Regression model is not fitted yet"):
            model.predict(X)
            
    def test_base_model_save_load(self):
        """Test model saving and loading"""
        model = LinearRegressionModel()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Fit model
        model.fit(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            model.save_model(tmp.name)
            
            # Load model
            new_model = LinearRegressionModel()
            new_model.load_model(tmp.name)
            
            # Check predictions match
            y_pred_original = model.predict(X)
            y_pred_loaded = new_model.predict(X)
            
            np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)
            
        # Cleanup
        os.unlink(tmp.name)

class TestLinearRegressionModel:
    """Test Linear Regression model"""
    
    def test_linear_regression_build(self):
        """Test linear regression model building"""
        model = LinearRegressionModel()
        built_model = model._build_model()
        
        assert hasattr(built_model, 'fit')
        assert hasattr(built_model, 'predict')
        
    def test_linear_regression_fit_predict(self):
        """Test linear regression fitting and prediction"""
        model = LinearRegressionModel()
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
        
        # Fit model
        train_metrics = model.fit(X, y)
        
        # Check metrics
        assert 'train_mse' in train_metrics
        assert 'train_mae' in train_metrics
        assert 'train_r2' in train_metrics
        assert model.is_fitted
        
        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
        # Test evaluation
        test_metrics = model.evaluate(X, y)
        assert 'mse' in test_metrics
        assert 'mae' in test_metrics
        assert 'r2' in test_metrics
        assert 'rmse' in test_metrics

class TestRandomForestModel:
    """Test Random Forest model"""
    
    def test_random_forest_build(self):
        """Test random forest model building"""
        model = RandomForestModel(n_estimators=50, max_depth=5)
        built_model = model._build_model()
        
        assert hasattr(built_model, 'fit')
        assert hasattr(built_model, 'predict')
        assert built_model.n_estimators == 50
        assert built_model.max_depth == 5
        
    def test_random_forest_fit_predict(self):
        """Test random forest fitting and prediction"""
        model = RandomForestModel(n_estimators=50, max_depth=5)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Fit model
        train_metrics = model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
        # Test evaluation
        test_metrics = model.evaluate(X, y)
        assert all(key in test_metrics for key in ['mse', 'mae', 'r2', 'rmse'])

class TestXGBoostModel:
    """Test XGBoost model"""
    
    def test_xgboost_build(self):
        """Test XGBoost model building"""
        model = XGBoostModel(n_estimators=50, max_depth=4, learning_rate=0.05)
        built_model = model._build_model()
        
        assert hasattr(built_model, 'fit')
        assert hasattr(built_model, 'predict')
        assert built_model.n_estimators == 50
        assert built_model.max_depth == 4
        assert built_model.learning_rate == 0.05
        
    def test_xgboost_fit_predict(self):
        """Test XGBoost fitting and prediction"""
        model = XGBoostModel(n_estimators=50, max_depth=4)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Fit model
        train_metrics = model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
        # Test evaluation
        test_metrics = model.evaluate(X, y)
        assert all(key in test_metrics for key in ['mse', 'mae', 'r2', 'rmse'])

class TestLightGBMModel:
    """Test LightGBM model"""
    
    def test_lightgbm_build(self):
        """Test LightGBM model building"""
        model = LightGBMModel(
            n_estimators=50, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10
        )
        built_model = model._build_model()
        
        assert hasattr(built_model, 'fit')
        assert hasattr(built_model, 'predict')
        assert built_model.n_estimators == 50
        assert built_model.max_depth == 4
        assert built_model.learning_rate == 0.05
        assert built_model.num_leaves == 15
        assert built_model.min_child_samples == 10
        
    def test_lightgbm_fit_predict(self):
        """Test LightGBM fitting and prediction"""
        model = LightGBMModel(n_estimators=50, max_depth=4)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # Fit model
        train_metrics = model.fit(X, y)
        assert model.is_fitted
        
        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
        # Test evaluation
        test_metrics = model.evaluate(X, y)
        assert all(key in test_metrics for key in ['mse', 'mae', 'r2', 'rmse'])

class TestModelBenchmark:
    """Test Model Benchmark functionality"""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        benchmark = ModelBenchmark()
        assert benchmark.models == {}
        assert benchmark.results == {}
        
    def test_benchmark_add_model(self):
        """Test adding models to benchmark"""
        benchmark = ModelBenchmark()
        model1 = LinearRegressionModel()
        model2 = RandomForestModel()
        
        benchmark.add_model(model1)
        benchmark.add_model(model2)
        
        assert len(benchmark.models) == 2
        assert "Linear Regression" in benchmark.models
        assert "Random Forest" in benchmark.models
        
    def test_benchmark_run(self):
        """Test running benchmark"""
        benchmark = ModelBenchmark()
        
        # Add models
        benchmark.add_model(LinearRegressionModel())
        benchmark.add_model(RandomForestModel(n_estimators=10))
        
        # Create test data
        X_train = np.random.randn(80, 5)
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        
        # Run benchmark
        results = benchmark.run_benchmark(X_train, y_train, X_test, y_test)
        
        # Check results
        assert not results.empty
        assert len(results) == 2
        assert 'Model' in results.columns
        assert 'Test_R2' in results.columns
        assert 'Test_RMSE' in results.columns
        
    def test_benchmark_get_best_model(self):
        """Test getting best model from benchmark"""
        benchmark = ModelBenchmark()
        
        # Add models
        benchmark.add_model(LinearRegressionModel())
        benchmark.add_model(RandomForestModel(n_estimators=10))
        
        # Create test data
        X_train = np.random.randn(80, 5)
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        
        # Run benchmark
        benchmark.run_benchmark(X_train, y_train, X_test, y_test)
        
        # Get best model
        best_model, best_score = benchmark.get_best_model('Test_R2')
        assert best_model in ["Linear Regression", "Random Forest"]
        assert isinstance(best_score, float)
        
    def test_benchmark_get_best_model_no_results(self):
        """Test getting best model when no results available"""
        benchmark = ModelBenchmark()
        
        with pytest.raises(ValueError, match="No benchmark results available"):
            benchmark.get_best_model()
            
    def test_benchmark_save_results(self):
        """Test saving benchmark results"""
        benchmark = ModelBenchmark()
        
        # Add models and run benchmark
        benchmark.add_model(LinearRegressionModel())
        X_train = np.random.randn(80, 5)
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        
        benchmark.run_benchmark(X_train, y_train, X_test, y_test)
        
        # Save results
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            benchmark.save_results(tmp.name)
            assert os.path.exists(tmp.name)
            
            # Cleanup
            os.unlink(tmp.name)

class TestModelIntegration:
    """Test integration between different components"""
    
    def test_model_scaling_consistency(self):
        """Test that scaling is consistent across models"""
        models = [
            LinearRegressionModel(),
            RandomForestModel(n_estimators=10),
            XGBoostModel(n_estimators=10),
            LightGBMModel(n_estimators=10)
        ]
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        predictions = []
        for model in models:
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)
            
        # All predictions should have same shape
        for pred in predictions:
            assert pred.shape == y.shape
            
    def test_benchmark_with_all_models(self):
        """Test benchmark with all model types"""
        benchmark = ModelBenchmark()
        
        # Add all model types
        benchmark.add_model(LinearRegressionModel())
        benchmark.add_model(RandomForestModel(n_estimators=10))
        benchmark.add_model(XGBoostModel(n_estimators=10))
        benchmark.add_model(LightGBMModel(n_estimators=10))
        
        # Create test data
        X_train = np.random.randn(80, 5)
        y_train = np.random.randn(80)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        
        # Run benchmark
        results = benchmark.run_benchmark(X_train, y_train, X_test, y_test)
        
        # Check all models are included
        assert len(results) == 4
        expected_models = ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]
        for model in expected_models:
            assert model in results['Model'].values
