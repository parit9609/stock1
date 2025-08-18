"""
Comprehensive Tests for LightGBM Integration
Validates model loading, predictions, and input-output consistency
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stock_prediction.models.ml_models import LightGBMModel
from stock_prediction.data.data_processor import DataProcessor

class TestLightGBMIntegration(unittest.TestCase):
    """Test suite for LightGBM integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LightGBMModel(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randn(20)
        
        # Create sample stock data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        self.sample_stock_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(len(dates)) + 100,
            'High': np.random.randn(len(dates)) + 102,
            'Low': np.random.randn(len(dates)) + 98,
            'Close': np.random.randn(len(dates)) + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        self.sample_stock_data.set_index('Date', inplace=True)
    
    def test_model_initialization(self):
        """Test LightGBM model initialization"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.model)
        
        # Check model parameters
        self.assertEqual(self.model.model.n_estimators, 10)
        self.assertEqual(self.model.model.max_depth, 3)
        self.assertEqual(self.model.model.learning_rate, 0.1)
        self.assertEqual(self.model.model.random_state, 42)
    
    def test_model_fitting(self):
        """Test LightGBM model fitting"""
        # Test fitting
        train_metrics = self.model.fit(self.X_train, self.y_train)
        
        # Check that training metrics are returned
        self.assertIsInstance(train_metrics, dict)
        self.assertIn('train_mse', train_metrics)
        self.assertIn('train_mae', train_metrics)
        self.assertIn('train_r2', train_metrics)
        
        # Check that metrics are numeric
        self.assertIsInstance(train_metrics['train_mse'], (int, float))
        self.assertIsInstance(train_metrics['train_mae'], (int, float))
        self.assertIsInstance(train_metrics['train_r2'], (int, float))
        
        # Check that model is fitted
        self.assertTrue(hasattr(self.model.model, 'feature_importances_'))
    
    def test_model_prediction(self):
        """Test LightGBM model prediction"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (20,))
        
        # Check that predictions are numeric
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))
        
        # Check that predictions are not all the same (model learned something)
        self.assertFalse(np.all(predictions == predictions[0]))
        
        # Check that predictions are finite
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_model_evaluation(self):
        """Test LightGBM model evaluation"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        test_metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Check that evaluation metrics are returned
        self.assertIsInstance(test_metrics, dict)
        self.assertIn('mse', test_metrics)
        self.assertIn('mae', test_metrics)
        self.assertIn('r2', test_metrics)
        self.assertIn('rmse', test_metrics)
        
        # Check that metrics are numeric
        for metric_name, metric_value in test_metrics.items():
            self.assertIsInstance(metric_value, (int, float), f"Metric {metric_name} is not numeric")
        
        # Check that MSE and RMSE are positive
        self.assertGreater(test_metrics['mse'], 0)
        self.assertGreater(test_metrics['rmse'], 0)
        
        # Check that MAE is positive
        self.assertGreater(test_metrics['mae'], 0)
        
        # Check that R² is between -inf and 1
        self.assertLessEqual(test_metrics['r2'], 1)
    
    def test_model_saving_and_loading(self):
        """Test LightGBM model saving and loading"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.model.save_model(model_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            loaded_model = LightGBMModel()
            loaded_model.load_model(model_path)
            
            # Check that loaded model can make predictions
            original_predictions = self.model.predict(self.X_test)
            loaded_predictions = loaded_model.predict(self.X_test)
            
            # Check that predictions are the same
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_feature_importance(self):
        """Test LightGBM feature importance"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)
        
        # Check that feature importance exists
        self.assertTrue(hasattr(self.model.model, 'feature_importances_'))
        
        # Check feature importance shape
        feature_importance = self.model.model.feature_importances_
        self.assertEqual(feature_importance.shape, (10,))  # 10 features
        
        # Check that feature importance values are numeric and non-negative
        self.assertTrue(np.issubdtype(feature_importance.dtype, np.number))
        self.assertTrue(np.all(feature_importance >= 0))
        
        # Check that feature importance sums to a reasonable value
        importance_sum = np.sum(feature_importance)
        self.assertGreater(importance_sum, 0)
    
    def test_input_validation(self):
        """Test LightGBM input validation"""
        # Test with invalid input shapes
        with self.assertRaises(Exception):
            self.model.predict(np.random.randn(20, 5))  # Wrong number of features
        
        # Test with empty input
        with self.assertRaises(Exception):
            self.model.predict(np.array([]))
        
        # Test with None input
        with self.assertRaises(Exception):
            self.model.predict(None)
    
    def test_data_consistency(self):
        """Test data consistency between training and prediction"""
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions on training data
        train_predictions = self.model.predict(self.X_train)
        
        # Check that training predictions have correct shape
        self.assertEqual(train_predictions.shape, self.y_train.shape)
        
        # Check that predictions are numeric
        self.assertTrue(np.issubdtype(train_predictions.dtype, np.number))
        
        # Check that predictions are finite
        self.assertTrue(np.all(np.isfinite(train_predictions)))
    
    def test_model_parameters(self):
        """Test LightGBM model parameters"""
        # Test different parameter combinations
        test_params = [
            {'n_estimators': 5, 'max_depth': 2, 'learning_rate': 0.05},
            {'n_estimators': 20, 'max_depth': 5, 'learning_rate': 0.2},
            {'n_estimators': 50, 'max_depth': 10, 'learning_rate': 0.01}
        ]
        
        for params in test_params:
            with self.subTest(params=params):
                model = LightGBMModel(**params)
                
                # Check that parameters are set correctly
                self.assertEqual(model.model.n_estimators, params['n_estimators'])
                self.assertEqual(model.model.max_depth, params['max_depth'])
                self.assertEqual(model.model.learning_rate, params['learning_rate'])
                
                # Test that model can be fitted
                model.fit(self.X_train, self.y_train)
                
                # Test that model can make predictions
                predictions = model.predict(self.X_test)
                self.assertEqual(predictions.shape, (20,))
    
    def test_integration_with_data_processor(self):
        """Test LightGBM integration with data processor"""
        # Create data processor
        data_processor = DataProcessor(sequence_length=60)
        
        # Create features
        df_with_features = data_processor.create_features(self.sample_stock_data)
        
        # Prepare data for ML models
        X, y = data_processor.prepare_ml_data(df_with_features)
        
        # Check that data is properly shaped
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
        self.assertEqual(y.shape[0], X.shape[0])
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y, train_ratio=0.8)
        
        # Fit LightGBM model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Check predictions
        self.assertEqual(predictions.shape, y_test.shape)
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_model_performance_consistency(self):
        """Test that model performance is consistent across multiple runs"""
        # Fit model multiple times with same data
        performances = []
        
        for _ in range(3):
            model = LightGBMModel(n_estimators=10, random_state=42)
            model.fit(self.X_train, self.y_train)
            metrics = model.evaluate(self.X_test, self.y_test)
            performances.append(metrics['r2'])
        
        # Check that performance is consistent (R² should be similar)
        performance_std = np.std(performances)
        self.assertLess(performance_std, 0.1, "Model performance varies too much across runs")
    
    def test_edge_cases(self):
        """Test edge cases for LightGBM model"""
        # Test with very small dataset
        X_small = np.random.randn(5, 3)
        y_small = np.random.randn(5)
        
        model_small = LightGBMModel(n_estimators=2, max_depth=2)
        model_small.fit(X_small, y_small)
        
        predictions_small = model_small.predict(X_small)
        self.assertEqual(predictions_small.shape, (5,))
        
        # Test with single sample
        X_single = np.random.randn(1, 3)
        y_single = np.random.randn(1)
        
        model_single = LightGBMModel(n_estimators=2, max_depth=2)
        model_single.fit(X_single, y_single)
        
        predictions_single = model_single.predict(X_single)
        self.assertEqual(predictions_single.shape, (1,))
    
    def test_model_serialization_consistency(self):
        """Test that model serialization preserves all properties"""
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Get original predictions and feature importance
        original_predictions = self.model.predict(self.X_test)
        original_importance = self.model.model.feature_importances_.copy()
        
        # Save and reload model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.model.save_model(model_path)
            
            loaded_model = LightGBMModel()
            loaded_model.load_model(model_path)
            
            # Check predictions are identical
            loaded_predictions = loaded_model.predict(self.X_test)
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
            
            # Check feature importance is identical
            loaded_importance = loaded_model.model.feature_importances_
            np.testing.assert_array_almost_equal(original_importance, loaded_importance)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

class TestLightGBMProductionReadiness(unittest.TestCase):
    """Test suite for LightGBM production readiness"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = LightGBMModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Create realistic stock data
        np.random.seed(42)
        dates = pd.date_range(start='2018-01-01', end='2024-01-01', freq='D')
        
        # Generate realistic stock prices with trends and volatility
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.stock_data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        self.stock_data.set_index('Date', inplace=True)
    
    def test_large_dataset_handling(self):
        """Test LightGBM performance with large datasets"""
        # Create large dataset
        X_large = np.random.randn(10000, 50)
        y_large = np.random.randn(10000)
        
        # Test that model can handle large dataset
        start_time = pd.Timestamp.now()
        self.model.fit(X_large, y_large)
        training_time = pd.Timestamp.now() - start_time
        
        # Check that training completes in reasonable time (less than 30 seconds)
        self.assertLess(training_time.total_seconds(), 30)
        
        # Test prediction performance
        start_time = pd.Timestamp.now()
        predictions = self.model.predict(X_large[:1000])  # Test on subset
        prediction_time = pd.Timestamp.now() - start_time
        
        # Check that prediction is fast (less than 5 seconds for 1000 samples)
        self.assertLess(prediction_time.total_seconds(), 5)
        
        # Check predictions
        self.assertEqual(predictions.shape, (1000,))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_memory_efficiency(self):
        """Test LightGBM memory efficiency"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and fit model
        X = np.random.randn(5000, 30)
        y = np.random.randn(5000)
        
        self.model.fit(X, y)
        
        # Get memory usage after training
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Check that memory increase is reasonable (less than 500 MB)
        self.assertLess(memory_increase, 500, f"Memory increase too high: {memory_increase:.2f} MB")
    
    def test_prediction_accuracy_improvement(self):
        """Test that LightGBM improves with more data"""
        # Test with different dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        performances = []
        
        for size in dataset_sizes:
            X_subset = np.random.randn(size, 20)
            y_subset = np.random.randn(size)
            
            # Split data
            split_idx = int(size * 0.8)
            X_train, X_test = X_subset[:split_idx], X_subset[split_idx:]
            y_train, y_test = y_subset[:split_idx], y_subset[split_idx:]
            
            # Fit model
            model = LightGBMModel(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            performances.append(metrics['r2'])
        
        # Check that performance generally improves or stays stable with more data
        # (Allow for some variance due to random data)
        self.assertGreaterEqual(performances[-1], performances[0] - 0.1)
    
    def test_feature_robustness(self):
        """Test LightGBM robustness to different feature types"""
        # Test with different feature scales
        X_normal = np.random.randn(1000, 10)
        X_scaled = X_normal * 1000  # Scale up features
        
        y = np.random.randn(1000)
        
        # Fit model on normal scale
        model_normal = LightGBMModel(n_estimators=50, random_state=42)
        model_normal.fit(X_normal, y)
        pred_normal = model_normal.predict(X_normal)
        
        # Fit model on scaled features
        model_scaled = LightGBMModel(n_estimators=50, random_state=42)
        model_scaled.fit(X_scaled, y)
        pred_scaled = model_scaled.predict(X_scaled)
        
        # Check that predictions are similar (allowing for some numerical differences)
        np.testing.assert_array_almost_equal(pred_normal, pred_scaled, decimal=2)
    
    def test_error_handling(self):
        """Test LightGBM error handling"""
        # Test with invalid hyperparameters
        with self.assertRaises(Exception):
            invalid_model = LightGBMModel(
                n_estimators=-1,  # Invalid parameter
                random_state=42
            )
        
        # Test with corrupted data
        X_corrupted = np.random.randn(100, 10)
        X_corrupted[0, 0] = np.nan  # Introduce NaN
        
        with self.assertRaises(Exception):
            self.model.fit(X_corrupted, np.random.randn(100))
        
        # Test with mismatched dimensions
        X_mismatch = np.random.randn(100, 5)
        y_mismatch = np.random.randn(100)
        
        self.model.fit(X_mismatch, y_mismatch)
        
        # Try to predict with different number of features
        with self.assertRaises(Exception):
            self.model.predict(np.random.randn(50, 10))

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
