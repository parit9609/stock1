"""
Traditional Machine Learning models for stock prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import joblib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseMLModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the model architecture"""
        pass
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit the model to the training data
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training targets
            
        Returns:
            Dict[str, Any]: Training results
        """
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Build and fit model
            self.model = self._build_model()
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # Make predictions on training data
            y_pred = self.predict(X)
            
            # Calculate training metrics
            train_mse = mean_squared_error(y, y_pred)
            train_mae = mean_absolute_error(y, y_pred)
            train_r2 = r2_score(y, y_pred)
            
            return {
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_r2': train_r2
            }
            
        except Exception as e:
            logger.error(f"Error fitting {self.model_name}: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} model is not fitted yet")
            
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_name}: {str(e)}")
            raise
            
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): Test targets
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            y_pred = self.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }
        except Exception as e:
            logger.error(f"Error evaluating {self.model_name}: {str(e)}")
            raise
            
    def save_model(self, filepath: str) -> None:
        """Save the fitted model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }
            joblib.dump(model_data, filepath)
            logger.info(f"{self.model_name} saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving {self.model_name}: {str(e)}")
            raise
            
    def load_model(self, filepath: str) -> None:
        """Load a fitted model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            logger.info(f"{self.model_name} loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
            raise
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      n_splits: int = 5, test_size: int = None,
                      gap: int = 0) -> Dict[str, Any]:
        """
        Perform time-series cross-validation using walk-forward approach
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_splits (int): Number of validation splits
            test_size (int): Size of test set
            gap (int): Gap between train and test sets
            
        Returns:
            Dict[str, Any]: Cross-validation results with fold-level metrics
        """
        try:
            from ..data.time_series_cv import WalkForwardValidator
            
            validator = WalkForwardValidator(
                n_splits=n_splits,
                test_size=test_size,
                gap=gap
            )
            
            # Get feature names if available
            feature_names = None
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            
            results = validator.validate_model(self.model, X, y, feature_names)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

class LinearRegressionModel(BaseMLModel):
    """Linear Regression model for stock prediction"""
    
    def __init__(self):
        super().__init__("Linear Regression")
        
    def _build_model(self) -> LinearRegression:
        return LinearRegression()

class RandomForestModel(BaseMLModel):
    """Random Forest model for stock prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        super().__init__("Random Forest")
        
    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

class XGBoostModel(BaseMLModel):
    """XGBoost model for stock prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        super().__init__("XGBoost")
        
    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )

class LightGBMModel(BaseMLModel):
    """LightGBM model for stock prediction"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, random_state: int = 42,
                 num_leaves: int = 31, min_child_samples: int = 20):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        super().__init__("LightGBM")
        
    def _build_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            n_jobs=-1,
            verbose=-1
        )

class ModelBenchmark:
    """Benchmark class to compare different ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def add_model(self, model: BaseMLModel) -> None:
        """Add a model to the benchmark"""
        self.models[model.model_name] = model
        
    def run_benchmark(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Run benchmark on all models
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            pd.DataFrame: Benchmark results
        """
        results = []
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                train_metrics = model.fit(X_train, y_train)
                
                # Evaluate model
                test_metrics = model.evaluate(X_test, y_test)
                
                # Store results
                result = {
                    'Model': name,
                    'Train_MSE': train_metrics['train_mse'],
                    'Train_MAE': train_metrics['train_mae'],
                    'Train_R2': train_metrics['train_r2'],
                    'Test_MSE': test_metrics['mse'],
                    'Test_MAE': test_metrics['mae'],
                    'Test_R2': test_metrics['r2'],
                    'Test_RMSE': test_metrics['rmse']
                }
                results.append(result)
                
                logger.info(f"{name} completed - Test R2: {test_metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error benchmarking {name}: {str(e)}")
                continue
                
        self.results = pd.DataFrame(results)
        return self.results
        
    def get_best_model(self, metric: str = 'Test_R2') -> Tuple[str, float]:
        """Get the best performing model based on a metric"""
        if self.results.empty:
            raise ValueError("No benchmark results available. Run benchmark first.")
            
        best_idx = self.results[metric].idxmax()
        best_model = self.results.loc[best_idx, 'Model']
        best_score = self.results.loc[best_idx, metric]
        
        return best_model, best_score
        
    def save_results(self, filepath: str) -> None:
        """Save benchmark results to CSV"""
        try:
            self.results.to_csv(filepath, index=False)
            logger.info(f"Benchmark results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def run_time_series_benchmark(self, X: np.ndarray, y: np.ndarray,
                                 n_splits: int = 5, test_size: int = None,
                                 gap: int = 0) -> Dict[str, Any]:
        """
        Run benchmark using time-series cross-validation (walk-forward)
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            n_splits (int): Number of validation splits
            test_size (int): Size of test set
            gap (int): Gap between train and test sets
            
        Returns:
            Dict[str, Any]: Comprehensive benchmark results with fold-level metrics
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Running time-series CV for {name}...")
                
                # Perform time-series cross-validation
                cv_results = model.cross_validate(X, y, n_splits, test_size, gap)
                
                results[name] = cv_results
                
                # Log overall performance
                if 'overall_metrics' in cv_results:
                    overall = cv_results['overall_metrics']
                    logger.info(f"{name} CV Results:")
                    logger.info(f"  R²: {overall.get('r2_mean', 'N/A'):.4f} ± {overall.get('r2_std', 'N/A'):.4f}")
                    logger.info(f"  RMSE: {overall.get('rmse_mean', 'N/A'):.4f} ± {overall.get('rmse_std', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"Error in time-series CV for {name}: {str(e)}")
                continue
        
        self.results = results
        return results
