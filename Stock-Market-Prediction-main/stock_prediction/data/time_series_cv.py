"""
Time Series Cross-Validation Module
Implements walk-forward validation to prevent data leakage in time series data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Generator, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
import logging

logger = logging.getLogger(__name__)

class TimeSeriesSplit(BaseCrossValidator):
    """
    Time Series Cross-Validator
    
    Provides train/test indices to split time series data samples.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    
    This cross-validation object is a variation of TimeSeriesSplit from sklearn
    with additional features for walk-forward validation.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = None, 
                 gap: int = 0, expand_window: bool = False):
        """
        Initialize TimeSeriesSplit
        
        Args:
            n_splits (int): Number of splits. Must be at least 2.
            test_size (int): Size of test set. If None, test_size = n_samples // (n_splits + 1)
            gap (int): Number of samples to exclude from the end of each training set
            expand_window (bool): If True, training set grows with each split (walk-forward)
                                 If False, training set has fixed size (sliding window)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expand_window = expand_window
        
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X (array-like): Training data
            y (array-like): Target variable
            groups (array-like): Group labels
            
        Yields:
            tuple: (train_index, test_index)
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater "
                f"than the number of samples: n_samples={n_samples}."
            )
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        if test_size == 0:
            raise ValueError(
                f"test_size={test_size} is too small. "
                f"n_samples={n_samples} cannot be split into {self.n_splits} splits."
            )
            
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate test indices
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            test_indices = indices[test_start:test_end]
            
            # Calculate train indices
            if self.expand_window:
                # Walk-forward: training set grows with each split
                train_indices = indices[:test_start - self.gap]
            else:
                # Sliding window: training set has fixed size
                train_start = max(0, test_start - test_size - self.gap)
                train_indices = indices[train_start:test_start - self.gap]
            
            if len(train_indices) == 0:
                logger.warning(f"Split {i}: No training samples available")
                continue
                
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

class WalkForwardValidator:
    """
    Walk-Forward Validation for Time Series Data
    
    Implements a more sophisticated walk-forward validation approach
    that ensures no data leakage and provides comprehensive validation metrics.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = None, 
                 gap: int = 0, min_train_size: int = 100):
        """
        Initialize WalkForwardValidator
        
        Args:
            n_splits (int): Number of validation splits
            test_size (int): Size of each test set
            gap (int): Gap between train and test sets to prevent leakage
            min_train_size (int): Minimum size of training set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.min_train_size = min_train_size
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            expand_window=True  # Always use walk-forward for validation
        )
        
    def validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Perform walk-forward validation on a model
        
        Args:
            model: Model with fit() and predict() methods
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            feature_names (List[str]): Names of features for importance analysis
            
        Returns:
            Dict[str, Any]: Validation results including fold-level metrics
        """
        results = {
            'fold_metrics': [],
            'predictions': [],
            'actuals': [],
            'feature_importance': [],
            'fold_indices': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Skip if training set is too small
            if len(X_train) < self.min_train_size:
                logger.warning(f"Fold {fold + 1}: Training set too small ({len(X_train)} < {self.min_train_size})")
                continue
            
            # Train model
            try:
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                fold_metrics = self._calculate_metrics(y_test, y_pred)
                fold_metrics['fold'] = fold + 1
                fold_metrics['train_size'] = len(train_idx)
                fold_metrics['test_size'] = len(test_idx)
                
                results['fold_metrics'].append(fold_metrics)
                results['predictions'].extend(y_pred)
                results['actuals'].extend(y_test)
                results['fold_indices'].append((train_idx, test_idx))
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(feature_names or range(X.shape[1]), 
                                       model.feature_importances_))
                    results['feature_importance'].append(importance)
                elif hasattr(model, 'coef_'):
                    importance = dict(zip(feature_names or range(X.shape[1]), 
                                       np.abs(model.coef_)))
                    results['feature_importance'].append(importance)
                    
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        # Calculate overall metrics
        if results['fold_metrics']:
            results['overall_metrics'] = self._calculate_overall_metrics(results['fold_metrics'])
            results['stability_metrics'] = self._calculate_stability_metrics(results['fold_metrics'])
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate individual fold metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def _calculate_overall_metrics(self, fold_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate overall metrics across all folds"""
        metrics_df = pd.DataFrame(fold_metrics)
        
        overall = {}
        for metric in ['mse', 'rmse', 'mae', 'r2', 'mape']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                overall[f'{metric}_mean'] = values.mean()
                overall[f'{metric}_std'] = values.std()
                overall[f'{metric}_min'] = values.min()
                overall[f'{metric}_max'] = values.max()
        
        return overall
    
    def _calculate_stability_metrics(self, fold_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate stability metrics across folds"""
        metrics_df = pd.DataFrame(fold_metrics)
        
        stability = {}
        for metric in ['rmse', 'r2']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 1:
                    # Coefficient of variation (lower is more stable)
                    stability[f'{metric}_cv'] = values.std() / values.mean()
                    # Range ratio (lower is more stable)
                    stability[f'{metric}_range_ratio'] = (values.max() - values.min()) / values.mean()
        
        return stability

def create_time_series_splits(X: np.ndarray, y: np.ndarray, 
                             n_splits: int = 5, test_size: int = None,
                             gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create time series splits for manual validation
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        n_splits (int): Number of splits
        test_size (int): Size of test set
        gap (int): Gap between train and test
        
    Returns:
        List[Tuple]: List of (X_train, X_test, y_train, y_test) tuples
    """
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    splits = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        splits.append((X_train, X_test, y_train, y_test))
    
    return splits
