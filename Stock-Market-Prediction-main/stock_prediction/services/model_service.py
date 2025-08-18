"""
Model Service Layer
Separates business logic from API and dashboard concerns
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import yaml

from stock_prediction.models.ml_models import (
    BaseMLModel, LinearRegressionModel, RandomForestModel,
    XGBoostModel, LightGBMModel, ModelBenchmark
)
from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.analysis.shap_analysis import SHAPAnalyzer

logger = logging.getLogger(__name__)


class ModelService:
    """Service layer for model operations"""
    
    def __init__(self, config_path: str):
        """
        Initialize the model service
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_processor = DataProcessor(
            sequence_length=self.config.get('sequence_length', 10)
        )
        self.models = {}
        self._load_models()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _load_models(self) -> None:
        """Load trained models from disk"""
        model_paths = {
            'linear_regression': self.config.get('lr_model_path', 'models/linear_regression_model.joblib'),
            'random_forest': self.config.get('rf_model_path', 'models/random_forest_model.joblib'),
            'xgboost': self.config.get('xgb_model_path', 'models/xgboost_model.joblib'),
            'lightgbm': self.config.get('lightgbm_model_path', 'models/lightgbm_model.joblib')
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if Path(model_path).exists():
                    model = self._create_model_instance(model_name)
                    model.load_model(model_path)
                    self.models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
    
    def _create_model_instance(self, model_name: str) -> BaseMLModel:
        """Create model instance based on name"""
        if model_name == 'linear_regression':
            return LinearRegressionModel()
        elif model_name == 'random_forest':
            return RandomForestModel(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', None),
                random_state=self.config.get('random_state', 42)
            )
        elif model_name == 'xgboost':
            return XGBoostModel(
                n_estimators=self.config.get('xgb_n_estimators', 100),
                max_depth=self.config.get('xgb_max_depth', 6),
                learning_rate=self.config.get('xgb_learning_rate', 0.1),
                random_state=self.config.get('random_state', 42),
                early_stopping_rounds=self.config.get('xgb_early_stopping_rounds', 50),
                eval_metric=self.config.get('xgb_eval_metric', 'rmse')
            )
        elif model_name == 'lightgbm':
            lightgbm_config = self.config.get('lightgbm', {})
            return LightGBMModel(
                n_estimators=lightgbm_config.get('n_estimators', 200),
                max_depth=lightgbm_config.get('max_depth', 8),
                learning_rate=lightgbm_config.get('learning_rate', 0.05),
                random_state=lightgbm_config.get('random_state', 42),
                num_leaves=lightgbm_config.get('num_leaves', 31),
                min_child_samples=lightgbm_config.get('min_child_samples', 20),
                early_stopping_rounds=lightgbm_config.get('early_stopping_rounds', 50),
                eval_metric=lightgbm_config.get('metric', 'rmse'),
                subsample=lightgbm_config.get('subsample', 0.8),
                colsample_bytree=lightgbm_config.get('colsample_bytree', 0.8),
                reg_alpha=lightgbm_config.get('reg_alpha', 0.1),
                reg_lambda=lightgbm_config.get('reg_lambda', 0.1)
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def predict(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using specified model
        
        Args:
            model_name (str): Name of the model to use
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available: {self.get_available_models()}")
        
        try:
            # Create features
            features_df = self.data_processor.create_features(data)
            
            # Prepare ML data
            X, y = self.data_processor.prepare_ml_data(features_df)
            
            # Make predictions
            model = self.models[model_name]
            predictions = model.predict(X)
            
            # Calculate metrics if true values available
            metrics = {}
            if y is not None and len(y) > 0:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': float(mean_squared_error(y, predictions)),
                    'mae': float(mean_absolute_error(y, predictions)),
                    'r2': float(r2_score(y, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(y, predictions)))
                }
            
            return {
                'model_name': model_name,
                'predictions': predictions.tolist(),
                'metrics': metrics,
                'n_predictions': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {str(e)}")
            raise
    
    def predict_all_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using all available models
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Predictions from all models
        """
        results = {}
        
        for model_name in self.get_available_models():
            try:
                results[model_name] = self.predict(model_name, data)
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        return {
            'name': model.model_name,
            'type': type(model).__name__,
            'is_fitted': model.is_fitted,
            'needs_scaling': model.needs_scaling,
            'parameters': self._get_model_parameters(model)
        }
    
    def _get_model_parameters(self, model: BaseMLModel) -> Dict[str, Any]:
        """Extract model parameters"""
        if isinstance(model, RandomForestModel):
            return {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'random_state': model.random_state
            }
        elif isinstance(model, XGBoostModel):
            return {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'learning_rate': model.learning_rate,
                'random_state': model.random_state,
                'early_stopping_rounds': model.early_stopping_rounds,
                'eval_metric': model.eval_metric
            }
        elif isinstance(model, LightGBMModel):
            return {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'learning_rate': model.learning_rate,
                'random_state': model.random_state,
                'num_leaves': model.num_leaves,
                'min_child_samples': model.min_child_samples,
                'early_stopping_rounds': model.early_stopping_rounds,
                'eval_metric': model.eval_metric
            }
        else:
            return {}
    
    def get_feature_importance(self, model_name: str, data: pd.DataFrame, 
                              top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance for a model
        
        Args:
            model_name (str): Name of the model
            data (pd.DataFrame): Input data for analysis
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, Any]: Feature importance information
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        try:
            # Create features
            features_df = self.data_processor.create_features(data)
            
            # Prepare ML data
            X, y = self.data_processor.prepare_ml_data(features_df)
            
            # Get feature names
            feature_names = list(features_df.columns)
            
            # Create SHAP analyzer
            model = self.models[model_name]
            analyzer = SHAPAnalyzer(model.model, feature_names)
            
            # Get feature importance
            importance_data = analyzer.plot_feature_importance(X, top_n=top_n)
            
            return {
                'model_name': model_name,
                'feature_importance': importance_data,
                'n_features': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {model_name}: {str(e)}")
            raise
    
    def retrain_model(self, model_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain a model with new data
        
        Args:
            model_name (str): Name of the model to retrain
            data (pd.DataFrame): Training data
            
        Returns:
            Dict[str, Any]: Training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        try:
            # Create features
            features_df = self.data_processor.create_features(data)
            
            # Prepare ML data
            X, y = self.data_processor.prepare_ml_data(features_df)
            
            # Retrain model
            model = self.models[model_name]
            train_metrics = model.fit(X, y)
            
            # Save updated model
            model_path = self.config.get(f'{model_name}_model_path', f'models/{model_name}_model.joblib')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            
            return {
                'model_name': model_name,
                'training_metrics': train_metrics,
                'model_saved_to': model_path
            }
            
        except Exception as e:
            logger.error(f"Error retraining {model_name}: {str(e)}")
            raise
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for all models
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        summary = {
            'total_models': len(self.models),
            'available_models': self.get_available_models(),
            'model_status': {}
        }
        
        for model_name, model in self.models.items():
            summary['model_status'][model_name] = {
                'is_fitted': model.is_fitted,
                'type': type(model).__name__,
                'needs_scaling': model.needs_scaling
            }
        
        return summary
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        try:
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        validation_results['warnings'].append(f"Column {col} is not numeric")
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                validation_results['warnings'].append(f"Found {missing_values} missing values")
            
            # Check for infinite values
            inf_values = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_values > 0:
                validation_results['warnings'].append(f"Found {inf_values} infinite values")
            
            # Data info
            validation_results['data_info'] = {
                'shape': data.shape,
                'date_range': {
                    'start': str(data.index.min()) if hasattr(data.index, 'min') else 'N/A',
                    'end': str(data.index.max()) if hasattr(data.index, 'max') else 'N/A'
                },
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict()
            }
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results


class ModelServiceFactory:
    """Factory for creating model services"""
    
    @staticmethod
    def create_service(config_path: str) -> ModelService:
        """
        Create a model service instance
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            ModelService: Configured model service
        """
        return ModelService(config_path)
    
    @staticmethod
    def create_service_from_env() -> ModelService:
        """
        Create a model service from environment configuration
        
        Returns:
            ModelService: Configured model service
        """
        import os
        config_path = os.getenv('STOCK_PREDICTION_CONFIG', 'config/training_config.yaml')
        return ModelService(config_path)
