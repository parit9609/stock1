"""
Central Training Pipeline for Stock Market Prediction
Supports training of multiple models including LightGBM, XGBoost, Random Forest, and LSTM
"""

import argparse
import logging
from pathlib import Path
import yaml
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, ModelBenchmark
)
from stock_prediction.models.lstm_model import LSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(config: dict, data_processor: DataProcessor):
    """Prepare data for training"""
    try:
        # Check if processed data exists, otherwise create it
        processed_data_path = config.get('processed_data_path', 'data/processed_stock_data.csv')
        if Path(processed_data_path).exists():
            logger.info(f"Loading processed data from {processed_data_path}")
            df_with_features = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        else:
            logger.info("Processed data not found, creating features from raw data...")
            df = data_processor.load_data(config['data_path'])
            df_with_features = data_processor.create_features(df)
            
            # Save processed data
            Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
            df_with_features.to_csv(processed_data_path)
            logger.info(f"Processed data saved to {processed_data_path}")
        
        return df_with_features
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_lightgbm(config: dict, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray):
    """Train LightGBM model"""
    try:
        logger.info("Training LightGBM model...")
        
        lightgbm_config = config.get('lightgbm', {})
        model = LightGBMModel(
            n_estimators=lightgbm_config.get('n_estimators', 100),
            max_depth=lightgbm_config.get('max_depth', 6),
            learning_rate=lightgbm_config.get('learning_rate', 0.1),
            random_state=lightgbm_config.get('random_state', 42),
            num_leaves=lightgbm_config.get('num_leaves', 31),
            min_child_samples=lightgbm_config.get('min_child_samples', 20)
        )
        
        # Train model
        train_metrics = model.fit(X_train, y_train)
        
        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = config.get('lightgbm_model_path', 'models/lightgbm_model.joblib')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"LightGBM model saved to: {model_path}")
        
        return model, train_metrics, test_metrics
        
    except Exception as e:
        logger.error(f"Error training LightGBM: {str(e)}")
        raise

def train_xgboost(config: dict, X_train: np.ndarray, y_train: np.ndarray, 
                  X_test: np.ndarray, y_test: np.ndarray):
    """Train XGBoost model"""
    try:
        logger.info("Training XGBoost model...")
        
        model = XGBoostModel(
            n_estimators=config.get('xgb_n_estimators', 100),
            max_depth=config.get('xgb_max_depth', 6),
            learning_rate=config.get('xgb_learning_rate', 0.1),
            random_state=config.get('random_state', 42)
        )
        
        # Train model
        train_metrics = model.fit(X_train, y_train)
        
        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = config.get('xgb_model_path', 'models/xgboost_model.joblib')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"XGBoost model saved to: {model_path}")
        
        return model, train_metrics, test_metrics
        
    except Exception as e:
        logger.error(f"Error training XGBoost: {str(e)}")
        raise

def train_random_forest(config: dict, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray):
    """Train Random Forest model"""
    try:
        logger.info("Training Random Forest model...")
        
        model = RandomForestModel(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', None),
            random_state=config.get('random_state', 42)
        )
        
        # Train model
        train_metrics = model.fit(X_train, y_train)
        
        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = config.get('rf_model_path', 'models/random_forest_model.joblib')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"Random Forest model saved to: {model_path}")
        
        return model, train_metrics, test_metrics
        
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        raise

def train_linear_regression(config: dict, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray):
    """Train Linear Regression model"""
    try:
        logger.info("Training Linear Regression model...")
        
        model = LinearRegressionModel()
        
        # Train model
        train_metrics = model.fit(X_train, y_train)
        
        # Evaluate model
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = config.get('lr_model_path', 'models/linear_regression_model.joblib')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        
        logger.info(f"Linear Regression model saved to: {model_path}")
        
        return model, train_metrics, test_metrics
        
    except Exception as e:
        logger.error(f"Error training Linear Regression: {str(e)}")
        raise

def train_lstm(config: dict, data_processor: DataProcessor, df: pd.DataFrame):
    """Train LSTM model"""
    try:
        logger.info("Training LSTM model...")
        
        # Prepare data for LSTM
        X_lstm, y_lstm = data_processor.prepare_data(df, target_column=config['target_column'])
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = data_processor.split_data(
            X_lstm, y_lstm, train_ratio=config['train_ratio']
        )
        
        # Initialize LSTM model
        lstm_model = LSTMModel(
            sequence_length=config['sequence_length'],
            n_features=config['n_features']
        )
        
        # Train LSTM
        history = lstm_model.train(
            X_train_lstm, y_train_lstm,
            X_test_lstm, y_test_lstm,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            model_path=config['model_path']
        )
        
        # Evaluate LSTM
        lstm_test_loss = lstm_model.evaluate(X_test_lstm, y_test_lstm)
        
        # Convert LSTM predictions for comparison
        y_pred_lstm = lstm_model.predict(X_test_lstm)
        y_test_lstm_flat = y_test_lstm.reshape(-1)
        y_pred_lstm_flat = y_pred_lstm.reshape(-1)
        
        # Calculate LSTM metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        lstm_mse = mean_squared_error(y_test_lstm_flat, y_pred_lstm_flat)
        lstm_mae = mean_absolute_error(y_test_lstm_flat, y_pred_lstm_flat)
        lstm_r2 = r2_score(y_test_lstm_flat, y_pred_lstm_flat)
        lstm_rmse = np.sqrt(lstm_mse)
        
        train_metrics = {
            'train_mse': history['loss'][-1],
            'train_mae': history['loss'][-1],
            'train_r2': 1 - history['loss'][-1]
        }
        
        test_metrics = {
            'mse': lstm_mse,
            'mae': lstm_mae,
            'r2': lstm_r2,
            'rmse': lstm_rmse
        }
        
        logger.info(f"LSTM model saved to: {config['model_path']}")
        
        return lstm_model, train_metrics, test_metrics
        
    except Exception as e:
        logger.error(f"Error training LSTM: {str(e)}")
        raise

def run_training_pipeline(config_path: str, models_to_train: list = None):
    """
    Run the complete training pipeline
    
    Args:
        config_path (str): Path to configuration file
        models_to_train (list): List of models to train. If None, trains all models
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Setup MLflow
        mlflow.set_experiment(config['experiment_name'])
        
        # Initialize data processor
        data_processor = DataProcessor(
            sequence_length=config['sequence_length']
        )
        
        # Prepare data
        df_with_features = prepare_data(config, data_processor)
        
        # Prepare data for ML models
        X, y = data_processor.prepare_ml_data(df_with_features)
        X_train, X_test, y_train, y_test = data_processor.split_data(
            X, y, train_ratio=config['train_ratio']
        )
        
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Test data: {X_test.shape[0]} samples")
        
        # Define available models
        available_models = {
            'lightgbm': train_lightgbm,
            'xgboost': train_xgboost,
            'random_forest': train_random_forest,
            'linear_regression': train_linear_regression,
            'lstm': train_lstm
        }
        
        # If no specific models specified, train all
        if models_to_train is None:
            models_to_train = list(available_models.keys())
        
        # Validate model names
        invalid_models = [m for m in models_to_train if m not in available_models]
        if invalid_models:
            raise ValueError(f"Invalid model names: {invalid_models}. Available: {list(available_models.keys())}")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'models_to_train': models_to_train,
                'train_ratio': config['train_ratio'],
                'n_features': X_train.shape[1]
            })
            
            # Train specified models
            results = {}
            for model_name in models_to_train:
                try:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Training {model_name.upper()}")
                    logger.info(f"{'='*50}")
                    
                    if model_name == 'lstm':
                        model, train_metrics, test_metrics = available_models[model_name](
                            config, data_processor, df_with_features
                        )
                    else:
                        model, train_metrics, test_metrics = available_models[model_name](
                            config, X_train, y_train, X_test, y_test
                        )
                    
                    # Store results
                    results[model_name] = {
                        'model': model,
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics
                    }
                    
                    # Log metrics
                    mlflow.log_metrics({
                        f"{model_name}_train_r2": train_metrics['train_r2'],
                        f"{model_name}_test_r2": test_metrics['r2'],
                        f"{model_name}_test_rmse": test_metrics['rmse']
                    })
                    
                    logger.info(f"{model_name.upper()} training completed successfully!")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Create results summary
            if results:
                summary_df = pd.DataFrame([
                    {
                        'Model': model_name.upper(),
                        'Train_R2': results[model_name]['train_metrics']['train_r2'],
                        'Test_R2': results[model_name]['test_metrics']['r2'],
                        'Test_RMSE': results[model_name]['test_metrics']['rmse'],
                        'Test_MAE': results[model_name]['test_metrics']['mae']
                    }
                    for model_name in results.keys()
                ])
                
                # Save results
                results_path = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path)
                
                # Print summary
                print("\n" + "="*80)
                print("TRAINING PIPELINE RESULTS SUMMARY")
                print("="*80)
                print(summary_df.to_string(index=False))
                print("="*80)
                
                # Find best model
                best_idx = summary_df['Test_R2'].idxmax()
                best_model = summary_df.loc[best_idx, 'Model']
                best_score = summary_df.loc[best_idx, 'Test_R2']
                
                print(f"\n🏆 Best Model: {best_model}")
                print(f"🏅 Best R² Score: {best_score:.4f}")
                print("="*80)
                
                logger.info(f"Training pipeline completed successfully!")
                logger.info(f"Results saved to: {results_path}")
                
                return results
            else:
                logger.error("No models were trained successfully")
                return {}
                
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Central training pipeline for stock prediction models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--models', nargs='+', 
                      choices=['lightgbm', 'xgboost', 'random_forest', 'linear_regression', 'lstm'],
                      help='Specific models to train (default: all models)')
    args = parser.parse_args()
    
    run_training_pipeline(args.config, args.models)

if __name__ == "__main__":
    main()
