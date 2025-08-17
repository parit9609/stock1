"""
Dedicated Training Script for LightGBM Model
Trains LightGBM model with comprehensive evaluation and model saving
"""

import argparse
import logging
from pathlib import Path
import yaml
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.ml_models import LightGBMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_training_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                         feature_importance: dict, output_dir: str):
    """Create training visualization plots"""
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Predicted vs Actual Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('LightGBM: Predicted vs Actual Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lightgbm_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residuals Plot
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('LightGBM: Residuals Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lightgbm_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance Plot
        if feature_importance:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:20]  # Top 20 features
            
            feature_names, importance_scores = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, importance_scores)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Feature Importance')
            plt.title('LightGBM: Top 20 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Time Series Plot (if dates available)
        try:
            # Create a simple time series plot of predictions
            plt.figure(figsize=(15, 6))
            plt.plot(y_true, label='Actual', alpha=0.7)
            plt.plot(y_pred, label='Predicted', alpha=0.7)
            plt.xlabel('Sample Index')
            plt.ylabel('Stock Price')
            plt.title('LightGBM: Actual vs Predicted Time Series')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/lightgbm_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create time series plot: {str(e)}")
        
        logger.info(f"Training plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating training plots: {str(e)}")
        raise

def train_lightgbm_model(config_path: str):
    """
    Train LightGBM model using configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
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
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        
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
        
        # Prepare data for ML models
        X, y = data_processor.prepare_ml_data(df_with_features)
        X_train, X_test, y_train, y_test = data_processor.split_data(
            X, y, train_ratio=config['train_ratio']
        )
        
        logger.info(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        logger.info(f"Test data: {X_test.shape[0]} samples")
        
        # Initialize LightGBM model
        lightgbm_config = config.get('lightgbm', {})
        model = LightGBMModel(
            n_estimators=lightgbm_config.get('n_estimators', 100),
            max_depth=lightgbm_config.get('max_depth', 6),
            learning_rate=lightgbm_config.get('learning_rate', 0.1),
            random_state=lightgbm_config.get('random_state', 42),
            num_leaves=lightgbm_config.get('num_leaves', 31),
            min_child_samples=lightgbm_config.get('min_child_samples', 20)
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'model_type': 'LightGBM',
                'n_estimators': lightgbm_config.get('n_estimators', 100),
                'max_depth': lightgbm_config.get('max_depth', 6),
                'learning_rate': lightgbm_config.get('learning_rate', 0.1),
                'num_leaves': lightgbm_config.get('num_leaves', 31),
                'min_child_samples': lightgbm_config.get('min_child_samples', 20),
                'train_ratio': config['train_ratio'],
                'n_features': X_train.shape[1]
            })
            
            # Train model
            logger.info("Training LightGBM model...")
            train_metrics = model.fit(X_train, y_train)
            
            # Log training metrics
            mlflow.log_metrics({
                'train_mse': train_metrics['train_mse'],
                'train_mae': train_metrics['train_mae'],
                'train_r2': train_metrics['train_r2']
            })
            
            logger.info("Training completed!")
            logger.info(f"Training R²: {train_metrics['train_r2']:.4f}")
            logger.info(f"Training MSE: {train_metrics['train_mse']:.4f}")
            logger.info(f"Training MAE: {train_metrics['train_mae']:.4f}")
            
            # Evaluate model
            logger.info("Evaluating model on test set...")
            test_metrics = model.evaluate(X_test, y_test)
            
            # Log test metrics
            mlflow.log_metrics({
                'test_mse': test_metrics['mse'],
                'test_mae': test_metrics['mae'],
                'test_r2': test_metrics['r2'],
                'test_rmse': test_metrics['rmse']
            })
            
            logger.info("Test Results:")
            logger.info(f"  R² Score: {test_metrics['r2']:.4f}")
            logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
            logger.info(f"  MAE: {test_metrics['mae']:.4f}")
            logger.info(f"  MSE: {test_metrics['mse']:.4f}")
            
            # Make predictions for visualization
            y_pred = model.predict(X_test)
            
            # Get feature importance if available
            feature_importance = {}
            try:
                if hasattr(model.model, 'feature_importances_'):
                    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                    feature_importance = dict(zip(feature_names, model.model.feature_importances_))
                    logger.info("Feature importance extracted successfully")
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {str(e)}")
            
            # Create visualizations
            output_dir = f"lightgbm_training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            create_training_plots(y_test, y_pred, feature_importance, output_dir)
            
            # Log visualization artifacts
            for plot_file in Path(output_dir).glob("*.png"):
                mlflow.log_artifact(str(plot_file))
            
            # Save model
            model_path = config.get('lightgbm_model_path', 'models/lightgbm_model.joblib')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
            
            # Save predictions for analysis
            predictions_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residuals': y_test - y_pred
            })
            
            predictions_path = f"lightgbm_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            predictions_df.to_csv(predictions_path)
            mlflow.log_artifact(predictions_path)
            
            # Save feature importance if available
            if feature_importance:
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in feature_importance.items()
                ]).sort_values('importance', ascending=False)
                
                importance_path = f"lightgbm_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
            
            logger.info(f"LightGBM model saved to: {model_path}")
            logger.info(f"Training plots saved to: {output_dir}")
            logger.info(f"Predictions saved to: {predictions_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("LIGHTGBM TRAINING SUMMARY")
            print("="*80)
            print(f"Training R²: {train_metrics['train_r2']:.4f}")
            print(f"Test R²: {test_metrics['r2']:.4f}")
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"Test MAE: {test_metrics['mae']:.4f}")
            print(f"Model saved to: {model_path}")
            print("="*80)
            
            return model, test_metrics
            
    except Exception as e:
        logger.error(f"Error in LightGBM training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model for stock prediction')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    train_lightgbm_model(args.config)

if __name__ == "__main__":
    main()
