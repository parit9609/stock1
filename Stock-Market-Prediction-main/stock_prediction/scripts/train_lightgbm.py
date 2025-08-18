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
    Train LightGBM model using configuration from YAML file with time-series CV
    
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
        
        # Use time-series cross-validation instead of simple split
        cv_config = config.get('cross_validation', {})
        n_splits = cv_config.get('n_splits', 5)
        test_size = cv_config.get('test_size')
        gap = cv_config.get('gap', 0)
        
        logger.info(f"Using time-series cross-validation with {n_splits} splits")
        logger.info(f"Test size: {test_size}, Gap: {gap}")
        
        # Create time-series splits
        time_series_splits = data_processor.create_time_series_splits(
            X, y, n_splits=n_splits, test_size=test_size, gap=gap
        )
        
        # Initialize LightGBM model with early stopping
        lightgbm_config = config.get('lightgbm', {})
        model = LightGBMModel(
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
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'model_type': 'LightGBM',
                'n_estimators': lightgbm_config.get('n_estimators', 200),
                'max_depth': lightgbm_config.get('max_depth', 8),
                'learning_rate': lightgbm_config.get('learning_rate', 0.05),
                'num_leaves': lightgbm_config.get('num_leaves', 31),
                'min_child_samples': lightgbm_config.get('min_child_samples', 20),
                'early_stopping_rounds': lightgbm_config.get('early_stopping_rounds', 50),
                'n_splits': n_splits,
                'gap': gap,
                'n_features': X.shape[1]
            })
            
            # Perform time-series cross-validation
            logger.info("Running time-series cross-validation...")
            cv_results = model.cross_validate(X, y, n_splits=n_splits, test_size=test_size, gap=gap)
            
            # Log cross-validation metrics
            if 'overall_metrics' in cv_results:
                overall = cv_results['overall_metrics']
                mlflow.log_metrics({
                    'cv_r2_mean': overall.get('r2_mean', 0),
                    'cv_r2_std': overall.get('r2_std', 0),
                    'cv_rmse_mean': overall.get('rmse_mean', 0),
                    'cv_rmse_std': overall.get('rmse_std', 0),
                    'cv_mae_mean': overall.get('mae_mean', 0),
                    'cv_mae_std': overall.get('mae_std', 0)
                })
                
                logger.info("Cross-validation Results:")
                logger.info(f"  R²: {overall.get('r2_mean', 0):.4f} ± {overall.get('r2_std', 0):.4f}")
                logger.info(f"  RMSE: {overall.get('rmse_mean', 0):.4f} ± {overall.get('rmse_std', 0):.4f}")
                logger.info(f"  MAE: {overall.get('mae_mean', 0):.4f} ± {overall.get('mae_std', 0):.4f}")
            
            # Train final model on all data for production use
            logger.info("Training final model on all data...")
            train_metrics = model.fit(X, y)
            
            # Log training metrics
            mlflow.log_metrics({
                'final_train_mse': train_metrics['train_mse'],
                'final_train_mae': train_metrics['train_mae'],
                'final_train_r2': train_metrics['train_r2']
            })
            
            logger.info("Final model training completed!")
            logger.info(f"Final Training R²: {train_metrics['train_r2']:.4f}")
            logger.info(f"Final Training MSE: {train_metrics['train_mse']:.4f}")
            logger.info(f"Final Training MAE: {train_metrics['train_mae']:.4f}")
            
            # Get feature importance if available
            feature_importance = {}
            try:
                if hasattr(model.model, 'feature_importances_'):
                    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                    feature_importance = dict(zip(feature_names, model.model.feature_importances_))
                    logger.info("Feature importance extracted successfully")
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {str(e)}")
            
            # Create visualizations
            output_dir = f"lightgbm_training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            create_training_plots(y, model.predict(X), feature_importance, output_dir)
            
            # Log visualization artifacts
            for plot_file in Path(output_dir).glob("*.png"):
                mlflow.log_artifact(str(plot_file))
            
            # Save model
            model_path = config.get('lightgbm_model_path', 'models/lightgbm_model.joblib')
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
            
            # Save cross-validation results
            cv_results_path = f"lightgbm_cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(cv_results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json.dump(cv_results, f, default=lambda x: float(x) if hasattr(x, 'dtype') else x, indent=2)
            mlflow.log_artifact(cv_results_path)
            
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
            logger.info(f"CV results saved to: {cv_results_path}")
            
            # Print summary
            print("\n" + "="*80)
            print("LIGHTGBM TRAINING SUMMARY (WITH TIME-SERIES CV)")
            print("="*80)
            if 'overall_metrics' in cv_results:
                overall = cv_results['overall_metrics']
                print(f"Cross-Validation R²: {overall.get('r2_mean', 0):.4f} ± {overall.get('r2_std', 0):.4f}")
                print(f"Cross-Validation RMSE: {overall.get('rmse_mean', 0):.4f} ± {overall.get('rmse_std', 0):.4f}")
                print(f"Cross-Validation MAE: {overall.get('mae_mean', 0):.4f} ± {overall.get('mae_std', 0):.4f}")
            print(f"Final Training R²: {train_metrics['train_r2']:.4f}")
            print(f"Model saved to: {model_path}")
            print("="*80)
            
            return model, cv_results
            
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
