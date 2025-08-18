"""
Consistent Validation Demo Script
Demonstrates the consistent use of walk-forward CV and early stopping across all models
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
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, ModelBenchmark
)
from stock_prediction.data.time_series_cv import WalkForwardValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_validation_comparison_plots(cv_results: dict, output_dir: str):
    """Create comparison plots for validation results"""
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. R² Comparison across folds
        plt.figure(figsize=(12, 8))
        models = list(cv_results.keys())
        n_models = len(models)
        
        # Extract R² scores for each fold
        r2_data = []
        model_names = []
        for model_name, results in cv_results.items():
            if 'fold_metrics' in results:
                for fold_idx, fold_metrics in enumerate(results['fold_metrics']):
                    r2_data.append(fold_metrics.get('r2', 0))
                    model_names.append(f"{model_name} (Fold {fold_idx+1})")
        
        # Create box plot
        plt.figure(figsize=(15, 8))
        r2_by_model = {}
        for model_name in models:
            if 'fold_metrics' in cv_results[model_name]:
                r2_by_model[model_name] = [
                    fold.get('r2', 0) for fold in cv_results[model_name]['fold_metrics']
                ]
        
        if r2_by_model:
            plt.boxplot(r2_by_model.values(), labels=r2_by_model.keys())
            plt.ylabel('R² Score')
            plt.title('R² Score Distribution Across Folds (Walk-Forward CV)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/r2_comparison_across_folds.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. RMSE Comparison across folds
        plt.figure(figsize=(15, 8))
        rmse_by_model = {}
        for model_name in models:
            if 'fold_metrics' in cv_results[model_name]:
                rmse_by_model[model_name] = [
                    fold.get('rmse', 0) for fold in cv_results[model_name]['fold_metrics']
                ]
        
        if rmse_by_model:
            plt.boxplot(rmse_by_model.values(), labels=rmse_by_model.keys())
            plt.ylabel('RMSE')
            plt.title('RMSE Distribution Across Folds (Walk-Forward CV)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/rmse_comparison_across_folds.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Training time comparison
        plt.figure(figsize=(12, 8))
        training_times = []
        model_labels = []
        for model_name, results in cv_results.items():
            if 'overall_metrics' in results and 'training_time_mean' in results['overall_metrics']:
                training_times.append(results['overall_metrics']['training_time_mean'])
                model_labels.append(model_name)
        
        if training_times:
            plt.bar(model_labels, training_times)
            plt.ylabel('Training Time (seconds)')
            plt.title('Average Training Time per Model')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/training_time_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Stability analysis (coefficient of variation)
        plt.figure(figsize=(12, 8))
        stability_metrics = []
        model_labels = []
        for model_name, results in cv_results.items():
            if 'overall_metrics' in results:
                overall = results['overall_metrics']
                if 'r2_mean' in overall and 'r2_std' in overall and overall['r2_mean'] > 0:
                    cv = overall['r2_std'] / overall['r2_mean']  # Coefficient of variation
                    stability_metrics.append(cv)
                    model_labels.append(model_name)
        
        if stability_metrics:
            plt.bar(model_labels, stability_metrics)
            plt.ylabel('Coefficient of Variation (Lower = More Stable)')
            plt.title('Model Stability Analysis (R² CV across folds)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/stability_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Validation comparison plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating validation comparison plots: {str(e)}")
        raise

def run_consistent_validation_demo(config_path: str):
    """
    Run demonstration of consistent validation across all models
    
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
        
        # Get CV configuration
        cv_config = config.get('cross_validation', {})
        n_splits = cv_config.get('n_splits', 5)
        test_size = cv_config.get('test_size')
        gap = cv_config.get('gap', 0)
        
        logger.info(f"Using time-series cross-validation with {n_splits} splits")
        logger.info(f"Test size: {test_size}, Gap: {gap}")
        logger.info(f"Total data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'validation_method': 'walk_forward_cv',
                'n_splits': n_splits,
                'gap': gap,
                'n_features': X.shape[1],
                'early_stopping': True
            })
            
            # Initialize models with early stopping
            models = {
                'Linear Regression': LinearRegressionModel(),
                'Random Forest': RandomForestModel(
                    n_estimators=config.get('rf_n_estimators', 100),
                    max_depth=config.get('rf_max_depth', None),
                    random_state=config.get('random_state', 42)
                ),
                'XGBoost': XGBoostModel(
                    n_estimators=config.get('xgb_n_estimators', 100),
                    max_depth=config.get('xgb_max_depth', 6),
                    learning_rate=config.get('xgb_learning_rate', 0.1),
                    random_state=config.get('random_state', 42),
                    early_stopping_rounds=config.get('xgb_early_stopping_rounds', 50),
                    eval_metric=config.get('xgb_eval_metric', 'rmse')
                ),
                'LightGBM': LightGBMModel(
                    n_estimators=config.get('lightgbm', {}).get('n_estimators', 200),
                    max_depth=config.get('lightgbm', {}).get('max_depth', 8),
                    learning_rate=config.get('lightgbm', {}).get('learning_rate', 0.05),
                    random_state=config.get('lightgbm', {}).get('random_state', 42),
                    num_leaves=config.get('lightgbm', {}).get('num_leaves', 31),
                    min_child_samples=config.get('lightgbm', {}).get('min_child_samples', 20),
                    early_stopping_rounds=config.get('lightgbm', {}).get('early_stopping_rounds', 50),
                    eval_metric=config.get('lightgbm', {}).get('metric', 'rmse'),
                    subsample=config.get('lightgbm', {}).get('subsample', 0.8),
                    colsample_bytree=config.get('lightgbm', {}).get('colsample_bytree', 0.8),
                    reg_alpha=config.get('lightgbm', {}).get('reg_alpha', 0.1),
                    reg_lambda=config.get('lightgbm', {}).get('reg_lambda', 0.1)
                )
            }
            
            # Run consistent validation across all models
            cv_results = {}
            for model_name, model in models.items():
                try:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Running consistent validation for {model_name}")
                    logger.info(f"{'='*50}")
                    
                    # Perform time-series cross-validation
                    cv_results[model_name] = model.cross_validate(X, y, n_splits, test_size, gap)
                    
                    # Log CV metrics
                    if 'overall_metrics' in cv_results[model_name]:
                        overall = cv_results[model_name]['overall_metrics']
                        mlflow.log_metrics({
                            f"{model_name.lower().replace(' ', '_')}_cv_r2": overall.get('r2_mean', 0),
                            f"{model_name.lower().replace(' ', '_')}_cv_rmse": overall.get('rmse_mean', 0),
                            f"{model_name.lower().replace(' ', '_')}_cv_mae": overall.get('mae_mean', 0),
                            f"{model_name.lower().replace(' ', '_')}_cv_stability": overall.get('r2_std', 0) / max(overall.get('r2_mean', 1), 1e-6)
                        })
                        
                        logger.info(f"{model_name} CV Results:")
                        logger.info(f"  R²: {overall.get('r2_mean', 0):.4f} ± {overall.get('r2_std', 0):.4f}")
                        logger.info(f"  RMSE: {overall.get('rmse_mean', 0):.4f} ± {overall.get('rmse_std', 0):.4f}")
                        logger.info(f"  MAE: {overall.get('mae_mean', 0):.4f} ± {overall.get('mae_std', 0):.4f}")
                        logger.info(f"  Training Time: {overall.get('training_time_mean', 0):.2f}s ± {overall.get('training_time_std', 0):.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error validating {model_name}: {str(e)}")
                    continue
            
            # Create comparison plots
            output_dir = f"consistent_validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            create_validation_comparison_plots(cv_results, output_dir)
            
            # Log visualization artifacts
            for plot_file in Path(output_dir).glob("*.png"):
                mlflow.log_artifact(str(plot_file))
            
            # Save CV results
            cv_results_path = f"consistent_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(cv_results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json.dump(cv_results, f, default=lambda x: float(x) if hasattr(x, 'dtype') else x, indent=2)
            mlflow.log_artifact(cv_results_path)
            
            # Create summary table
            summary_data = []
            for model_name, results in cv_results.items():
                if 'overall_metrics' in results:
                    overall = results['overall_metrics']
                    summary_data.append({
                        'Model': model_name,
                        'CV_R2_Mean': f"{overall.get('r2_mean', 0):.4f}",
                        'CV_R2_Std': f"±{overall.get('r2_std', 0):.4f}",
                        'CV_RMSE_Mean': f"{overall.get('rmse_mean', 0):.4f}",
                        'CV_RMSE_Std': f"±{overall.get('rmse_std', 0):.4f}",
                        'CV_MAE_Mean': f"{overall.get('mae_mean', 0):.4f}",
                        'CV_MAE_Std': f"±{overall.get('mae_std', 0):.4f}",
                        'Training_Time_Mean': f"{overall.get('training_time_mean', 0):.2f}s",
                        'Training_Time_Std': f"±{overall.get('training_time_std', 0):.2f}s",
                        'Stability': f"{overall.get('r2_std', 0) / max(overall.get('r2_mean', 1), 1e-6):.4f}"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = f"consistent_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                summary_df.to_csv(summary_path, index=False)
                mlflow.log_artifact(summary_path)
                
                # Print summary
                print("\n" + "="*120)
                print("CONSISTENT VALIDATION RESULTS SUMMARY")
                print("="*120)
                print(summary_df.to_string(index=False))
                print("="*120)
                
                # Find best model by R²
                best_idx = summary_df['CV_R2_Mean'].astype(float).idxmax()
                best_model = summary_df.loc[best_idx, 'Model']
                best_score = summary_df.loc[best_idx, 'CV_R2_Mean']
                
                print(f"\n🏆 Best Model by R²: {best_model}")
                print(f"🏅 Best CV R² Score: {best_score}")
                print("="*120)
            
            logger.info(f"Consistent validation demo completed successfully!")
            logger.info(f"Results saved to: {cv_results_path}")
            logger.info(f"Plots saved to: {output_dir}")
            
            return cv_results
            
    except Exception as e:
        logger.error(f"Error in consistent validation demo: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Demonstrate consistent validation across all models')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    run_consistent_validation_demo(args.config)

if __name__ == "__main__":
    main()
