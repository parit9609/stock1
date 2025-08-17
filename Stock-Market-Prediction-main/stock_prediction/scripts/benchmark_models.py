"""
Script to benchmark different models for stock prediction
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
from stock_prediction.models.lstm_model import LSTMModel
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, ModelBenchmark
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_visualizations(benchmark_results: pd.DataFrame, output_dir: str):
    """Create visualization plots for benchmark results"""
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Test R2 Score
        axes[0, 0].bar(benchmark_results['Model'], benchmark_results['Test_R2'])
        axes[0, 0].set_title('Test R² Score')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Test RMSE
        axes[0, 1].bar(benchmark_results['Model'], benchmark_results['Test_RMSE'])
        axes[0, 1].set_title('Test RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training vs Test R2
        axes[1, 0].scatter(benchmark_results['Train_R2'], benchmark_results['Test_R2'])
        for i, model in enumerate(benchmark_results['Model']):
            axes[1, 0].annotate(model, (benchmark_results['Train_R2'].iloc[i], 
                                       benchmark_results['Test_R2'].iloc[i]))
        axes[1, 0].set_xlabel('Training R²')
        axes[1, 0].set_ylabel('Test R²')
        axes[1, 0].set_title('Training vs Test R²')
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # Test MSE vs MAE
        axes[1, 1].scatter(benchmark_results['Test_MSE'], benchmark_results['Test_MAE'])
        for i, model in enumerate(benchmark_results['Model']):
            axes[1, 1].annotate(model, (benchmark_results['Test_MSE'].iloc[i], 
                                       benchmark_results['Test_MAE'].iloc[i]))
        axes[1, 1].set_xlabel('Test MSE')
        axes[1, 1].set_ylabel('Test MAE')
        axes[1, 1].set_title('Test MSE vs MAE')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Metrics Heatmap
        metrics_cols = ['Train_MSE', 'Train_MAE', 'Train_R2', 'Test_MSE', 'Test_MAE', 'Test_R2', 'Test_RMSE']
        metrics_df = benchmark_results[metrics_cols].copy()
        metrics_df.index = benchmark_results['Model']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Metric Value'})
        plt.title('Detailed Model Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance Ranking
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Rankings', fontsize=16, fontweight='bold')
        
        # R2 Ranking
        r2_ranked = benchmark_results.sort_values('Test_R2', ascending=True)
        axes[0].barh(r2_ranked['Model'], r2_ranked['Test_R2'])
        axes[0].set_title('R² Score Ranking')
        axes[0].set_xlabel('R² Score')
        
        # RMSE Ranking
        rmse_ranked = benchmark_results.sort_values('Test_RMSE', ascending=False)
        axes[1].barh(rmse_ranked['Model'], rmse_ranked['Test_RMSE'])
        axes[1].set_title('RMSE Ranking')
        axes[1].set_xlabel('RMSE')
        
        # MAE Ranking
        mae_ranked = benchmark_results.sort_values('Test_MAE', ascending=False)
        axes[2].barh(mae_ranked['Model'], mae_ranked['Test_MAE'])
        axes[2].set_title('MAE Ranking')
        axes[2].set_xlabel('MAE')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

def benchmark_models(config_path: str):
    """
    Benchmark different models using configuration from YAML file
    
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
        df = data_processor.load_data(config['data_path'])
        
        # Create features for ML models
        logger.info("Creating features for ML models...")
        df_with_features = data_processor.create_features(df)
        
        # Prepare data for ML models
        X_ml, y_ml = data_processor.prepare_ml_data(df_with_features)
        X_train_ml, X_test_ml, y_train_ml, y_test_ml = data_processor.split_data(
            X_ml, y_ml, train_ratio=config['train_ratio']
        )
        
        # Prepare data for LSTM
        logger.info("Preparing data for LSTM...")
        X_lstm, y_lstm = data_processor.prepare_data(df, target_column=config['target_column'])
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = data_processor.split_data(
            X_lstm, y_lstm, train_ratio=config['train_ratio']
        )
        
        # Initialize benchmark
        benchmark = ModelBenchmark()
        
        # Add ML models
        logger.info("Adding ML models to benchmark...")
        benchmark.add_model(LinearRegressionModel())
        benchmark.add_model(RandomForestModel(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', None),
            random_state=config.get('random_state', 42)
        ))
        benchmark.add_model(XGBoostModel(
            n_estimators=config.get('xgb_n_estimators', 100),
            max_depth=config.get('xgb_max_depth', 6),
            learning_rate=config.get('xgb_learning_rate', 0.1),
            random_state=config.get('random_state', 42)
        ))
        benchmark.add_model(LightGBMModel(
            n_estimators=config.get('lgb_n_estimators', 100),
            max_depth=config.get('lgb_max_depth', 6),
            learning_rate=config.get('lgb_learning_rate', 0.1),
            random_state=config.get('random_state', 42),
            num_leaves=config.get('lgb_num_leaves', 31),
            min_child_samples=config.get('lgb_min_child_samples', 20)
        ))
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'sequence_length': config['sequence_length'],
                'n_features': config['n_features'],
                'train_ratio': config['train_ratio'],
                'rf_n_estimators': config.get('rf_n_estimators', 100),
                'xgb_n_estimators': config.get('xgb_n_estimators', 100),
                'lgb_n_estimators': config.get('lgb_n_estimators', 100)
            })
            
            # Run ML model benchmark
            logger.info("Running ML model benchmark...")
            ml_results = benchmark.run_benchmark(
                X_train_ml, y_train_ml, X_test_ml, y_test_ml
            )
            
            # Train and evaluate LSTM separately
            logger.info("Training and evaluating LSTM model...")
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
            
            # Add LSTM results to benchmark
            lstm_result = {
                'Model': 'LSTM',
                'Train_MSE': history['loss'][-1],
                'Train_MAE': history['loss'][-1],  # LSTM doesn't provide MAE during training
                'Train_R2': 1 - history['loss'][-1],  # Approximation
                'Test_MSE': lstm_mse,
                'Test_MAE': lstm_mae,
                'Test_R2': lstm_r2,
                'Test_RMSE': lstm_rmse
            }
            
            # Combine results
            all_results = pd.concat([ml_results, pd.DataFrame([lstm_result])], ignore_index=True)
            
            # Log metrics
            for _, row in all_results.iterrows():
                mlflow.log_metrics({
                    f"{row['Model']}_Test_R2": row['Test_R2'],
                    f"{row['Model']}_Test_RMSE": row['Test_RMSE'],
                    f"{row['Model']}_Test_MAE": row['Test_MAE']
                })
            
            # Get best model
            best_model, best_score = benchmark.get_best_model('Test_R2')
            logger.info(f"Best model: {best_model} with R² score: {best_score:.4f}")
            
            # Save results
            results_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            all_results.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            
            # Create visualizations
            output_dir = f"benchmark_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            create_visualizations(all_results, output_dir)
            
            # Log visualization artifacts
            for plot_file in Path(output_dir).glob("*.png"):
                mlflow.log_artifact(str(plot_file))
            
            # Save best model
            if best_model != 'LSTM':
                best_model_instance = benchmark.models[best_model]
                best_model_path = f"best_{best_model.lower().replace(' ', '_')}.joblib"
                best_model_instance.save_model(best_model_path)
                mlflow.log_artifact(best_model_path)
            else:
                # LSTM is already saved
                mlflow.log_artifact(config['model_path'])
            
            logger.info(f"Benchmark completed successfully!")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Plots saved to: {output_dir}")
            logger.info(f"Best model: {best_model}")
            
            # Print summary
            print("\n" + "="*80)
            print("BENCHMARK RESULTS SUMMARY")
            print("="*80)
            print(all_results.to_string(index=False))
            print(f"\nBest Model: {best_model}")
            print(f"Best R² Score: {best_score:.4f}")
            print("="*80)
            
    except Exception as e:
        logger.error(f"Error in benchmarking: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Benchmark different models for stock prediction')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    benchmark_models(args.config)

if __name__ == "__main__":
    main()
