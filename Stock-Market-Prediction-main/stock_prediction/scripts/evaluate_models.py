"""
Comprehensive Model Evaluation and Benchmarking Script
Includes LightGBM and compares all models using RMSE, MAE, and R² metrics
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, ModelBenchmark
)
from stock_prediction.models.lstm_model import LSTMModel

# ML and evaluation libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_performance_plots(results: dict, output_dir: str):
    """Create performance comparison plots"""
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models_list = list(results.keys())
        r2_scores = [results[name]['test_metrics']['r2'] for name in models_list]
        rmse_scores = [results[name]['test_metrics']['rmse'] for name in models_list]
        mae_scores = [results[name]['test_metrics']['mae'] for name in models_list]
        train_r2 = [results[name]['train_metrics']['train_r2'] for name in models_list]
        
        # R² Score Comparison
        axes[0, 0].bar(models_list, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE Comparison
        axes[0, 1].bar(models_list, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('RMSE Comparison (Lower is Better)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE Comparison
        axes[1, 0].bar(models_list, mae_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('MAE Comparison (Lower is Better)')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training vs Test R²
        x = np.arange(len(models_list))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.7)
        axes[1, 1].bar(x + width/2, r2_scores, width, label='Test R²', alpha=0.7)
        axes[1, 1].set_title('Training vs Test R² Scores')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models_list, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Predicted vs Actual plots
        n_models = len(results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, result) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            
            y_pred = result['predictions']
            y_test = result['y_test']
            
            axes[row, col].scatter(y_test, y_pred, alpha=0.6, s=20)
            axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[row, col].set_xlabel('Actual Values')
            axes[row, col].set_ylabel('Predicted Values')
            axes[row, col].set_title(f'{name}: Predicted vs Actual')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2 = result['test_metrics']['r2']
            axes[row, col].text(0.05, 0.95, f'R² = {r2:.4f}', 
                               transform=axes[row, col].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating performance plots: {str(e)}")
        raise

def create_feature_importance_plots(results: dict, X_train: np.ndarray, output_dir: str):
    """Create feature importance plots for tree-based models"""
    try:
        # Find tree-based models
        tree_models = {}
        for name, result in results.items():
            if hasattr(result['model'].model, 'feature_importances_'):
                tree_models[name] = result
        
        if not tree_models:
            print("No tree-based models available for feature importance analysis")
            return
        
        print(f"Creating feature importance plots for {len(tree_models)} tree-based models...")
        
        # Create feature names
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Plot feature importance for each model
        n_models = len(tree_models)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, result) in enumerate(tree_models.items()):
            row = idx // cols
            col = idx % cols
            
            # Get feature importance
            importance = result['model'].model.feature_importances_
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[::-1]
            top_features = sorted_idx[:15]  # Top 15 features
            
            # Plot
            y_pos = np.arange(len(top_features))
            axes[row, col].barh(y_pos, importance[top_features])
            axes[row, col].set_yticks(y_pos)
            axes[row, col].set_yticklabels([feature_names[i] for i in top_features])
            axes[row, col].set_xlabel('Feature Importance')
            axes[row, col].set_title(f'{name}: Top 15 Feature Importance')
            axes[row, col].invert_yaxis()
        
        # Hide empty subplots
        for idx in range(len(tree_models), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error creating feature importance plots: {str(e)}")

def evaluate_models(config_path: str):
    """Main function to evaluate all models"""
    try:
        print("🚀 Starting Comprehensive Model Evaluation...")
        
        # Load configuration
        config = load_config(config_path)
        
        # Initialize data processor
        data_processor = DataProcessor(sequence_length=config['sequence_length'])
        
        # Load and prepare data
        print("\n📊 Loading and preparing data...")
        
        processed_data_path = config.get('processed_data_path', 'data/processed_stock_data.csv')
        if Path(processed_data_path).exists():
            print(f"Loading processed data from {processed_data_path}")
            df_with_features = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        else:
            print("Creating features from raw data...")
            df = data_processor.load_data(config['data_path'])
            df_with_features = data_processor.create_features(df)
            
            # Save processed data
            Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
            df_with_features.to_csv(processed_data_path)
            print(f"Processed data saved to {processed_data_path}")
        
        # Prepare data for ML models
        X, y = data_processor.prepare_ml_data(df_with_features)
        X_train, X_test, y_train, y_test = data_processor.split_data(
            X, y, train_ratio=config['train_ratio']
        )
        
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test data: {X_test.shape[0]} samples")
        
        # Initialize all models
        print("\n🤖 Initializing models...")
        models = {}
        
        # Linear Regression
        models['Linear Regression'] = LinearRegressionModel()
        
        # Random Forest
        models['Random Forest'] = RandomForestModel(
            n_estimators=config.get('rf_n_estimators', 100),
            max_depth=config.get('rf_max_depth', None),
            random_state=config.get('random_state', 42)
        )
        
        # XGBoost
        models['XGBoost'] = XGBoostModel(
            n_estimators=config.get('xgb_n_estimators', 100),
            max_depth=config.get('xgb_max_depth', 6),
            learning_rate=config.get('xgb_learning_rate', 0.1),
            random_state=config.get('random_state', 42)
        )
        
        # LightGBM
        lightgbm_config = config.get('lightgbm', {})
        models['LightGBM'] = LightGBMModel(
            n_estimators=lightgbm_config.get('n_estimators', 200),
            max_depth=lightgbm_config.get('max_depth', 8),
            learning_rate=lightgbm_config.get('learning_rate', 0.05),
            random_state=lightgbm_config.get('random_state', 42),
            num_leaves=lightgbm_config.get('num_leaves', 31),
            min_child_samples=lightgbm_config.get('min_child_samples', 20)
        )
        
        print(f"Initialized {len(models)} models:")
        for name in models.keys():
            print(f"  - {name}")
        
        # Train and evaluate all models
        print("\n🎯 Training and evaluating models...")
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
            
            try:
                # Train model
                train_metrics = model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store results
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'y_test': y_test,
                    'train_metrics': train_metrics,
                    'test_metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': rmse
                    }
                }
                
                print(f"✅ {name} training completed!")
                print(f"   R² Score: {r2:.4f}")
                print(f"   RMSE: {rmse:.4f}")
                print(f"   MAE: {mae:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        if not results:
            print("❌ No models were trained successfully")
            return
        
        print(f"\n🎉 Successfully trained {len(results)} models!")
        
        # Create results summary
        print("\n📋 Creating results summary...")
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Train_R2': result['train_metrics']['train_r2'],
                'Test_R2': result['test_metrics']['r2'],
                'Test_RMSE': result['test_metrics']['rmse'],
                'Test_MAE': result['test_metrics']['mae'],
                'Test_MSE': result['test_metrics']['mse']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_R2', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        # Find best model
        best_model = summary_df.iloc[0]['Model']
        best_score = summary_df.iloc[0]['Test_R2']
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"🏅 Best R² Score: {best_score:.4f}")
        print("="*80)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        output_dir = f"model_evaluation_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        create_performance_plots(results, output_dir)
        create_feature_importance_plots(results, X_train, output_dir)
        
        # Save results
        print("\n💾 Saving results...")
        results_file = f"{output_dir}/model_benchmark_results.csv"
        summary_df.to_csv(results_file, index=False)
        print(f"📁 Results saved to: {results_file}")
        
        # Save individual model predictions
        for name, result in results.items():
            pred_df = pd.DataFrame({
                'actual': result['y_test'],
                'predicted': result['predictions'],
                'residuals': result['y_test'] - result['predictions']
            })
            
            pred_file = f"{output_dir}/{name.lower().replace(' ', '_')}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            print(f"📁 {name} predictions saved to: {pred_file}")
        
        # Save models
        models_dir = f"{output_dir}/trained_models"
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        
        for name, result in results.items():
            try:
                model_file = f"{models_dir}/{name.lower().replace(' ', '_')}_model.joblib"
                result['model'].save_model(model_file)
                print(f"💾 {name} model saved to: {model_file}")
            except Exception as e:
                print(f"⚠️  Could not save {name} model: {str(e)}")
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL BENCHMARKING SUMMARY")
        print("="*100)
        
        # Performance ranking
        print("\n📊 PERFORMANCE RANKING (by Test R² Score):")
        for i, (_, row) in enumerate(summary_df.iterrows()):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{medal} {i+1:2d}. {row['Model']:<20} R²: {row['Test_R2']:.4f} | RMSE: {row['Test_RMSE']:.4f} | MAE: {row['Test_MAE']:.4f}")
        
        # Best model details
        best_name = summary_df.iloc[0]['Model']
        best_result = results[best_name]
        
        print(f"\n🏆 BEST MODEL ANALYSIS - {best_name}:")
        print(f"  • Test R² Score: {best_result['test_metrics']['r2']:.4f}")
        print(f"  • Test RMSE: {best_result['test_metrics']['rmse']:.4f}")
        print(f"  • Test MAE: {best_result['test_metrics']['mae']:.4f}")
        print(f"  • Training R²: {best_result['train_metrics']['train_r2']:.4f}")
        
        # Overfitting analysis
        train_test_diff = best_result['train_metrics']['train_r2'] - best_result['test_metrics']['r2']
        if train_test_diff > 0.1:
            print(f"  ⚠️  Potential overfitting (Train-Test R² diff: {train_test_diff:.4f})")
        else:
            print(f"  ✅ Good generalization (Train-Test R² diff: {train_test_diff:.4f})")
        
        print("="*100)
        print(f"\n✅ All results and models saved to: {output_dir}")
        
        return results, summary_df
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {str(e)}")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation and benchmarking')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    evaluate_models(args.config)

if __name__ == "__main__":
    main()
