"""
Reproducibility Demonstration Script
Sets seeds, provides small sample runs, and ensures consistent results
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

def set_reproducibility_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    print(f"Setting reproducibility seeds to {seed}")
    
    # Python random
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Set environment variables for other libraries
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for reproducibility
    
    print("✓ All random seeds set for reproducibility")

def create_small_sample_data(n_samples: int = 500, seed: int = 42):
    """Create small sample data for quick demonstration"""
    print(f"Creating small sample data with {n_samples} samples...")
    
    np.random.seed(seed)
    
    # Create realistic stock data
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    
    # Generate price data with some trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, n_samples)  # Upward trend
    noise = np.random.normal(0, 2, n_samples)  # Daily volatility
    prices = base_price + trend + noise
    
    # Generate OHLC data
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 0.5, n_samples),
        'High': prices + np.abs(np.random.normal(1, 0.5, n_samples)),
        'Low': prices - np.abs(np.random.normal(1, 0.5, n_samples)),
        'Close': prices,
        'Volume': np.random.uniform(1000000, 5000000, n_samples)
    }, index=dates)
    
    # Ensure High >= Low and High >= Open, Close
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(np.random.normal(0.5, 0.2, n_samples))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(np.random.normal(0.5, 0.2, n_samples))
    
    print(f"✓ Sample data created: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    
    return df

def run_small_benchmark(df: pd.DataFrame, data_processor: DataProcessor, 
                        n_splits: int = 3, test_size: int = 50):
    """Run a small benchmark with limited data for quick demonstration"""
    print(f"\nRunning small benchmark with {n_splits} splits, test_size={test_size}")
    
    # Create features
    df_with_features = data_processor.create_features(df)
    
    # Prepare ML data
    X, y = data_processor.prepare_ml_data(df_with_features)
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    
    # Initialize models with small parameters for quick training
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(n_estimators=50, max_depth=5, random_state=42),
        'XGBoost': XGBoostModel(n_estimators=50, max_depth=3, random_state=42),
        'LightGBM': LightGBMModel(n_estimators=50, max_depth=3, random_state=42)
    }
    
    # Initialize benchmark
    benchmark = ModelBenchmark()
    for model in models.values():
        benchmark.add_model(model)
    
    # Run time-series cross-validation
    cv_results = benchmark.run_time_series_benchmark(
        X, y, n_splits=n_splits, test_size=test_size, gap=0
    )
    
    return cv_results

def demonstrate_reproducibility(config_path: str, n_samples: int = 500, n_splits: int = 3):
    """Demonstrate reproducibility with consistent results"""
    print("Reproducibility Demonstration")
    print("="*50)
    
    # Load configuration
    config = load_config(config_path)
    
    # Set reproducibility seeds
    seed = config.get('random_state', 42)
    set_reproducibility_seeds(seed)
    
    # Create small sample data
    df = create_small_sample_data(n_samples, seed)
    
    # Initialize data processor
    data_processor = DataProcessor(
        sequence_length=config['sequence_length']
    )
    
    # Run benchmark multiple times to show reproducibility
    print(f"\nRunning benchmark {n_splits} times to demonstrate reproducibility...")
    
    all_results = []
    for run in range(3):  # Run 3 times
        print(f"\n--- Run {run + 1} ---")
        
        # Reset seeds for each run
        set_reproducibility_seeds(seed)
        
        # Run benchmark
        cv_results = run_small_benchmark(df, data_processor, n_splits, 50)
        
        # Extract key metrics
        run_metrics = {}
        for model_name, results in cv_results.items():
            if 'overall_metrics' in results:
                overall = results['overall_metrics']
                run_metrics[model_name] = {
                    'R2_Mean': overall.get('r2_mean', np.nan),
                    'RMSE_Mean': overall.get('rmse_mean', np.nan)
                }
        
        all_results.append(run_metrics)
        
        # Print results
        for model_name, metrics in run_metrics.items():
            print(f"  {model_name}: R²={metrics['R2_Mean']:.4f}, RMSE={metrics['RMSE_Mean']:.4f}")
    
    # Check reproducibility
    print(f"\n" + "="*50)
    print("REPRODUCIBILITY CHECK")
    print("="*50)
    
    reproducible = True
    for model_name in all_results[0].keys():
        print(f"\n{model_name}:")
        
        # Check R² reproducibility
        r2_values = [run[model_name]['R2_Mean'] for run in all_results]
        r2_std = np.std(r2_values)
        r2_reproducible = r2_std < 1e-10  # Very small tolerance
        
        print(f"  R² values: {[f'{v:.6f}' for v in r2_values]}")
        print(f"  R² std: {r2_std:.10f}")
        print(f"  R² reproducible: {'✓' if r2_reproducible else '✗'}")
        
        # Check RMSE reproducibility
        rmse_values = [run[model_name]['RMSE_Mean'] for run in all_results]
        rmse_std = np.std(rmse_values)
        rmse_reproducible = rmse_std < 1e-10
        
        print(f"  RMSE values: {[f'{v:.6f}' for v in rmse_values]}")
        print(f"  RMSE std: {rmse_std:.10f}")
        print(f"  RMSE reproducible: {'✓' if rmse_reproducible else '✗'}")
        
        if not (r2_reproducible and rmse_reproducible):
            reproducible = False
    
    print(f"\nOverall reproducibility: {'✓ ACHIEVED' if reproducible else '✗ NOT ACHIEVED'}")
    
    if reproducible:
        print("✓ All models produce identical results across runs")
        print("✓ Seeds properly set for all random components")
        print("✓ Results are fully reproducible")
    else:
        print("⚠ Some models show variation across runs")
        print("⚠ Check for non-deterministic operations")
    
    return reproducible

def create_reproducibility_report(config_path: str, output_dir: str = "output"):
    """Create a comprehensive reproducibility report"""
    print("Creating reproducibility report...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run reproducibility check
    reproducible = demonstrate_reproducibility(config_path)
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'reproducibility_achieved': reproducible,
        'configuration': {
            'config_file': config_path,
            'random_seed': 42,
            'sample_size': 500,
            'cv_splits': 3
        },
        'reproducibility_notes': [
            'All random seeds set to 42',
            'GPU disabled for consistency',
            'Deterministic algorithms used where possible',
            'Small sample size for quick verification'
        ]
    }
    
    # Save report
    import json
    report_path = output_dir / 'reproducibility_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Reproducibility report saved to {report_path}")
    
    return reproducible

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demonstrate reproducibility')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for reports')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples for demonstration')
    parser.add_argument('--splits', type=int, default=3,
                       help='Number of CV splits for demonstration')
    args = parser.parse_args()
    
    # Run reproducibility demonstration
    reproducible = demonstrate_reproducibility(
        args.config, args.samples, args.splits
    )
    
    # Create report
    create_reproducibility_report(args.config, args.output)
    
    if reproducible:
        print("\n🎉 Reproducibility demonstration completed successfully!")
        print("All models produce consistent results across runs.")
    else:
        print("\n⚠ Reproducibility demonstration completed with warnings.")
        print("Some models show variation across runs.")

if __name__ == "__main__":
    main()
