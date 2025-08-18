"""
Quick Start Example for Stock Market Prediction System
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, ModelBenchmark
)

def create_sample_data():
    """Create sample stock data for demonstration"""
    np.random.seed(42)
    
    # Generate 1000 days of sample stock data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate realistic stock prices with trends and volatility
    base_price = 100
    trend = np.linspace(0, 20, 1000)  # Upward trend
    noise = np.random.normal(0, 2, 1000)  # Daily noise
    seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 1000))  # Seasonal pattern
    
    close_prices = base_price + trend + noise + seasonal
    
    # Generate other OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 1))
        high = close + volatility
        low = close - volatility
        open_price = close + np.random.normal(0, 0.5)
        volume = int(np.random.uniform(100000, 1000000))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

def demonstrate_feature_engineering():
    """Demonstrate the enhanced feature engineering capabilities"""
    print("🔧 Creating sample stock data...")
    df = create_sample_data()
    
    print("📊 Original data shape:", df.shape)
    print("📈 Sample data:")
    print(df.head())
    
    print("\n🔧 Creating technical indicators and features...")
    processor = DataProcessor(sequence_length=60)
    df_with_features = processor.create_features(df)
    
    print("📊 Enhanced data shape:", df_with_features.shape)
    print("🔍 New features created:")
    
    # Show some of the new features
    new_features = [col for col in df_with_features.columns if col not in df.columns]
    for i, feature in enumerate(new_features[:10]):  # Show first 10
        print(f"  {i+1:2d}. {feature}")
    
    if len(new_features) > 10:
        print(f"  ... and {len(new_features) - 10} more features")
    
    return df_with_features

def demonstrate_ml_models():
    """Demonstrate training and comparing ML models"""
    print("\n🤖 Demonstrating Machine Learning Models...")
    
    # Prepare data
    df_with_features = demonstrate_feature_engineering()
    processor = DataProcessor()
    
    X, y = processor.prepare_ml_data(df_with_features)
    X_train, X_test, y_train, y_test = processor.split_data(X, y, train_ratio=0.8)
    
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Test data: {X_test.shape[0]} samples")
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegressionModel(),
        "Random Forest": RandomForestModel(n_estimators=50, max_depth=5),
        "XGBoost": XGBoostModel(n_estimators=50, max_depth=4, learning_rate=0.1),
        "LightGBM": LightGBMModel(n_estimators=50, max_depth=4, learning_rate=0.1)
    }
    
    # Train and evaluate each model
    results = []
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        try:
            # Train model
            train_metrics = model.fit(X_train, y_train)
            
            # Evaluate model
            test_metrics = model.evaluate(X_test, y_test)
            
            # Store results
            result = {
                'Model': name,
                'Train_R2': train_metrics['train_r2'],
                'Test_R2': test_metrics['r2'],
                'Test_RMSE': test_metrics['rmse'],
                'Test_MAE': test_metrics['mae']
            }
            results.append(result)
            
            print(f"✅ {name} completed:")
            print(f"   Training R²: {train_metrics['train_r2']:.4f}")
            print(f"   Test R²: {test_metrics['r2']:.4f}")
            print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
            
        except Exception as e:
            print(f"❌ Error training {name}: {str(e)}")
    
    return pd.DataFrame(results)

def demonstrate_benchmarking():
    """Demonstrate the benchmarking system"""
    print("\n🏆 Demonstrating Model Benchmarking...")
    
    # Prepare data
    df_with_features = demonstrate_feature_engineering()
    processor = DataProcessor()
    
    X, y = processor.prepare_ml_data(df_with_features)
    X_train, X_test, y_train, y_test = processor.split_data(X, y, train_ratio=0.8)
    
    # Initialize benchmark
    benchmark = ModelBenchmark()
    
    # Add models
    benchmark.add_model(LinearRegressionModel())
    benchmark.add_model(RandomForestModel(n_estimators=50, max_depth=5))
    benchmark.add_model(XGBoostModel(n_estimators=50, max_depth=4))
    benchmark.add_model(LightGBMModel(n_estimators=50, max_depth=4))
    
    # Run benchmark
    print("🏁 Running comprehensive benchmark...")
    results = benchmark.run_benchmark(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_model, best_score = benchmark.get_best_model('Test_R2')
    
    print("\n🏆 BENCHMARK RESULTS:")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)
    print(f"🥇 Best Model: {best_model}")
    print(f"🏅 Best R² Score: {best_score:.4f}")
    
    return results

def demonstrate_prediction():
    """Demonstrate making predictions with trained models"""
    print("\n🔮 Demonstrating Predictions...")
    
    # Prepare data and train a model
    df_with_features = demonstrate_feature_engineering()
    processor = DataProcessor()
    
    X, y = processor.prepare_ml_data(df_with_features)
    X_train, X_test, y_train, y_test = processor.split_data(X, y, train_ratio=0.8)
    
    # Train LightGBM model
    print("🚀 Training LightGBM model for predictions...")
    model = LightGBMModel(n_estimators=50, max_depth=4)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    # Show some predictions
    print("\n📊 Sample Predictions:")
    print("Actual vs Predicted (first 10 samples):")
    for i in range(min(10, len(y_test))):
        print(f"  Sample {i+1:2d}: Actual: {y_test[i]:8.2f} | Predicted: {y_pred[i]:8.2f} | Diff: {abs(y_test[i] - y_pred[i]):6.2f}")
    
    # Calculate accuracy metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📈 Prediction Accuracy:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

def main():
    """Main demonstration function"""
    print("🚀 STOCK MARKET PREDICTION SYSTEM - QUICK START DEMO")
    print("=" * 60)
    
    try:
        # Demonstrate feature engineering
        demonstrate_feature_engineering()
        
        # Demonstrate individual ML models
        demonstrate_ml_models()
        
        # Demonstrate benchmarking
        demonstrate_benchmarking()
        
        # Demonstrate predictions
        demonstrate_prediction()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Run: python -m stock_prediction.scripts.benchmark_models --config config/training_config.yaml")
        print("   2. Start API: python -m stock_prediction.api.prediction_api")
        print("   3. Launch Dashboard: python -m stock_prediction.dashboard.app")
        print("   4. Run tests: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
