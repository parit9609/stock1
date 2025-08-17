# LightGBM Integration Guide

## Overview

This guide provides comprehensive instructions for using LightGBM (Light Gradient Boosting Machine) in the Stock Market Prediction project. LightGBM is a high-performance gradient boosting framework that offers excellent performance for regression tasks like stock price prediction.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python stock_prediction/scripts/prepare_data.py --config config/training_config.yaml
```

### 3. Train LightGBM Model

```bash
python stock_prediction/scripts/train_lightgbm.py --config config/training_config.yaml
```

### 4. Run Complete Benchmark

```bash
python stock_prediction/scripts/evaluate_models.py --config config/training_config.yaml
```

## 📊 Model Architecture

### LightGBM Implementation

The LightGBM model is implemented in `stock_prediction/models/ml_models.py` with the following features:

- **Base Class**: Inherits from `BaseMLModel` for consistent interface
- **Hyperparameters**: Configurable via YAML configuration
- **Feature Importance**: Automatic extraction and visualization
- **Model Persistence**: Save/load functionality using joblib
- **Performance Metrics**: Comprehensive evaluation (RMSE, MAE, R²)

### Key Parameters

```yaml
lightgbm:
  n_estimators: 200          # Number of boosting rounds
  max_depth: 8               # Maximum tree depth
  learning_rate: 0.05        # Learning rate (shrinkage)
  random_state: 42           # Random seed for reproducibility
  num_leaves: 31             # Maximum number of leaves
  min_child_samples: 20      # Minimum samples per leaf
  subsample: 0.8             # Subsample ratio for rows
  colsample_bytree: 0.8      # Subsample ratio for columns
  reg_alpha: 0.1             # L1 regularization
  reg_lambda: 0.1            # L2 regularization
```

## 🎯 Training Process

### 1. Data Preparation

The system automatically creates comprehensive features:

- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Moving Averages**: Multiple timeframes (5, 10, 20, 50 days)
- **Volatility Measures**: Rolling standard deviations
- **Lag Features**: Previous price movements
- **Volume Features**: Volume-based indicators
- **Time Features**: Day of week, month, quarter

### 2. Feature Engineering

```python
from stock_prediction.data.data_processor import DataProcessor

# Initialize data processor
data_processor = DataProcessor(sequence_length=60)

# Create features
df_with_features = data_processor.create_features(df)

# Prepare for ML models
X, y = data_processor.prepare_ml_data(df_with_features)
```

### 3. Model Training

```python
from stock_prediction.models.ml_models import LightGBMModel

# Initialize model
model = LightGBMModel(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)

# Train model
train_metrics = model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📈 Evaluation & Benchmarking

### Performance Metrics

The system evaluates models using:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Square Error

### Model Comparison

```bash
# Run comprehensive evaluation
python stock_prediction/scripts/evaluate_models.py --config config/training_config.yaml
```

This will:
1. Train all models (Linear Regression, Random Forest, XGBoost, LightGBM, LSTM)
2. Evaluate performance on test set
3. Generate comparison visualizations
4. Save results and models

### Expected Performance

Based on typical stock prediction tasks:

| Model | Expected R² | Expected RMSE | Training Time |
|-------|-------------|---------------|---------------|
| Linear Regression | 0.3-0.5 | High | < 1s |
| Random Forest | 0.6-0.8 | Medium | 5-15s |
| XGBoost | 0.7-0.85 | Low | 10-30s |
| **LightGBM** | **0.75-0.9** | **Low** | **8-25s** |
| LSTM | 0.6-0.8 | Medium | 2-10min |

## 🎨 Visualization & Analysis

### Training Plots

The LightGBM training script generates:

1. **Predicted vs Actual**: Scatter plot with perfect prediction line
2. **Residuals Plot**: Error distribution analysis
3. **Feature Importance**: Top 20 most important features
4. **Time Series**: Actual vs predicted comparison

### Feature Importance Analysis

```python
# Extract feature importance
if hasattr(model.model, 'feature_importances_'):
    importance = model.model.feature_importances_
    
    # Get top features
    top_features = np.argsort(importance)[::-1][:10]
    
    for i, feat_idx in enumerate(top_features):
        print(f"{i+1}. Feature_{feat_idx}: {importance[feat_idx]:.4f}")
```

## 🔧 Configuration Management

### YAML Configuration

All LightGBM parameters are configurable via `config/training_config.yaml`:

```yaml
# LightGBM specific configuration
lightgbm:
  n_estimators: 200
  max_depth: 8
  learning_rate: 0.05
  random_state: 42
  num_leaves: 31
  min_child_samples: 20
  
# General configuration
train_ratio: 0.8
random_state: 42
experiment_name: "stock_prediction_lightgbm_benchmark"
```

### Environment Variables

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="mlruns"

# Set experiment name
export MLFLOW_EXPERIMENT_NAME="stock_prediction_lightgbm"
```

## 🚀 Production Deployment

### Model Serving

```python
# Load trained model
model = LightGBM()
model.load_model("models/lightgbm_model.joblib")

# Make predictions
predictions = model.predict(new_data)
```

### API Integration

```python
from fastapi import FastAPI
from stock_prediction.api.prediction_api import app

# Start API server
# uvicorn stock_prediction.api.prediction_api:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "stock_prediction.api.prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📊 Dashboard Usage

### Enhanced Dashboard

The enhanced dashboard (`stock_prediction/dashboard/enhanced_app.py`) provides:

1. **Model Comparison**: Side-by-side predictions from all models
2. **Interactive Charts**: Plotly-based visualizations
3. **Feature Analysis**: Detailed feature importance exploration
4. **Trading Recommendations**: Buy/sell/hold suggestions

### Running Dashboard

```bash
# Install Streamlit
pip install streamlit

# Run dashboard
streamlit run stock_prediction/dashboard/enhanced_app.py
```

## 🧪 Testing & Validation

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run LightGBM specific tests
python -m pytest tests/test_lightgbm_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=stock_prediction --cov-report=html
```

### Test Coverage

The test suite covers:

- ✅ Model initialization and parameter validation
- ✅ Training and prediction functionality
- ✅ Model saving and loading
- ✅ Feature importance extraction
- ✅ Input validation and error handling
- ✅ Performance consistency
- ✅ Production readiness (large datasets, memory efficiency)

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size or number of estimators
   lightgbm:
     n_estimators: 100  # Reduce from 200
     max_depth: 6       # Reduce from 8
   ```

2. **Training Time Too Long**
   ```bash
   # Increase learning rate, reduce estimators
   lightgbm:
     learning_rate: 0.1  # Increase from 0.05
     n_estimators: 100  # Reduce from 200
   ```

3. **Overfitting**
   ```bash
   # Add regularization
   lightgbm:
     reg_alpha: 0.2      # Increase L1 regularization
     reg_lambda: 0.2     # Increase L2 regularization
     min_child_samples: 50  # Increase minimum samples
   ```

### Performance Optimization

1. **Feature Selection**: Remove low-importance features
2. **Hyperparameter Tuning**: Use grid search or Bayesian optimization
3. **Early Stopping**: Implement early stopping to prevent overfitting
4. **Cross-Validation**: Use k-fold cross-validation for robust evaluation

## 📚 Advanced Usage

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Perform grid search
grid_search = GridSearchCV(
    LightGBMModel(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Ensemble Methods

```python
# Combine multiple LightGBM models
models = []
for seed in [42, 123, 456]:
    model = LightGBMModel(random_state=seed)
    model.fit(X_train, y_train)
    models.append(model)

# Average predictions
predictions = np.mean([model.predict(X_test) for model in models], axis=0)
```

### Custom Loss Functions

```python
# Implement custom objective function
def custom_objective(y_true, y_pred):
    # Custom loss calculation
    return gradient, hessian

# Use in model
model = LightGBMModel(objective=custom_objective)
```

## 📈 Monitoring & Maintenance

### Model Performance Tracking

```python
import mlflow

# Log metrics
mlflow.log_metric("test_r2", r2_score)
mlflow.log_metric("test_rmse", rmse_score)

# Log model
mlflow.sklearn.log_model(model, "lightgbm_model")
```

### Model Retraining

```python
# Schedule periodic retraining
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()
scheduler.add_job(retrain_model, 'cron', day_of_week='mon', hour=2)
scheduler.start()
```

### A/B Testing

```python
# Compare model versions
def compare_models(model_a, model_b, X_test, y_test):
    pred_a = model_a.predict(X_test)
    pred_b = model_b.predict(X_test)
    
    score_a = r2_score(y_test, pred_a)
    score_b = r2_score(y_test, pred_b)
    
    return score_a, score_b
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📞 Support

### Getting Help

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check this guide and project README
- **Examples**: Review example scripts and notebooks

### Community Resources

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Gradient Boosting Guide](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
- [Stock Prediction Best Practices](https://github.com/your-repo/wiki)

---

## 🎯 Next Steps

1. **Train your first LightGBM model** using the quick start guide
2. **Experiment with hyperparameters** to optimize performance
3. **Integrate with your data pipeline** for automated training
4. **Deploy to production** using the provided API and dashboard
5. **Monitor performance** and retrain as needed

Happy modeling! 🚀📈
