# Stock Market Prediction with LightGBM and Model Benchmarking

A comprehensive stock market prediction system that integrates LightGBM (Light Gradient Boosting Machine) alongside Linear Regression, Random Forest, XGBoost, and LSTM models. The project provides a robust benchmarking framework to compare model performance and identify the best predictive model for stock price forecasting.

## 🚀 Features

### **Machine Learning Models**
- **LightGBM**: High-performance gradient boosting framework with optimized parameters
- **Linear Regression**: Baseline linear model for comparison
- **Random Forest**: Ensemble method with configurable parameters
- **XGBoost**: Extreme gradient boosting with advanced features
- **LSTM**: Deep learning model for sequential data analysis

### **Advanced Features**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Feature Engineering**: Automatic creation of 30+ technical features
- **Model Benchmarking**: Comprehensive comparison of all models
- **Performance Metrics**: MSE, MAE, R², RMSE evaluation
- **Visualization**: Automated generation of performance charts and heatmaps
- **MLflow Integration**: Experiment tracking and model versioning
- **RESTful API**: Production-ready prediction endpoints
- **Batch Processing**: Support for bulk predictions

## 📊 Model Performance Comparison

The system automatically benchmarks all models and provides:
- **Performance Rankings**: R², RMSE, and MAE comparisons
- **Training vs Test Metrics**: Overfitting detection
- **Visual Analysis**: Interactive charts and heatmaps
- **Best Model Selection**: Automatic identification of top performer

## 🏗️ Architecture

```
Stock-Market-Prediction/
├── stock_prediction/
│   ├── models/
│   │   ├── lstm_model.py          # LSTM neural network
│   │   └── ml_models.py          # Traditional ML models (LightGBM, XGBoost, etc.)
│   ├── data/
│   │   └── data_processor.py     # Enhanced data processing with technical indicators
│   ├── api/
│   │   ├── main.py               # Original API
│   │   └── prediction_api.py     # Enhanced prediction API
│   ├── dashboard/
│   │   └── app.py                # Streamlit dashboard
│   └── scripts/
│       ├── train_model.py        # LSTM training script
│       └── benchmark_models.py   # Model benchmarking script
├── config/
│   └── training_config.yaml      # Configuration for all models
├── tests/                        # Comprehensive test suite
└── requirements.txt              # Enhanced dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Stock-Market-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- **LightGBM**: Fast gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **scikit-learn**: Traditional ML algorithms
- **TensorFlow/Keras**: Deep learning (LSTM)
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **FastAPI**: REST API framework
- **MLflow**: Experiment tracking

## 📈 Usage

### 1. Model Benchmarking

Run comprehensive benchmarking of all models:

```bash
python -m stock_prediction.scripts.benchmark_models --config config/training_config.yaml
```

This will:
- Train all models (Linear Regression, Random Forest, XGBoost, LightGBM, LSTM)
- Generate performance comparisons
- Create visualization plots
- Save results to CSV
- Log experiments in MLflow

### 2. Individual Model Training

Train specific models:

```bash
# Train LSTM only
python -m stock_prediction.scripts.train_model --config config/training_config.yaml

# Train ML models individually
python -c "
from stock_prediction.models.ml_models import LightGBMModel
from stock_prediction.data.data_processor import DataProcessor

# Load and prepare data
processor = DataProcessor()
df = processor.load_data('data/stock_data.csv')
df_features = processor.create_features(df)
X, y = processor.prepare_ml_data(df_features)

# Train LightGBM
model = LightGBMModel(n_estimators=100, learning_rate=0.1)
train_metrics = model.fit(X, y)
model.save_model('models/lightgbm_model.joblib')
"
```

### 3. API Usage

Start the prediction API:

```bash
python -m stock_prediction.api.prediction_api
```

Make predictions:

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "model_type": "lightgbm",
    "data": {
        "Open": 100.0,
        "High": 105.0,
        "Low": 98.0,
        "Close": 102.0,
        "Volume": 1000000
    }
})

# Batch prediction
with open('stock_data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/predict_batch",
        params={"model_type": "lightgbm"},
        files={"file": f}
    )
```

### 4. Dashboard

Launch the interactive dashboard:

```bash
python -m stock_prediction.dashboard.app
```

## ⚙️ Configuration

Edit `config/training_config.yaml` to customize model parameters:

```yaml
# LightGBM configuration
lgb_n_estimators: 100
lgb_max_depth: 6
lgb_learning_rate: 0.1
lgb_num_leaves: 31
lgb_min_child_samples: 20

# XGBoost configuration
xgb_n_estimators: 100
xgb_max_depth: 6
xgb_learning_rate: 0.1

# Random Forest configuration
rf_n_estimators: 100
rf_max_depth: null  # unlimited depth
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_ml_models.py
pytest tests/test_data_processor.py
pytest tests/test_lstm_model.py

# Run with coverage
pytest --cov=stock_prediction tests/
```

## 📊 Technical Indicators

The system automatically generates 30+ technical features:

- **Price-based**: Price changes, ratios, volatility
- **Moving Averages**: 5, 10, 20, 50-day averages
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Volume Analysis**: Volume ratios and moving averages
- **Time Features**: Day of week, month, quarter
- **Lag Features**: Previous price values

## 🔍 Model Selection Guide

### **When to Use Each Model:**

- **LightGBM**: Best for most cases - fast, accurate, handles categorical data
- **XGBoost**: High accuracy, good for complex patterns
- **Random Forest**: Robust, handles outliers well
- **Linear Regression**: Baseline, interpretable, fast
- **LSTM**: Best for sequential patterns, requires more data

### **Performance Considerations:**
- **Speed**: Linear Regression > Random Forest > LightGBM > XGBoost > LSTM
- **Accuracy**: LightGBM ≈ XGBoost > Random Forest > LSTM > Linear Regression
- **Memory**: LSTM > XGBoost > LightGBM > Random Forest > Linear Regression

## 📈 Results Interpretation

### **Key Metrics:**
- **R² Score**: Higher is better (0.0 to 1.0)
- **RMSE**: Lower is better (root mean squared error)
- **MAE**: Lower is better (mean absolute error)

### **Overfitting Detection:**
- Compare Training vs Test R² scores
- Large gaps indicate overfitting
- Use regularization parameters to control

## 🚀 Production Deployment

### **Docker Support:**
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "stock_prediction.api.prediction_api"]
```

### **Environment Variables:**
```bash
export MLFLOW_TRACKING_URI=your_mlflow_server
export MODEL_PATH=/path/to/models
export DATA_PATH=/path/to/data
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LightGBM Team**: For the excellent gradient boosting framework
- **XGBoost Team**: For the powerful XGBoost implementation
- **scikit-learn**: For the comprehensive ML toolkit
- **TensorFlow/Keras**: For the deep learning capabilities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

---

**Note**: This enhanced version provides a production-ready, benchmarked stock prediction system that can automatically select the best performing model for your specific use case.
