"""
Enhanced Dashboard for Stock Market Prediction
Includes LightGBM predictions and model comparison capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.ml_models import (
    LinearRegressionModel, RandomForestModel, XGBoostModel, LightGBMModel
)
from stock_prediction.models.lstm_model import LSTMModel

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

def load_models(config: dict):
    """Load trained models"""
    models = {}
    
    try:
        # Load LightGBM model
        if Path(config.get('lightgbm_model_path', 'models/lightgbm_model.joblib')).exists():
            lightgbm_model = LightGBMModel()
            lightgbm_model.load_model(config.get('lightgbm_model_path', 'models/lightgbm_model.joblib'))
            models['LightGBM'] = lightgbm_model
            st.success("✅ LightGBM model loaded successfully")
        
        # Load XGBoost model
        if Path(config.get('xgb_model_path', 'models/xgboost_model.joblib')).exists():
            xgb_model = XGBoostModel()
            xgb_model.load_model(config.get('xgb_model_path', 'models/xgboost_model.joblib'))
            models['XGBoost'] = xgb_model
            st.success("✅ XGBoost model loaded successfully")
        
        # Load Random Forest model
        if Path(config.get('rf_model_path', 'models/random_forest_model.joblib')).exists():
            rf_model = RandomForestModel()
            rf_model.load_model(config.get('rf_model_path', 'models/random_forest_model.joblib'))
            models['Random Forest'] = rf_model
            st.success("✅ Random Forest model loaded successfully")
        
        # Load Linear Regression model
        if Path(config.get('lr_model_path', 'models/linear_regression_model.joblib')).exists():
            lr_model = LinearRegressionModel()
            lr_model.load_model(config.get('lr_model_path', 'models/linear_regression_model.joblib'))
            models['Linear Regression'] = lr_model
            st.success("✅ Linear Regression model loaded successfully")
        
        # Load LSTM model
        if Path(config.get('model_path', 'models/best_model.h5')).exists():
            lstm_model = LSTMModel(
                sequence_length=config.get('sequence_length', 60),
                n_features=config.get('n_features', 1)
            )
            lstm_model.load_model(config.get('model_path', 'models/best_model.h5'))
            models['LSTM'] = lstm_model
            st.success("✅ LSTM model loaded successfully")
        
        if not models:
            st.warning("⚠️ No trained models found. Please train models first.")
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

def create_sample_data():
    """Create sample stock data for demonstration"""
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock data
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return df

def prepare_features(df: pd.DataFrame, data_processor: DataProcessor):
    """Prepare features for prediction"""
    try:
        # Create features
        df_with_features = data_processor.create_features(df)
        
        # Prepare data for ML models
        X, y = data_processor.prepare_ml_data(df_with_features)
        
        return X, y, df_with_features
        
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None, None, None

def make_predictions(models: dict, X: np.ndarray, data_processor: DataProcessor, df: pd.DataFrame):
    """Make predictions using all available models"""
    predictions = {}
    
    for name, model in models.items():
        try:
            if name == 'LSTM':
                # Prepare data for LSTM
                X_lstm, _ = data_processor.prepare_data(df, target_column='Close')
                # Use last sequence for prediction
                X_lstm_pred = X_lstm[-1:] if len(X_lstm) > 0 else X_lstm
                if len(X_lstm_pred) > 0:
                    pred = model.predict(X_lstm_pred)
                    predictions[name] = pred.flatten()[-1] if len(pred) > 0 else None
                else:
                    predictions[name] = None
            else:
                # Use last sample for prediction
                X_pred = X[-1:] if len(X) > 0 else X
                if len(X_pred) > 0:
                    pred = model.predict(X_pred)
                    predictions[name] = pred[0] if len(pred) > 0 else None
                else:
                    predictions[name] = None
                    
        except Exception as e:
            st.warning(f"Could not make prediction with {name}: {str(e)}")
            predictions[name] = None
    
    return predictions

def plot_stock_data(df: pd.DataFrame, predictions: dict = None):
    """Plot stock data with predictions"""
    fig = go.Figure()
    
    # Plot actual stock data
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Actual',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add predictions if available
    if predictions:
        last_date = df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        for model_name, pred_value in predictions.items():
            if pred_value is not None:
                fig.add_trace(go.Scatter(
                    x=[last_date, next_date],
                    y=[df['Close'].iloc[-1], pred_value],
                    mode='lines+markers',
                    name=f'{model_name} Prediction',
                    line=dict(dash='dash'),
                    marker=dict(size=8)
                ))
    
    fig.update_layout(
        title='Stock Price Chart with Model Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True
    )
    
    return fig

def plot_model_comparison(predictions: dict, actual_price: float):
    """Plot model comparison"""
    if not predictions:
        return None
    
    # Filter out None predictions
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        return None
    
    # Calculate prediction errors
    errors = {k: abs(v - actual_price) for k, v in valid_predictions.items()}
    
    # Create comparison plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Predicted Values', 'Prediction Errors', 'Prediction vs Actual', 'Model Performance'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Predicted Values
    fig.add_trace(
        go.Bar(x=list(valid_predictions.keys()), y=list(valid_predictions.values()),
               name='Predicted', marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. Prediction Errors
    fig.add_trace(
        go.Bar(x=list(errors.keys()), y=list(errors.values()),
               name='Error', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # 3. Prediction vs Actual
    fig.add_trace(
        go.Scatter(x=list(valid_predictions.keys()), y=list(valid_predictions.values()),
                  mode='markers+text', name='Predicted',
                  text=[f'${v:.2f}' for v in valid_predictions.values()],
                  textposition='top center'),
        row=2, col=1
    )
    fig.add_hline(y=actual_price, line_dash="dash", line_color="red",
                  annotation_text=f"Actual: ${actual_price:.2f}")
    
    # 4. Model Performance (lower error is better)
    fig.add_trace(
        go.Bar(x=list(errors.keys()), y=list(errors.values()),
               name='Error (Lower is Better)', marker_color='lightgreen'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def main():
    """Main dashboard function"""
    st.title("🚀 Enhanced Stock Market Prediction Dashboard")
    st.markdown("Compare predictions across multiple models including **LightGBM**, XGBoost, Random Forest, Linear Regression, and LSTM")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Load configuration
    config = load_config()
    
    # Load models
    st.sidebar.subheader("🤖 Model Status")
    models = load_models(config)
    
    if not models:
        st.error("❌ No models available. Please train models first.")
        st.info("💡 Use the training scripts to train models before using the dashboard.")
        return
    
    st.sidebar.success(f"✅ {len(models)} models loaded successfully")
    
    # Data selection
    st.sidebar.subheader("📈 Data Options")
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        df = create_sample_data()
        st.sidebar.info("📊 Using sample data for demonstration")
    else:
        # File upload option
        uploaded_file = st.sidebar.file_uploader(
            "Upload Stock Data (CSV)",
            type=['csv'],
            help="Upload your stock data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                df.reset_index(inplace=True)
                st.sidebar.success("✅ Data uploaded successfully")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("📁 Please upload a CSV file or use sample data")
            return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Stock Data Overview")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # Data statistics
        st.subheader("📈 Data Statistics")
        col1_stats, col2_stats, col3_stats = st.columns(3)
        
        with col1_stats:
            st.metric("Total Records", len(df))
            st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        with col2_stats:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            st.metric("Price Change", f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}%")
        
        with col3_stats:
            st.metric("Highest Price", f"${df['High'].max():.2f}")
            st.metric("Lowest Price", f"${df['Low'].min():.2f}")
    
    with col2:
        st.subheader("🎯 Quick Actions")
        
        if st.button("🔄 Refresh Predictions", type="primary"):
            st.rerun()
        
        if st.button("📊 Show Model Info"):
            st.info("Model Information:")
            for name, model in models.items():
                st.write(f"• **{name}**: {type(model).__name__}")
        
        # Model selection for detailed analysis
        st.subheader("🔍 Model Analysis")
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            list(models.keys()),
            help="Choose a model to analyze its predictions in detail"
        )
    
    # Feature preparation
    st.subheader("⚙️ Feature Engineering")
    
    try:
        data_processor = DataProcessor(sequence_length=config.get('sequence_length', 60))
        X, y, df_with_features = prepare_features(df, data_processor)
        
        if X is not None and y is not None:
            st.success(f"✅ Features prepared successfully: {X.shape[1]} features")
            
            # Show feature information
            with st.expander("🔍 Feature Details"):
                st.write(f"**Total Features:** {X.shape[1]}")
                st.write(f"**Training Samples:** {X.shape[0]}")
                st.write(f"**Target Range:** {y.min():.2f} to {y.max():.2f}")
                
                # Show some feature names
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                st.write("**Feature Names:**", feature_names[:10], "...")
        
        else:
            st.error("❌ Feature preparation failed")
            return
            
    except Exception as e:
        st.error(f"❌ Error in feature preparation: {str(e)}")
        return
    
    # Make predictions
    st.subheader("🔮 Model Predictions")
    
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions = make_predictions(models, X, data_processor, df)
            
            if predictions:
                st.success("✅ Predictions generated successfully!")
                
                # Display predictions
                col1_pred, col2_pred, col3_pred = st.columns(3)
                
                with col1_pred:
                    st.subheader("📊 Prediction Results")
                    for model_name, pred_value in predictions.items():
                        if pred_value is not None:
                            current_price = df['Close'].iloc[-1]
                            change = pred_value - current_price
                            change_pct = (change / current_price) * 100
                            
                            st.metric(
                                model_name,
                                f"${pred_value:.2f}",
                                f"{change:+.2f} ({change_pct:+.2f}%)"
                            )
                        else:
                            st.metric(model_name, "N/A", "Error")
                
                with col2_pred:
                    st.subheader("📈 Price Movement")
                    current_price = df['Close'].iloc[-1]
                    
                    # Find best prediction
                    valid_preds = {k: v for k, v in predictions.items() if v is not None}
                    if valid_preds:
                        best_model = min(valid_preds.keys(), key=lambda x: abs(valid_preds[x] - current_price))
                        best_pred = valid_preds[best_model]
                        
                        st.info(f"🎯 **Best Prediction:** {best_model}")
                        st.info(f"💰 **Predicted Price:** ${best_pred:.2f}")
                        st.info(f"📊 **Current Price:** ${current_price:.2f}")
                
                with col3_pred:
                    st.subheader("📊 Model Status")
                    for model_name, pred_value in predictions.items():
                        if pred_value is not None:
                            st.success(f"✅ {model_name}")
                        else:
                            st.error(f"❌ {model_name}")
                
                # Plot stock data with predictions
                st.subheader("📈 Stock Chart with Predictions")
                fig_stock = plot_stock_data(df, predictions)
                st.plotly_chart(fig_stock, use_container_width=True)
                
                # Model comparison
                st.subheader("🏆 Model Comparison")
                fig_comparison = plot_model_comparison(predictions, df['Close'].iloc[-1])
                if fig_comparison:
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed analysis for selected model
                if selected_model in predictions and predictions[selected_model] is not None:
                    st.subheader(f"🔍 Detailed Analysis: {selected_model}")
                    
                    col1_analysis, col2_analysis = st.columns(2)
                    
                    with col1_analysis:
                        st.write(f"**Model Type:** {type(models[selected_model]).__name__}")
                        st.write(f"**Prediction:** ${predictions[selected_model]:.2f}")
                        st.write(f"**Current Price:** ${df['Close'].iloc[-1]:.2f}")
                        
                        # Calculate confidence metrics
                        if selected_model in ['LightGBM', 'XGBoost', 'Random Forest']:
                            try:
                                # Get feature importance if available
                                if hasattr(models[selected_model].model, 'feature_importances_'):
                                    importance = models[selected_model].model.feature_importances_
                                    top_features = np.argsort(importance)[::-1][:5]
                                    st.write("**Top 5 Features:**")
                                    for i, feat_idx in enumerate(top_features):
                                        st.write(f"  {i+1}. Feature_{feat_idx}: {importance[feat_idx]:.4f}")
                            except:
                                pass
                    
                    with col2_analysis:
                        # Prediction confidence based on model type
                        if selected_model == 'Linear Regression':
                            st.info("📊 **Confidence:** Linear models provide baseline predictions")
                        elif selected_model in ['LightGBM', 'XGBoost']:
                            st.success("🚀 **Confidence:** High - Gradient boosting models typically perform well")
                        elif selected_model == 'Random Forest':
                            st.success("🌳 **Confidence:** High - Ensemble method with good generalization")
                        elif selected_model == 'LSTM':
                            st.info("🧠 **Confidence:** Medium - Neural networks can capture complex patterns")
                        
                        # Trading recommendation
                        current_price = df['Close'].iloc[-1]
                        pred_price = predictions[selected_model]
                        change_pct = ((pred_price - current_price) / current_price) * 100
                        
                        if change_pct > 2:
                            st.success("📈 **Recommendation:** Strong Buy")
                        elif change_pct > 0.5:
                            st.info("📈 **Recommendation:** Buy")
                        elif change_pct < -2:
                            st.error("📉 **Recommendation:** Strong Sell")
                        elif change_pct < -0.5:
                            st.warning("📉 **Recommendation:** Sell")
                        else:
                            st.info("➡️ **Recommendation:** Hold")
                
                # Save predictions
                if st.button("💾 Save Predictions"):
                    predictions_df = pd.DataFrame([
                        {'Model': k, 'Prediction': v, 'Current_Price': df['Close'].iloc[-1]}
                        for k, v in predictions.items() if v is not None
                    ])
                    
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"stock_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Failed to generate predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🚀 Enhanced Stock Market Prediction Dashboard | Powered by LightGBM, XGBoost, Random Forest, Linear Regression & LSTM</p>
        <p>📊 Compare predictions across multiple models for better decision making</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
