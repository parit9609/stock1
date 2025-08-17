"""
Script to train the LSTM model
"""

import argparse
import logging
from pathlib import Path
import yaml
import mlflow
import mlflow.keras
from datetime import datetime

from stock_prediction.data.data_processor import DataProcessor
from stock_prediction.models.lstm_model import LSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path: str):
    """
    Train the LSTM model using configuration from YAML file
    
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
        df = data_processor.load_data(config['data_path'])
        X, y = data_processor.prepare_data(df, target_column=config['target_column'])
        X_train, X_test, y_train, y_test = data_processor.split_data(
            X, y, train_ratio=config['train_ratio']
        )
        
        # Initialize model
        model = LSTMModel(
            sequence_length=config['sequence_length'],
            n_features=config['n_features']
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'sequence_length': config['sequence_length'],
                'n_features': config['n_features'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'train_ratio': config['train_ratio']
            })
            
            # Train model
            history = model.train(
                X_train, y_train,
                X_test, y_test,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                model_path=config['model_path']
            )
            
            # Log metrics
            for epoch in range(len(history['loss'])):
                mlflow.log_metrics({
                    'loss': history['loss'][epoch],
                    'val_loss': history['val_loss'][epoch]
                }, step=epoch)
            
            # Evaluate model
            test_loss = model.evaluate(X_test, y_test)
            mlflow.log_metric('test_loss', test_loss)
            
            # Save model
            mlflow.keras.log_model(model.model, "model")
            
            logger.info(f"Model training completed. Test loss: {test_loss}")
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for stock prediction')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    train_model(args.config)

if __name__ == "__main__":
    main() 