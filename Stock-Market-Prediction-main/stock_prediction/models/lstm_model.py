"""
LSTM model for stock prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, sequence_length: int, n_features: int = 1):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build LSTM model architecture
        
        Returns:
            Sequential: Compiled Keras model
        """
        try:
            model = Sequential([
                LSTM(units=50, return_sequences=True, 
                     input_shape=(self.sequence_length, self.n_features)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='mean_squared_error')
            
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              model_path: str = 'best_model.h5') -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            model_path (str): Path to save best model
            
        Returns:
            Dict[str, Any]: Training history
        """
        try:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            return history.history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def load_weights(self, model_path: str) -> None:
        """
        Load model weights from file
        
        Args:
            model_path (str): Path to model weights file
        """
        try:
            self.model.load_weights(model_path)
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate model performance
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            float: Test loss
        """
        try:
            return self.model.evaluate(X_test, y_test, verbose=0)
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise 