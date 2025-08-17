"""
Data processing module for stock prediction
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import talib

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the data processor
        
        Args:
            sequence_length (int): Length of sequence for LSTM input
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load stock data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and preprocessed DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for ML models
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        try:
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Need: {required_cols}")
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Open_Close_Ratio'] = df['Open'] / df['Close']
            
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Moving average ratios
            df['MA_5_20_Ratio'] = df['MA_5'] / df['MA_20']
            df['MA_10_50_Ratio'] = df['MA_10'] / df['MA_50']
            
            # Volatility features
            df['Volatility_5'] = df['Close'].rolling(window=5).std()
            df['Volatility_10'] = df['Close'].rolling(window=10).std()
            df['Volatility_20'] = df['Close'].rolling(window=20).std()
            
            # Volume features
            df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            
            # RSI (Relative Strength Index)
            try:
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
            except:
                # Fallback RSI calculation if talib is not available
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(df['Close'].values)
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
                df['MACD_Histogram'] = macd_hist
            except:
                # Fallback MACD calculation
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            df['BB_Upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
            df['BB_Lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
            
            # Time-based features
            df['Day_of_Week'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            
            # Lag features
            df['Close_Lag_1'] = df['Close'].shift(1)
            df['Close_Lag_2'] = df['Close'].shift(2)
            df['Close_Lag_3'] = df['Close'].shift(3)
            df['Close_Lag_5'] = df['Close'].shift(5)
            
            # Target variable (next day's close price)
            df['Target'] = df['Close'].shift(-1)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise
            
    def prepare_ml_data(self, df: pd.DataFrame, target_column: str = 'Target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for traditional ML models
        
        Args:
            df (pd.DataFrame): DataFrame with features
            target_column (str): Column to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for ML models
        """
        try:
            # Select feature columns (exclude target and date-related columns)
            exclude_cols = [target_column, 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            X = df[feature_cols].values
            y = df[target_column].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model (original method)
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column to predict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays for model training
        """
        try:
            # Scale the data
            data = self.scaler.fit_transform(df[[target_column]])
            
            # Create sequences
            X = []
            y = []
            
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:(i + self.sequence_length)])
                y.append(data[i + self.sequence_length])
                
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            train_ratio (float): Ratio of training data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and testing sets
        """
        try:
            train_size = int(len(X) * train_ratio)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data
        
        Args:
            data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        try:
            return self.scaler.inverse_transform(data)
        except Exception as e:
            logger.error(f"Error in inverse transform: {str(e)}")
            raise
            
    def prepare_prediction_data(self, df: pd.DataFrame, target_column: str = 'Close') -> np.ndarray:
        """
        Prepare data for making predictions (LSTM)
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column to predict
            
        Returns:
            np.ndarray: Prepared data for prediction
        """
        try:
            data = self.scaler.fit_transform(df[[target_column]])
            X = []
            X.append(data[-self.sequence_length:])
            return np.array(X)
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise 