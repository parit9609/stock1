"""
Data Preparation Script for Stock Market Prediction
Creates lag features, rolling averages, and volatility measures from stock closing prices
"""

import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stock_prediction.data.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_stock_data(config_path: str):
    """
    Prepare stock data with technical indicators and features
    
    Args:
        config_path (str): Path to configuration file
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Initialize data processor
        data_processor = DataProcessor(
            sequence_length=config['sequence_length']
        )
        
        # Load raw data
        logger.info(f"Loading data from {config['data_path']}")
        df = data_processor.load_data(config['data_path'])
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Create features
        logger.info("Creating technical indicators and features...")
        df_with_features = data_processor.create_features(df)
        
        logger.info(f"Enhanced data shape: {df_with_features.shape}")
        
        # Show feature information
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        new_features = [col for col in df_with_features.columns if col not in original_cols]
        
        logger.info(f"Created {len(new_features)} new features:")
        for i, feature in enumerate(new_features[:20]):  # Show first 20
            logger.info(f"  {i+1:2d}. {feature}")
        
        if len(new_features) > 20:
            logger.info(f"  ... and {len(new_features) - 20} more features")
        
        # Save processed dataset
        output_path = Path(config.get('processed_data_path', 'data/processed_stock_data.csv'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_with_features.to_csv(output_path)
        logger.info(f"Processed dataset saved to: {output_path}")
        
        # Save feature summary
        feature_summary = {
            'total_features': len(df_with_features.columns),
            'original_features': len(original_cols),
            'new_features': len(new_features),
            'data_shape': df_with_features.shape,
            'date_range': {
                'start': str(df_with_features.index.min()),
                'end': str(df_with_features.index.max())
            },
            'feature_list': list(df_with_features.columns)
        }
        
        summary_path = output_path.parent / 'feature_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(feature_summary, f, default_flow_style=False)
        
        logger.info(f"Feature summary saved to: {summary_path}")
        
        # Display sample of processed data
        logger.info("\nSample of processed data:")
        logger.info(f"\n{df_with_features.head()}")
        
        # Check for missing values
        missing_values = df_with_features.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning("Missing values detected:")
            logger.warning(missing_values[missing_values > 0])
        else:
            logger.info("No missing values in processed dataset")
        
        return df_with_features, output_path
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Prepare stock data with technical indicators')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    prepare_stock_data(args.config)

if __name__ == "__main__":
    main()
