"""
Data Preprocessing Module
Handles data loading, cleaning, and feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(filepath, scaler_path=None):
    """
    Load CSV data and apply preprocessing
    
    Args:
        filepath: Path to CSV file
        scaler_path: Path to saved scaler (optional)
    
    Returns:
        Preprocessed data, feature names, scaler
    """
    try:
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"✓ Data loaded: {df.shape}")
        
        # Remove missing values
        df = df.dropna()
        logger.info(f"✓ Missing values removed: {df.shape}")
        
        # Separate features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values if df.shape[1] > 1 else None
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("✓ Data standardized")
        
        # Save scaler
        with open(scaler_path or '../models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("✓ Scaler saved")
        
        return X_scaled, y, df.columns[:-1], scaler
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


def load_scaler(scaler_path):
    """Load a previously saved scaler"""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✓ Scaler loaded")
    return scaler


def transform_data(X, scaler):
    """Apply saved scaler to new data"""
    return scaler.transform(X)


if __name__ == "__main__":
    # Example usage
    X, y, features, scaler = load_and_preprocess_data("../data/iris_train.csv")
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Features: {list(features)}")
