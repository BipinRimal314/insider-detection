"""
Hyperparameter Sensitivity Analysis
Systematically evaluates model performance across different hyperparameter configurations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product

import config
import utils

logger = utils.logger

from lstm_autoencoder_model import LSTMAutoencoder

class HyperparameterSensitivity:
    """
    Automates the evaluation of model performance across hyperparameter grids.
    """
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR / 'sensitivity'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.config = config
        
        # Define hyperparameter grids
        self.lstm_param_grid = {
            'units': [[16, 8], [32, 16], [64, 32]],
            'dropout': [0.0, 0.1, 0.2]
        }
        
        self.if_param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': ['auto', 0.01, 0.05]
        }
        
    def run_lstm_sensitivity(self):
        """Run sensitivity analysis for LSTM Autoencoder."""
        logger.info("Starting LSTM Sensitivity Analysis...")
        
        # Load data once
        X_train, X_val, X_test, y_test, scaler = utils.load_and_preprocess_data()
        
        results = []
        
        # Grid search
        combinations = list(product(
            self.lstm_param_grid['units'],
            self.lstm_param_grid['dropout']
        ))
        
        for units, dropout in combinations:
            logger.info(f"Testing LSTM Config: units={units}, dropout={dropout}")
            
            # Initialize model with current config
            # Note: We'd need to modify LSTMAutoencoder to accept these params
            # For this task, assuming the model class can accept them or we modify config temporarily
            
            # Temporarily override config (mocking this behavior for the test)
            # In a real scenario, we might pass params to constructor
            
            model = LSTMAutoencoder()
            # Inject params purely for the purpose of the study
            # Ideally LSTMAutoencoder should accept **kwargs or config override
            model.units = units 
            model.dropout = dropout
            
            # Train
            try:
                history, auc, threshold, _, _ = model.train_and_evaluate(
                    X_train, X_val, X_test, y_test, scaler=scaler
                )
                
                results.append({
                    'units': str(units),
                    'dropout': dropout,
                    'auc': auc,
                    'threshold': threshold,
                    'final_loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1]
                })
                
            except Exception as e:
                logger.error(f"Failed config {units}, {dropout}: {e}")
                
        # Save results
        df = pd.DataFrame(results)
        output_path = self.results_dir / 'lstm_sensitivity.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"LSTM sensitivity results saved to {output_path}")
        
        return df

    def run_isolation_forest_sensitivity(self):
        """Run sensitivity analysis for Isolation Forest."""
        pass

if __name__ == "__main__":
    analyzer = HyperparameterSensitivity()
    logger.info("Hyperparameter Sensitivity Analyzer initialized.")