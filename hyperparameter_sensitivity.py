"""
Hyperparameter Sensitivity Analysis

Systematically evaluates the impact of key hyperparameters on model performance.
This helps justify the choice of "magic numbers" used in the final configuration.

- LSTM Architecture: Varies units and dropout rates.
- Isolation Forest: Varies the number of estimators.

Usage:
    python hyperparameter_sensitivity.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Environment setup for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import config
import utils
from isolation_forest_model import train_isolation_forest
from lstm_autoencoder_model import build_lstm_autoencoder

logger = utils.logger

class HyperparameterStudy:
    """
    Performs sensitivity analysis for model hyperparameters.
    """
    def __init__(self):
        self.results_dir = config.RESULTS_DIR / 'sensitivity'
        self.results_dir.mkdir(exist_ok=True)
        
        self.daily_features = None
        self.sequences = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        """Load preprocessed data needed for the studies."""
        logger.info("Loading data for sensitivity analysis...")
        try:
            # Data for Isolation Forest
            daily_path = config.DAILY_FEATURES_FILE
            if daily_path.exists():
                self.daily_features = pd.read_parquet(daily_path)
                logger.info(f"Loaded daily features: {self.daily_features.shape}")

            # Data for LSTM
            seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
            labels_path = config.SEQUENCE_LABELS_FILE
            if seq_path.exists() and labels_path.exists():
                # Limit samples for speed if configured
                max_samples = getattr(config, 'MAX_SEQUENCE_SAMPLES', 50000) # Cap for speed
                
                X = np.load(seq_path)
                y = np.load(labels_path)

                if max_samples and len(X) > max_samples:
                    logger.warning(f"Slicing sequence data to {max_samples} for speed.")
                    indices = np.random.choice(len(X), max_samples, replace=False)
                    self.sequences = X[indices]
                    self.labels = y[indices]
                else:
                    self.sequences = X
                    self.labels = y
                    
                logger.info(f"Loaded sequences: {self.sequences.shape}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            sys.exit(1)

    def run_lstm_sensitivity(self):
        """
        Evaluates sensitivity of the LSTM Autoencoder to its hyperparameters.
        """
        logger.info(utils.generate_report_header("LSTM HYPERPARAMETER SENSITIVITY"))

        if self.sequences is None:
            logger.error("Sequence data not loaded. Cannot run LSTM sensitivity study.")
            return

        # Define hyperparameter ranges to test
        lstm_units_options = [[16], [16, 8], [32, 16], [64, 32]]
        dropout_options = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        results = []

        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(self.sequences, self.labels, test_size=0.3, random_state=config.RANDOM_SEED)
        X_train_normal = X_train[y_train == 0]

        if len(X_train_normal) == 0 or len(X_test) == 0:
            logger.error("Not enough data to perform LSTM sensitivity analysis.")
            return
            
        input_shape = (self.sequences.shape[1], self.sequences.shape[2])

        for units in lstm_units_options:
            for dropout in dropout_options:
                logger.info(f"Testing LSTM with units: {units}, dropout: {dropout}")
                
                # Temporarily override config
                original_lstm_config = config.LSTM_AUTOENCODER.copy()
                config.LSTM_AUTOENCODER['lstm_units'] = units
                config.LSTM_AUTOENCODER['dropout_rate'] = dropout

                try:
                    # Build and train model
                    model = build_lstm_autoencoder(input_shape)
                    model.fit(
                        X_train_normal, X_train_normal,
                        epochs=10, # Reduced epochs for speed
                        batch_size=config.LSTM_AUTOENCODER['batch_size'],
                        validation_split=0.1,
                        callbacks=[EarlyStopping(patience=3)],
                        verbose=0
                    )
                    
                    # Evaluate
                    predictions = model.predict(X_test)
                    mse = np.mean(np.square(X_test - predictions), axis=(1, 2))
                    
                    # Check if there are positive samples in the test set
                    if np.sum(y_test) > 0:
                        auc = roc_auc_score(y_test, mse)
                    else:
                        auc = 0.5 # Cannot calculate AUC without positive samples
                        logger.warning("No positive samples in test set for this run.")

                    results.append({
                        'lstm_units': str(units),
                        'dropout_rate': dropout,
                        'auc_roc': auc
                    })
                    logger.info(f"  --> AUC: {auc:.4f}")

                except Exception as e:
                    logger.error(f"  --> FAILED: {e}")
                
                finally:
                    # Restore config
                    config.LSTM_AUTOENCODER = original_lstm_config

        if not results:
            logger.error("LSTM sensitivity analysis produced no results.")
            return

        # Save and plot results
        results_df = pd.DataFrame(results)
        save_path = self.results_dir / 'lstm_hyperparameter_sensitivity.csv'
        results_df.to_csv(save_path, index=False)
        logger.info(f"LSTM sensitivity results saved to {save_path}")

        self._plot_heatmap(results_df, 'dropout_rate', 'lstm_units', 'auc_roc', 'LSTM Hyperparameter Sensitivity', 'lstm_sensitivity_heatmap.png')
        return results_df

    def run_isoforest_sensitivity(self):
        """
        Evaluates sensitivity of Isolation Forest to its hyperparameters.
        """
        logger.info(utils.generate_report_header("ISOLATION FOREST HYPERPARAMETER SENSITIVITY"))
        
        if self.daily_features is None:
            logger.error("Daily features not loaded. Cannot run Isolation Forest sensitivity study.")
            return

        # Define hyperparameter ranges
        n_estimators_options = [50, 100, 200, 300, 500]
        
        results = []

        # Prepare data
        exclude_cols = ['user', 'day', 'is_anomaly', 'is_insider']
        feature_cols = [c for c in self.daily_features.columns if c not in exclude_cols]
        X = self.daily_features[feature_cols].fillna(0).values
        y = self.daily_features['is_anomaly'].values if 'is_anomaly' in self.daily_features.columns else np.zeros(len(self.daily_features))

        if np.sum(y) == 0:
            logger.error("No positive labels in data. Cannot run Isolation Forest sensitivity analysis.")
            return

        for n_estimators in n_estimators_options:
            logger.info(f"Testing Isolation Forest with n_estimators: {n_estimators}")
            
            try:
                # Correctly call the refactored function with the feature matrix and hyperparameter
                model = train_isolation_forest(X, n_estimators=n_estimators)
                scores = -model.decision_function(X)
                auc = roc_auc_score(y, scores)

                results.append({
                    'n_estimators': n_estimators,
                    'auc_roc': auc
                })
                logger.info(f"  --> AUC: {auc:.4f}")

            except Exception as e:
                 logger.error(f"  --> FAILED: {e}")

        if not results:
            logger.error("Isolation Forest sensitivity analysis produced no results.")
            return
            
        # Save and plot results
        results_df = pd.DataFrame(results)
        save_path = self.results_dir / 'isoforest_hyperparameter_sensitivity.csv'
        results_df.to_csv(save_path, index=False)
        logger.info(f"Isolation Forest sensitivity results saved to {save_path}")

        self._plot_line(results_df, 'n_estimators', 'auc_roc', 'Isolation Forest Sensitivity (n_estimators)', 'isoforest_sensitivity_line.png')
        return results_df

    def _plot_heatmap(self, df, x_col, y_col, val_col, title, filename):
        """Generates and saves a heatmap."""
        try:
            pivot_df = df.pivot(index=y_col, columns=x_col, values=val_col)
            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis")
            plt.title(title, fontsize=16)
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            save_path = self.results_dir / filename
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")

    def _plot_line(self, df, x_col, y_col, title, filename):
        """Generates and saves a line plot."""
        try:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x=x_col, y=y_col, marker='o')
            plt.title(title, fontsize=16)
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            save_path = self.results_dir / filename
            plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
            plt.close()
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to generate line plot: {e}")

    def run_all(self):
        """Run all sensitivity analysis studies."""
        self.run_lstm_sensitivity()
        self.run_isoforest_sensitivity()
        logger.info("All hyperparameter sensitivity studies complete.")


if __name__ == '__main__':
    study = HyperparameterStudy()
    study.run_all()