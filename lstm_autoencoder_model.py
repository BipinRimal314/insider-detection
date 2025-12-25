"""
LSTM Autoencoder for Insider Threat Detection (Anomaly Detection)
Learns normal sequential patterns and flags deviations.
"""

import os
# Prevent TensorFlow from using Metal GPU which can cause hangs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU to avoid GPU issues

import numpy as np
import pandas as pd
import tensorflow as tf

# Disable GPU for stability on Mac
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import config
import utils

logger = utils.logger

class LSTMAutoencoder:
    """
    LSTM Autoencoder model wrapper for training and evaluation.
    Supports dependency injection for hyperparameter tuning.
    """
    
    def __init__(self, units=None, dropout=None):
        self.units = units if units is not None else config.LSTM_AUTOENCODER['lstm_units']
        self.dropout = dropout if dropout is not None else config.LSTM_AUTOENCODER['dropout_rate']
        self.model = None

    def build_model(self, input_shape):
        """Build the LSTM Autoencoder architecture."""
        inputs = Input(shape=input_shape)
        
        # --- Dynamically build the Encoder ---
        encoded = inputs
        # Iterate through all but the last layer which should not return sequences
        for i, u in enumerate(self.units):
            return_sequences = (i < len(self.units) - 1)
            encoded = LSTM(u, activation='relu', return_sequences=return_sequences)(encoded)
            if i < len(self.units) - 1: # No dropout on the latent space layer
                 encoded = Dropout(self.dropout)(encoded)

        # --- Latent space representation ---
        latent_space = encoded
        
        decoded = RepeatVector(input_shape[0])(latent_space)
        
        # --- Dynamically build the Decoder ---
        # Iterate in reverse
        for u in reversed(self.units):
            decoded = LSTM(u, activation='relu', return_sequences=True)(decoded)
            decoded = Dropout(self.dropout)(decoded)
            
        # --- Output Layer ---
        output = TimeDistributed(Dense(input_shape[1]))(decoded)
        
        self.model = Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LSTM_AUTOENCODER['learning_rate']), 
                      loss='mse')
        
        return self.model

    def train_and_evaluate(self, X_train, X_val, X_test, y_test, scaler=None):
        """
        Train the model and evaluate on test set.
        Returns history, auc, threshold, prediction_df, metrics
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.build_model(input_shape)
        
        if config.LSTM_AUTOENCODER['verbose']:
            self.model.summary(print_fn=logger.info)
            
        callbacks = [
            EarlyStopping(patience=config.LSTM_AUTOENCODER['patience'], restore_best_weights=True),
            # Only save checkpoint if we are using default config (main run), otherwise skip to save disk space
            # or save to temp path
        ]
        
        logger.info(f"Starting training with units={self.units}, dropout={self.dropout}")
        
        history = self.model.fit(
            X_train, X_train, # Autoencoder target is input
            epochs=config.LSTM_AUTOENCODER['epochs'],
            batch_size=config.LSTM_AUTOENCODER['batch_size'],
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        X_test_pred = self.model.predict(X_test)
        test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
        
        anomaly_scores = np.mean(test_mae_loss, axis=1)
        threshold = np.percentile(anomaly_scores, 95)
        predictions = (anomaly_scores > threshold).astype(int)
        
        metrics = utils.calculate_metrics(y_test, predictions, anomaly_scores)
        
        return history, metrics['AUC-ROC'], threshold, predictions, metrics


# Compatibility function for legacy calls
def build_lstm_autoencoder(input_shape):
    model_wrapper = LSTMAutoencoder()
    return model_wrapper.build_model(input_shape)

def load_data():
    """Load sequences and labels"""
    seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
    label_path = config.SEQUENCE_LABELS_FILE
    
    if not seq_path.exists():
        logger.error(f"Sequence data not found at {seq_path}")
        return None, None
        
    try:
        X = np.load(seq_path)
        y = np.load(label_path) if label_path.exists() else np.zeros(len(X))
        
        # --- FAST DEBUGGING FIX ---
        max_samples = getattr(config, 'MAX_SEQUENCE_SAMPLES', None)
        if max_samples and len(X) > max_samples:
            logger.warning(f"DEBUG MODE: Slicing data from {len(X)} to {max_samples} samples.")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        logger.info(f"Loaded sequences: {X.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading sequences: {e}")
        return None, None

def main():
    logger.info(utils.generate_report_header("LSTM AUTOENCODER TRAINING"))
    
    X, y = load_data()
    if X is None:
        raise FileNotFoundError("Training data not found.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - config.TRAIN_RATIO), random_state=config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_SEED
    )
    
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    
    logger.info(f"Training on {len(X_train_normal)} normal sequences")
    
    # Use the new Class
    model_wrapper = LSTMAutoencoder()
    history, auc, threshold, predictions, metrics = model_wrapper.train_and_evaluate(
        X_train_normal, X_val_normal, X_test, y_test
    )
    
    # Save Model
    model_wrapper.model.save(str(config.MODEL_PATHS['lstm_autoencoder']))
    
    # Save Results
    output_path = config.RESULTS_DIR / 'lstm_autoencoder_predictions.csv'
    pd.DataFrame({
        'true_label': y_test,
        'prediction': predictions
    }).to_csv(output_path, index=False)
    
    utils.print_metrics(metrics, "LSTM Autoencoder")

if __name__ == "__main__":
    main()