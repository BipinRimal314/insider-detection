"""
Transformer-Based Sequence Model for Insider Threat Detection

Uses self-attention mechanism to capture long-range dependencies in user behavior.
Compared to LSTM, Transformers can:
1. Process sequences in parallel (faster training)
2. Capture long-range dependencies better
3. Provide attention weights for interpretability

Usage:
    python transformer_model.py

Architecture:
    Input (seq_len, features) -> Positional Encoding -> 
    Multi-Head Attention -> Feed Forward -> 
    Global Pooling -> Dense -> Reconstruction
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# TensorFlow configuration for Apple Silicon
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Check for Apple Silicon Metal support
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")

# Enable memory growth to avoid OOM
for device in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

from tensorflow import keras
from tensorflow.keras import layers

import config
import utils

logger = utils.logger


class TransformerEncoder(layers.Layer):
    """
    Transformer Encoder block with multi-head self-attention.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """
    Adds positional information to input embeddings.
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_encoding = self._get_positional_encoding(max_len, embed_dim)
    
    def _get_positional_encoding(self, max_len, embed_dim):
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(embed_dim)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / embed_dim)
        
        # Apply sin to even indices, cos to odd
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        return tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


class TransformerAutoencoder:
    """
    Transformer-based autoencoder for anomaly detection.
    """
    
    def __init__(self, seq_length, n_features, embed_dim=32, num_heads=4, 
                 ff_dim=64, num_transformer_blocks=2, dropout=0.1):
        self.seq_length = seq_length
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_transformer_blocks
        self.dropout = dropout
        self.model = None
        self.threshold = None
        
    def build_model(self):
        """Build Transformer Autoencoder model."""
        inputs = layers.Input(shape=(self.seq_length, self.n_features))
        
        # Project to embedding dimension
        x = layers.Dense(self.embed_dim)(inputs)
        
        # Add positional encoding
        x = PositionalEncoding(self.seq_length, self.embed_dim)(x)
        
        # Transformer encoder blocks
        for _ in range(self.num_blocks):
            x = TransformerEncoder(self.embed_dim, self.num_heads, 
                                   self.ff_dim, self.dropout)(x)
        
        # Bottleneck - compress to latent representation
        latent = layers.GlobalAveragePooling1D()(x)
        latent = layers.Dense(self.embed_dim // 2, activation='relu')(latent)
        
        # Decoder - expand back to sequence
        x = layers.RepeatVector(self.seq_length)(latent)
        
        # Transformer decoder blocks
        for _ in range(self.num_blocks):
            x = TransformerEncoder(self.embed_dim // 2, self.num_heads // 2,
                                   self.ff_dim // 2, self.dropout)(x)
        
        # Output projection
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(x)
        
        self.model = keras.Model(inputs, outputs)
        # Add run_eagerly=True here
        self.model.compile(optimizer='adam', loss='mse', run_eagerly=True) 
        
        return self.model
    
    def train(self, X_train, X_val=None, epochs=20, batch_size=256):
        """Train the model on normal sequences."""
        logger.info(f"Training Transformer Autoencoder for {epochs} epochs...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold on training data
        train_pred = self.model.predict(X_train, verbose=0)
        train_mse = np.mean(np.square(X_train - train_pred), axis=(1, 2))
        self.threshold = np.percentile(train_mse, 95)
        
        logger.info(f"Training complete. Threshold: {self.threshold:.4f}")
        return history
    
    def predict(self, X):
        """Get reconstruction errors."""
        reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return mse
    
    def detect_anomalies(self, X):
        """Detect anomalies based on threshold."""
        errors = self.predict(X)
        return (errors > self.threshold).astype(int), errors
    
    def get_attention_weights(self, X):
        """
        Extract attention weights for interpretability.
        Returns attention patterns showing which timesteps the model focuses on.
        """
        # Build attention extraction model
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, TransformerEncoder):
                attention_layer = layer.att
                break
        
        if attention_layer is None:
            return None
            
        # Create intermediate model to extract attention
        # Note: This is a simplified approach; full attention extraction would require
        # modifying the model architecture to output attention weights
        return None  # TODO: Implement full attention extraction
    
    def save(self, path):
        """Save model."""
        self.model.save(path)
        # Save threshold
        np.save(str(path).replace('.keras', '_threshold.npy'), self.threshold)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model."""
        self.model = keras.models.load_model(path, custom_objects={
            'TransformerEncoder': TransformerEncoder,
            'PositionalEncoding': PositionalEncoding
        })
        threshold_path = str(path).replace('.keras', '_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)
        logger.info(f"Model loaded from {path}")


def train_and_evaluate():
    """Main training and evaluation function."""
    logger.info("=" * 80)
    logger.info("TRANSFORMER AUTOENCODER TRAINING")
    logger.info("=" * 80)
    
    # Load data
    seq_path = config.PROCESSED_DATA_DIR / 'sequences.npy'
    labels_path = config.SEQUENCE_LABELS_FILE
    
    if not seq_path.exists():
        logger.error(f"Sequences not found at {seq_path}. Run feature engineering first.")
        return
    
    X = np.load(seq_path)
    y = np.load(labels_path) if labels_path.exists() else None
    
    logger.info(f"Loaded sequences: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train only on normal sequences
    X_train_normal = X_train[y_train == 0]
    
    # IMPORTANT: Sample for CPU-feasible training
    # Transformers are compute-intensive; full 1.8M samples would take hours on CPU
    MAX_TRAIN_SAMPLES = 30000  # ~3 minutes training time on MacBook
    if len(X_train_normal) > MAX_TRAIN_SAMPLES:
        logger.info(f"Sampling {MAX_TRAIN_SAMPLES} from {len(X_train_normal)} for CPU-feasible training...")
        indices = np.random.choice(len(X_train_normal), MAX_TRAIN_SAMPLES, replace=False)
        X_train_normal = X_train_normal[indices]
    
    X_train_normal, X_val = train_test_split(X_train_normal, test_size=0.1, random_state=42)
    
    logger.info(f"Training on {len(X_train_normal)} normal sequences")
    logger.info(f"Validation: {len(X_val)} sequences")
    logger.info(f"Test: {len(X_test)} sequences ({y_test.sum()} anomalies)")
    
    # Build and train model
    model = TransformerAutoencoder(
        seq_length=X.shape[1],
        n_features=X.shape[2],
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=2
    )
    model.build_model()
    model.model.summary()
    
    # Get config epochs
    epochs = getattr(config, 'TRANSFORMER_EPOCHS', 20)
    # Change batch_size=512 to 32 or 64
    model.train(X_train_normal, X_val, epochs=epochs, batch_size=64)  # Larger batch for speed
    
    # Evaluate
    predictions, scores = model.detect_anomalies(X_test)
    
    if y_test is not None and y_test.sum() > 0:
        auc = roc_auc_score(y_test, scores)
        logger.info("\n" + "=" * 60)
        logger.info("TRANSFORMER AUTOENCODER PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info(f"Predictions: {predictions.sum()} anomalies")
        logger.info(f"True Positives: {(predictions & y_test).sum()} / {y_test.sum()}")
        
        # Save results
        results_df = pd.DataFrame({
            'anomaly_score': scores,
            'prediction': predictions,
            'true_label': y_test
        })
        results_path = config.RESULTS_DIR / 'transformer_predictions.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Predictions saved to {results_path}")
    
    # Save model
    model_path = config.MODELS_DIR / 'transformer_autoencoder.keras'
    model.save(model_path)
    
    return model


if __name__ == "__main__":
    train_and_evaluate()
